import re
import os
import json
import queue
import datetime
import subprocess
from pathlib import Path

import numpy as np
import noisereduce as nr
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # required chunk size for Silero VAD at 16kHz

# Multilingual (no ".en" suffix) so Malayalam speech transcribes as Malayalam instead
# of being forced through an English-only decoder, which just produces English-sounding
# gibberish for non-English audio. "small" trades some speed for the accuracy Malayalam
# needs on a CPU-bound target like a Raspberry Pi 5 (swap to "base" if that's too slow).
STT_MODEL = "small"

PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG = "models/piper/en_US-lessac-medium.onnx.json"

# Answers come from Luna 2.0's pipeline (brain.process: commands, knowledge base,
# the "luna" Ollama model, Wikipedia, weather) in the sree-realtime checkout. It
# needs that checkout's own venv, so it runs as a subprocess (scripts/brain_bridge.py).
LUNA_MAIN = Path(os.environ.get("LUNA_MAIN", r"C:\Users\LOQ\luna-brain-main"))
BRIDGE_PYTHON = LUNA_MAIN / "venv" / "Scripts" / "python.exe"
BRIDGE_SCRIPT = Path(__file__).with_name("brain_bridge.py")
REPLY_MARKER = "@@REPLY "

STOP_PHRASES = {"bye", "goodbye", "good bye", "stop", "exit", "quit", "stop listening"}

# Software gain applied to the raw mic signal before VAD/denoise/STT ever see it —
# compensates for a quiet mic/room without needing to touch Windows input settings.
# Raise if it's still missing quieter speech, lower if quiet parts start clipping/distorting.
MIC_GAIN = 2.5

# Noise reduction before Whisper. Set LUNA_DENOISE=0 to compare without it — on a
# quiet room it can remove parts of the voice along with the noise.
DENOISE = os.environ.get("LUNA_DENOISE", "1") != "0"

# The last utterance is saved before and after denoising, so you can listen to
# exactly what Whisper was given when it mishears something.
DEBUG_RAW_WAV = Path("logs/last_heard_raw.wav")
DEBUG_CLEAN_WAV = Path("logs/last_heard_clean.wav")


def save_wav(path, audio):
    import wave
    path.parent.mkdir(exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def report_level(audio):
    """Warn when MIC_GAIN pushes speech into clipping — clipped audio is
    distorted and Whisper mishears it."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipped = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0
    if clipped > 0.005:
        print(f"(mic too loud: {clipped:.1%} of the audio clipped — lower MIC_GAIN)")
    elif peak < 0.1:
        print(f"(mic very quiet: peak {peak:.2f} — raise MIC_GAIN or move closer)")

# Whisper's base.en model reliably mangles names and acronyms it hasn't seen (e.g. it
# heard "TKM" as "P.K.M." and "Aneesh G Nath" as "Anish G. Nuth" in testing). Passing
# the actual vocabulary as an initial_prompt biases the decoder toward these spellings.
DOMAIN_VOCABULARY = (
    "TKM College of Engineering, Thangal Kunju Musaliar, Karikode, Kollam, Kerala, "
    "Kilikollur, Dr. Aneesh G Nath, Dr. Dimple A Shajahan, Dr. Thushara A, "
    "Dr. Manu J Pillai, Dr. Anantha Padmanabhan N K, Dr. Shyna A, Dr. Reshma Sheik, "
    "Reena Mary George, Rehannara Begum T, Dr. Jini Raju, B. Mohammed Siyad, Nisa A K, "
    "Dr. Shameem Ansar, Dr. Sadiq A, NAAC, KIRF, NBA, KTU, CGPU, HOD, Luna, "
    "Michael Harold Sony, Sreenandana, Devadathan, Raspberry Pi."
)

LOG_PATH = Path("logs/voice_session.log")


def log_exchange(heard, reply):
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}]\n")
        f.write(f"You said: {heard}\n")
        f.write(f"Luna: {reply}\n\n")

# Piper's phonemizer regularly mangles acronyms and initialisms it hasn't seen — it
# guesses a "word" pronunciation instead of reading the letters. Spelling them out
# with periods ("T. K. M.") is a reliable way to force letter-by-letter reading.
# This is a starting guess at what should be spelled out — adjust freely, and note
# some (like IEEE) have an idiomatic spoken form ("eye triple ee") rather than a
# strict letter-by-letter one if you want to hand-tune those further.
ACRONYMS = ["TKM", "NAAC", "NBA", "KIRF", "KTU", "CGPU", "HOD", "IEEE", "ISTE", "IEI", "CSI", "LPA",
            "PIN", "IQAC", "AICTE", "MCA", "CSE", "ECE", "EEE", "IEDC", "CCF", "NIRF", "UGC"]
_ACRONYM_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ACRONYMS) + r")\b")


def _spell_out(word):
    return " ".join(f"{c}." for c in word)


def fix_pronunciation(text):
    return _ACRONYM_RE.sub(lambda m: _spell_out(m.group(0)), text)


def is_stop_phrase(text):
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    words = normalized.split()
    if not words:
        return False
    # Treat it as a stop command only if the whole utterance is short and IS a stop
    # phrase, so "why did the bus stop" doesn't accidentally end the program.
    return normalized in STOP_PHRASES or (len(words) <= 3 and words[-1] in {"bye", "stop", "exit", "quit"})


class BrainBridge:
    """One long-lived brain_bridge.py subprocess: a question per line in, a
    REPLY_MARKER-prefixed JSON line out. Its other output (brain's own logs)
    is echoed so you can see which path answered."""

    def __init__(self):
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [str(BRIDGE_PYTHON), "-u", str(BRIDGE_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, bufsize=1,
        )
        self._read_reply()           # blocks until brain + the LLM are loaded

    def _read_reply(self):
        for line in self.proc.stdout:
            if line.startswith(REPLY_MARKER):
                return json.loads(line[len(REPLY_MARKER):])
            print(f"  | {line.rstrip()}")
        raise RuntimeError("Luna's brain process exited — see the log lines above.")

    def ask(self, text):
        self.proc.stdin.write(text.replace("\n", " ") + "\n")
        self.proc.stdin.flush()
        return self._read_reply()["reply"]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# Only these languages are allowed. Free auto-detection mislabels Indian-accented
# English as Arabic, Tamil or Urdu (seen in testing) and then transcribes it in
# that script; restricting the choice to English or Malayalam keeps both working.
ALLOWED_LANGUAGES = ("en", "ml")


def transcribe(stt, audio):
    segments, info = stt.transcribe(audio, initial_prompt=DOMAIN_VOCABULARY)
    if info.language not in ALLOWED_LANGUAGES:
        # segments is a lazy generator, so discarding it costs nothing.
        probs = dict(info.all_language_probs or [])
        language = max(ALLOWED_LANGUAGES, key=lambda lang: probs.get(lang, 0.0))
        segments, info = stt.transcribe(audio, language=language,
                                        initial_prompt=DOMAIN_VOCABULARY)
    return " ".join(seg.text for seg in segments).strip()


def speak(tts, text):
    for chunk in tts.synthesize(fix_pronunciation(text)):
        sd.play(chunk.audio_int16_array, chunk.sample_rate)
        sd.wait()


def listen_for_utterance(audio_queue, vad_iterator, max_seconds=20):
    """Blocks until it captures one spoken utterance (speech start -> sustained
    silence), using Silero VAD to find the boundaries. Returns a float32 numpy array."""
    buffer = []
    triggered = False
    max_chunks = int(max_seconds * SAMPLE_RATE / VAD_CHUNK_SAMPLES)

    while True:
        chunk = audio_queue.get()
        event = vad_iterator(chunk, return_seconds=False)

        if event and "start" in event and not triggered:
            triggered = True
            buffer = [chunk]
        elif triggered:
            buffer.append(chunk)
            if (event and "end" in event) or len(buffer) >= max_chunks:
                vad_iterator.reset_states()
                return np.concatenate(buffer)


def drain_queue(audio_queue):
    try:
        while True:
            audio_queue.get_nowait()
    except queue.Empty:
        pass


def main():
    print("Loading Luna's brain (knowledge base + luna model)...")
    brain = BrainBridge()

    print("Loading speech-to-text model...")
    stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")

    print("Loading text-to-speech voice...")
    tts = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)

    print("Loading voice activity detector...")
    vad_model = load_silero_vad()
    vad_iterator = VADIterator(
        vad_model, sampling_rate=SAMPLE_RATE, threshold=0.3,
        min_silence_duration_ms=600, speech_pad_ms=300,
    )

    audio_queue = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        boosted = np.clip(indata[:, 0] * MIC_GAIN, -1.0, 1.0)
        audio_queue.put(boosted.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=VAD_CHUNK_SAMPLES, callback=callback,
    )

    print(f"Noise reduction: {'on' if DENOISE else 'off'}, mic gain: {MIC_GAIN}")
    print("Luna voice loop ready. Speak any time; say 'bye' or 'stop' to exit.\n")
    try:
        with stream:
            speak(tts, "Hi, I'm listening.")
            # The mic picks up Luna's own greeting through the speakers; without this,
            # that echo gets transcribed as the first "utterance" from the user.
            drain_queue(audio_queue)
            vad_iterator.reset_states()
            while True:
                audio = listen_for_utterance(audio_queue, vad_iterator)

                # Denoise before transcription — helps separate the speaker from steady
                # background noise (fans, hum, room tone). prop_decrease < 1.0 keeps this
                # gentle so it doesn't eat quieter speech along with the noise.
                report_level(audio)
                save_wav(DEBUG_RAW_WAV, audio)
                if DENOISE:
                    try:
                        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=False, prop_decrease=0.6)
                    except Exception as e:
                        print(f"(denoise skipped: {e})")
                save_wav(DEBUG_CLEAN_WAV, audio)

                heard = transcribe(stt, audio)

                # Discard whatever the mic picked up while Luna was talking, and reset
                # VAD state, so playback audio can't be misread as the next utterance.
                drain_queue(audio_queue)
                vad_iterator.reset_states()

                if not heard:
                    continue

                print(f"You said: {heard}")

                if is_stop_phrase(heard):
                    log_exchange(heard, "(stop phrase — session ended)")
                    speak(tts, "Goodbye.")
                    drain_queue(audio_queue)
                    break

                reply = brain.ask(heard)
                print(f"Luna: {reply}\n")
                log_exchange(heard, reply)
                speak(tts, reply)
                drain_queue(audio_queue)
                vad_iterator.reset_states()
    finally:
        brain.close()


if __name__ == "__main__":
    main()
