"""Thin client for Sarvam AI's speech-to-text, translation and text-to-speech
APIs (https://docs.sarvam.ai), used by voice_loop.py so Luna can be asked in
any Indian language and answer in that same language.

The API key is read from the SARVAM_API_KEY environment variable, or from a
git-ignored `.env` file in the repo root containing a line

    SARVAM_API_KEY=your-key-here

It is never printed or logged. Sarvam is a cloud service, so all of this needs
internet; voice_loop.py falls back to the offline Whisper + Piper path whenever
these calls fail.
"""

import base64
import io
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import requests

BASE_URL = "https://api.sarvam.ai"
STT_MODEL = "saaras:v4"
TTS_MODEL = "bulbul:v3"
# Voice for non-English replies. Speaker names are per model; set SARVAM_SPEAKER
# in .env (scripts/pick_sarvam_voice.py helps choose one close to Luna's voice).
DEFAULT_SPEAKER = "simran"   # measured closest to Luna's Piper voice
# bulbul:v3's female voices — Luna's English Piper voice is female.
FEMALE_SPEAKERS = ("ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita",
                   "shreya", "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali")
TTS_SAMPLE_RATE = 22050

ENGLISH = "en-IN"
# Languages bulbul can speak. Anything else Sarvam hears gets an English reply.
TTS_LANGUAGES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN",
                 "od-IN", "pa-IN", "ta-IN", "te-IN"}
# mayura:v1 handles these and has a conversational mode, which sounds far more
# natural spoken aloud; sarvam-translate:v1 covers the other scheduled languages.
MAYURA_LANGUAGES = TTS_LANGUAGES

LANGUAGE_NAMES = {"bn-IN": "Bengali", "en-IN": "English", "gu-IN": "Gujarati",
                  "hi-IN": "Hindi", "kn-IN": "Kannada", "ml-IN": "Malayalam",
                  "mr-IN": "Marathi", "od-IN": "Odia", "pa-IN": "Punjabi",
                  "ta-IN": "Tamil", "te-IN": "Telugu", "ur-IN": "Urdu"}

# (connect, read) seconds. A short connect timeout means a dropped network
# costs ~3 s once, not a long hang on every turn.
TIMEOUT = (3, 20)
# After a network failure, don't try Sarvam again for this long.
OFFLINE_COOLDOWN = 60


class SarvamError(Exception):
    pass


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def env_setting(name, default=""):
    """A setting from the environment, else from the git-ignored .env file."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip('"').strip("'") or default
    return default


def load_api_key():
    return env_setting("SARVAM_API_KEY")


def language_name(code):
    return LANGUAGE_NAMES.get(code, code)


def _wav_bytes(audio, sample_rate):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _read_wav(data):
    with wave.open(io.BytesIO(data), "rb") as wf:
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() > 1:
            pcm = pcm.reshape(-1, wf.getnchannels())[:, 0]
    return pcm, rate


class SarvamClient:
    def __init__(self, api_key, keyterms=(), speaker=None):
        self.speaker = speaker or env_setting("SARVAM_SPEAKER", DEFAULT_SPEAKER)
        try:
            self.pace = float(env_setting("SARVAM_PACE", "1.0"))
        except ValueError:
            self.pace = 1.0
        self._session = requests.Session()
        self._session.headers["api-subscription-key"] = api_key
        self._keyterms = json.dumps([t[:64] for t in keyterms][:50]) if keyterms else None
        self._offline_until = 0.0
        self.disabled_reason = None   # set when the key is rejected

    def available(self):
        return self.disabled_reason is None and time.monotonic() >= self._offline_until

    def _post(self, path, **kwargs):
        try:
            resp = self._session.post(BASE_URL + path, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as e:
            self._offline_until = time.monotonic() + OFFLINE_COOLDOWN
            raise SarvamError(f"can't reach Sarvam ({type(e).__name__})") from None
        if resp.status_code in (401, 403):
            self.disabled_reason = "Sarvam rejected the API key (check SARVAM_API_KEY)"
            raise SarvamError(self.disabled_reason)
        if resp.status_code == 429:
            raise SarvamError("Sarvam quota or rate limit reached")
        if resp.status_code >= 400:
            try:
                message = resp.json()["error"]["message"]
            except Exception:
                message = resp.text[:200]
            raise SarvamError(f"Sarvam {path} failed ({resp.status_code}): {message}")
        return resp.json()

    def transcribe(self, audio, sample_rate=16000):
        """Speech -> (transcript, language code such as 'ml-IN'), with the
        language auto-detected."""
        data = {"model": STT_MODEL, "language_code": "unknown"}
        if self._keyterms:
            data["keyterms"] = self._keyterms
        files = {"file": ("speech.wav", _wav_bytes(audio, sample_rate), "audio/wav")}
        result = self._post("/speech-to-text", data=data, files=files)
        return (result.get("transcript") or "").strip(), result.get("language_code") or ENGLISH

    def translate(self, text, source, target):
        if not text or source == target:
            return text
        indic = target if source == ENGLISH else source
        if indic in MAYURA_LANGUAGES:
            body = {"model": "mayura:v1", "mode": "modern-colloquial"}
        else:
            body = {"model": "sarvam-translate:v1"}
        body.update(input=text[:1000], source_language_code=source,
                    target_language_code=target)
        return (self._post("/translate", json=body).get("translated_text") or "").strip()

    def synthesize(self, text, language):
        """Text -> (int16 samples, sample rate)."""
        body = {"text": text[:2500], "language_code": language, "model": TTS_MODEL,
                "speaker": self.speaker, "pace": self.pace, "speech_sample_rate": TTS_SAMPLE_RATE}
        audios = self._post("/text-to-speech", json=body).get("audios") or []
        if not audios:
            raise SarvamError("Sarvam returned no audio")
        parts = [_read_wav(base64.b64decode(a)) for a in audios]
        return np.concatenate([pcm for pcm, _ in parts]), parts[0][1]
