"""Listen to Sarvam voices next to Luna's own (Piper) voice and pick the one she
uses for non-English replies. Saves the choice as SARVAM_SPEAKER (and
SARVAM_PACE) in the git-ignored .env file.

    venv\\Scripts\\python.exe scripts\\pick_sarvam_voice.py            # closest few
    venv\\Scripts\\python.exe scripts\\pick_sarvam_voice.py --all      # all female voices
"""

import sys

import numpy as np
import sounddevice as sd
from piper import PiperVoice

import sarvam

PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG = "models/piper/en_US-lessac-medium.onnx.json"

ENGLISH_TEXT = "Hi, I'm Luna, the robot of the Computer Science department at TKM College of Engineering."
MALAYALAM_TEXT = ("ഹായ്, ഞാൻ ലൂണ. ടി.കെ.എം. കോളേജ് ഓഫ് എഞ്ചിനീയറിംഗിലെ "
                  "കമ്പ്യൂട്ടർ സയൻസ് ഡിപ്പാർട്ട്മെന്റിന്റെ റോബോട്ട് ആണ് ഞാൻ.")

# Closest to Luna's Piper voice by measured pitch, intonation and speed
# (Piper lessac: ~188 Hz). neha's pitch is closest but she speaks ~40% slower,
# so she is tried sped up.
CLOSEST = [("simran", 1.0), ("priya", 1.0), ("kavya", 1.0), ("neha", 1.35)]


def play(pcm, rate):
    sd.play(pcm, rate)
    sd.wait()


def play_luna():
    voice = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)
    for chunk in voice.synthesize(ENGLISH_TEXT):
        play(chunk.audio_int16_array, chunk.sample_rate)


def save_choice(speaker, pace):
    lines = []
    if sarvam.ENV_FILE.is_file():
        lines = [l for l in sarvam.ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
                 if not l.strip().startswith(("SARVAM_SPEAKER=", "SARVAM_PACE="))]
    lines += [f"SARVAM_SPEAKER={speaker}", f"SARVAM_PACE={pace}"]
    sarvam.ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = sarvam.load_api_key()
    if not key:
        sys.exit(r"No Sarvam key yet — run venv\Scripts\python.exe scripts\set_sarvam_key.py first.")
    candidates = ([(s, 1.0) for s in sarvam.FEMALE_SPEAKERS] if "--all" in sys.argv
                  else CLOSEST)
    client = sarvam.SarvamClient(key)

    print("Generating samples...")
    samples = {}
    for speaker, pace in candidates:
        client.speaker, client.pace = speaker, pace
        try:
            samples[speaker] = (pace, client.synthesize(ENGLISH_TEXT, "en-IN"),
                                client.synthesize(MALAYALAM_TEXT, "ml-IN"))
        except sarvam.SarvamError as e:
            print(f"  {speaker}: {e}")

    while True:
        print("\nLuna's own voice (Piper):")
        play_luna()
        for i, (speaker, (pace, english, malayalam)) in enumerate(samples.items(), 1):
            print(f"{i}. {speaker}" + (f" (pace {pace})" if pace != 1.0 else ""))
            play(*english)
            play(*malayalam)
        names = list(samples)
        answer = input(f"\nType a number 1-{len(names)} to choose, Enter to hear them again, "
                       "or q to quit: ").strip().lower()
        if answer == "q":
            return
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            speaker = names[int(answer) - 1]
            save_choice(speaker, samples[speaker][0])
            print(f"Saved: Luna will speak other languages with '{speaker}'.")
            return


if __name__ == "__main__":
    main()
