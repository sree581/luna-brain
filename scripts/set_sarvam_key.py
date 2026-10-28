"""Save your Sarvam AI API key into the git-ignored .env file and check it works.

    venv\\Scripts\\python.exe scripts\\set_sarvam_key.py

The key is typed/pasted at a hidden prompt (nothing shows on screen — that's
normal), so it never lands in your shell history or on screen.
"""

import getpass
import sys
from pathlib import Path

import numpy as np

import sarvam

ENV_FILE = sarvam.ENV_FILE


def main():
    key = getpass.getpass("Paste your Sarvam API key (hidden), then press Enter: ").strip()
    if not key:
        sys.exit("No key entered — nothing saved.")

    lines = []
    if ENV_FILE.is_file():
        lines = [l for l in ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
                 if not l.strip().startswith("SARVAM_API_KEY=")]
    lines.append(f"SARVAM_API_KEY={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved to {ENV_FILE} (git-ignored).")

    print("Checking the key with Sarvam...")
    client = sarvam.SarvamClient(key)
    try:
        client.transcribe(np.zeros(16000, dtype=np.float32))
        print("Key works. Luna will now use Sarvam for multilingual speech.")
    except sarvam.SarvamError as e:
        print(f"Saved, but the check failed: {e}")


if __name__ == "__main__":
    main()
