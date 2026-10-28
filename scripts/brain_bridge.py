"""Answer questions with Luna 2.0's real pipeline (brain.process from the
sree-realtime checkout), one question per stdin line, one JSON reply per stdout
line prefixed with REPLY_MARKER. Runs under that checkout's own venv, because
its packages (sentence-transformers, numpy 1.26) differ from this repo's
training venv. voice_loop.py starts it as a subprocess; everything else brain
prints is passed through as log output."""
import json
import os
import sys

LUNA_MAIN = os.environ.get("LUNA_MAIN", r"C:\Users\LOQ\luna-brain-main")
REPLY_MARKER = "@@REPLY "

os.chdir(LUNA_MAIN)              # knowledge.txt and data paths are relative
sys.path.insert(0, LUNA_MAIN)

import brain     # noqa: E402
import commands  # noqa: E402
import llm_local  # noqa: E402

_said = []


# "Hmm, let me think." while the model is slow: spoken live by the robot,
# but here it would only be glued onto the front of the reply.
try:
    from config import LLM_FILLERS
except ImportError:
    LLM_FILLERS = ()
_FILLERS = {f.strip().lower() for f in LLM_FILLERS}


def _capture(x, *a, **k):
    text = x if isinstance(x, str) else "".join(x)
    if text.strip().lower() not in _FILLERS:
        _said.append(text)


brain.speak = _capture           # voice_loop.py does the speaking
commands._speak = _capture
llm_local.warmup()
print(REPLY_MARKER + json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    question = line.strip()
    if not question:
        continue
    _said.clear()
    try:
        brain.process(question)
        reply = " ".join(s.strip() for s in _said if s.strip())
    except Exception as e:       # one bad turn must not kill the bridge
        print(f"[bridge] error: {e}", flush=True)
        reply = ""
    print(REPLY_MARKER + json.dumps({"reply": reply or "Sorry, I didn't catch that."}),
          flush=True)
