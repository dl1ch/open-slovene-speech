#!/usr/bin/env python3
"""Minimal Slovene transcription example.

    python stt/transcribe.py posnetek.wav
    python stt/transcribe.py posnetek.wav --cpu

The model is whisper-large-v3 fine-tuned on 853 h of ARTUR and converted to
CTranslate2 int8_float16. See docs/MODEL-CARD-STT.md for what it is and is not
good at.
"""
import argparse
import os
import sys

DEFAULT_MODEL = os.environ.get(
    "SLOVENE_STT_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "slovene-whisper-ct2"))

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--cpu", action="store_true", help="run on CPU (int8), no GPU needed")
# beam 2 is the measured knee: 8.19%% WER at 5.3x realtime, against 8.12%% at
# 4.2x for beam 5. Larger beams buy almost nothing on this model.
ap.add_argument("--beam", type=int, default=2)
ap.add_argument("--timestamps", action="store_true")
a = ap.parse_args()

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("faster-whisper is not installed:  pip install faster-whisper")

if not os.path.isdir(a.model):
    sys.exit("model not found at %s\nrun scripts/fetch-models.sh stt" % a.model)

model = WhisperModel(
    a.model,
    device="cpu" if a.cpu else "cuda",
    compute_type="int8" if a.cpu else "int8_float16",
)

# language is pinned: the model is specialised to Slovene, and letting Whisper
# auto-detect on a short or noisy clip is a needless way to get the wrong one.
segments, info = model.transcribe(a.audio, language="sl", beam_size=a.beam)

for s in segments:
    if a.timestamps:
        print("[%7.2f -> %7.2f] %s" % (s.start, s.end, s.text.strip()))
    else:
        print(s.text.strip())
