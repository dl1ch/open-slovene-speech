#!/usr/bin/env python3
"""Out-of-domain evaluation: both models on FLEURS Slovene.

Every WER in this project is measured on a held-out split of ARTUR, the corpus
the model was fine-tuned on. Stock Whisper has never seen ARTUR, so part of the
reported gap is domain adaptation rather than general Slovene ability. FLEURS
is read speech from Wikipedia sentences: NEITHER model trained on it, which is
the only way to compare them fairly.

    eval_fleurs.py <fleurs_dir> <our_model_dir> [--limit N]

Prints WER for the fine-tune and for stock whisper-large-v3 on the same audio,
with the same decoding settings and the same text normalisation.
"""
import argparse
import csv
import os
import re
import sys
import time
import unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("fleurs")
ap.add_argument("model")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--compute", default="int8_float16")
ap.add_argument("--beam", type=int, default=2)
a = ap.parse_args()

from faster_whisper import WhisperModel
import jiwer

# Same normalisation for both systems and the reference, or the comparison is
# meaningless: lowercase, strip punctuation, collapse whitespace. Slovene
# diacritics are preserved - folding them would hide real errors.
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s):
    s = unicodedata.normalize("NFC", s or "").lower()
    s = PUNCT.sub(" ", s)
    return " ".join(s.split())


rows = []
with open(os.path.join(a.fleurs, "test.tsv"), encoding="utf-8") as f:
    for r in csv.reader(f, delimiter="\t"):
        # id, filename, raw_transcription, transcription, ...
        if len(r) < 4:
            continue
        wav = os.path.join(a.fleurs, "test", r[1])
        if os.path.exists(wav):
            rows.append((wav, r[3] if r[3].strip() else r[2]))
if a.limit:
    rows = rows[:a.limit]
print("  %d FLEURS test utterances" % len(rows), flush=True)
if not rows:
    sys.exit("  no audio found - check the extraction path")


def run(tag, model_id):
    print("\n  === %s" % tag, flush=True)
    t0 = time.time()
    m = WhisperModel(model_id, device=a.device, compute_type=a.compute)
    print("  loaded in %.0fs" % (time.time() - t0), flush=True)
    refs, hyps = [], []
    t0 = time.time()
    for i, (wav, ref) in enumerate(rows, 1):
        segs, _ = m.transcribe(wav, language="sl", beam_size=a.beam)
        hyps.append(norm(" ".join(s.text for s in segs)))
        refs.append(norm(ref))
        if i % 100 == 0:
            print("    %d/%d  %.0fs" % (i, len(rows), time.time() - t0), flush=True)
    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100
    print("  WER %.2f%%   CER %.2f%%   (%.0fs)" % (wer, cer, time.time() - t0), flush=True)
    del m
    return wer, cer, refs, hyps


ours = run("fine-tuned on ARTUR (this project)", a.model)
stock = run("stock whisper-large-v3", "large-v3")

print("\n  " + "=" * 58)
print("  %-38s %8s %8s" % ("FLEURS Slovene (out of domain)", "WER", "CER"))
print("  %-38s %7.2f%% %7.2f%%" % ("stock whisper-large-v3", stock[0], stock[1]))
print("  %-38s %7.2f%% %7.2f%%" % ("this project", ours[0], ours[1]))
rel = (stock[0] - ours[0]) / stock[0] * 100 if stock[0] else 0
print("  %-38s %7.1f%%" % ("relative WER reduction", rel))
print("  " + "=" * 58)
print("\n  For contrast, in-domain on ARTUR: 18.41%% -> 7.77%% (57.8%% reduction)")

with open("/tmp/fleurs_examples.txt", "w", encoding="utf-8") as f:
    for i in range(min(25, len(rows))):
        f.write("REF   %s\nOURS  %s\nSTOCK %s\n\n" % (ours[2][i], ours[3][i], stock[3][i]))
print("  25 side-by-side examples -> /tmp/fleurs_examples.txt")
