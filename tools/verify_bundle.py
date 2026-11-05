#!/usr/bin/env python3
"""Verify a TTS bundle actually works, before trusting a dependency bump.

    tools/verify_bundle.py /path/to/slovene-voice

This project tracks current versions rather than the widest compatibility, so a
Dependabot PR can raise the project's Python minimum. That is an acceptable
trade only if the bundle still behaves identically afterwards. Run this on the
new stack before merging; exit status is non-zero on any regression.

What it checks, in order of how quietly each would otherwise fail:

1. The lexicon and English word list load, with the expected entry counts. A
   path that silently resolves to nothing loads an empty dict and disables a
   feature without erroring - that has happened twice in this project.
2. Phonemisation is UNCHANGED against known-good output. This is the real
   regression test: audio cannot be compared between runs because VITS
   sampling is stochastic, but phonemes are deterministic.
3. Synthesis produces audio of a plausible duration.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("bundle", nargs="?", default=".")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.bundle))
os.environ.setdefault("SLOVENE_NORMALIZE", "0")

# Verified identical on Python 3.9.6 / 3.12.14 / 3.13.15, and across
# onnxruntime 1.19.2-1.31.1 and numpy 2.0.2-2.5.4.
EXPECTED = {
    "Danes je lep dan.": "ˈdaːnɛs ˈjɛ ˈleːp ˈdaːn.",
    "Govoril je o blockchain tehnologiji.":
        "ɡɔˈʋoːɾiu̯ ˈjɛ ɔ blˈoːktʃɛjn tɛxnɔlɔˈɡiːji.",
}
MIN_LEXICON = 2_900_000
MIN_ENGLISH = 100_000

fail = []


def check(name, ok, detail=""):
    if not a.quiet or not ok:
        print("  %-4s %-34s %s" % ("ok" if ok else "FAIL", name, detail))
    if not ok:
        fail.append(name)


try:
    import numpy
    import onnxruntime
    import platform
    print("  python %s | numpy %s | onnxruntime %s"
          % (platform.python_version(), numpy.__version__, onnxruntime.__version__))
except ImportError as e:
    sys.exit("  cannot import the runtime: %s" % e)

from slovene_voice import SloveneVoice

t0 = time.time()
voice = SloveneVoice()
load = time.time() - t0
ph = voice.phonemizer

check("lexicon loaded", len(ph.lexicon) >= MIN_LEXICON, "%d entries" % len(ph.lexicon))
eng = getattr(ph, "english", None) or []
check("english word list loaded", len(eng) >= MIN_ENGLISH,
      "%d words%s" % (len(eng), "" if eng else "  <-- embedded English is DISABLED"))

for text, want in EXPECTED.items():
    got = "".join("".join(s) for s in ph.phonemize(text))
    check("phonemes unchanged", got == want,
          text[:28] if got == want else "\n       want %s\n       got  %s" % (want, got))

t0 = time.time()
dur = voice.to_wav("Pozdravljen, svet. Danes je lep dan.", os.devnull)
el = time.time() - t0
check("synthesis", 3.0 < dur < 6.0, "%.2fs audio in %.2fs (%.1fx realtime)" % (dur, el, dur / el))
print("  load %.1fs" % load)

if fail:
    print("\n  %d CHECK(S) FAILED: %s" % (len(fail), ", ".join(fail)))
    sys.exit(1)
print("\n  bundle verified on this stack")
