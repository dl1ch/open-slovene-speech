#!/usr/bin/env python3
"""Check a pronunciation lexicon against the symbols the voice was trained on.

    tools/check_lexicon.py <lexicon.tsv[.xz]> [--inventory tts/trained_phonemes.tsv]

WHY THIS EXISTS

The lexicon and the model are updated independently. Sloleks gets a new release
every so often, and swapping it in costs minutes: regenerate the lookup table,
rebuild the bundle, done. No GPU, no retraining.

What that cheap path cannot do is notice that the new lexicon transcribes some
word with a symbol the model has never produced. The config declares 166
symbols; over the whole training set the model emitted exactly 46.

Two different failures follow, neither of which raises an error:

  in the id map, never trained   an embedding that learned nothing -> artefact
  absent from the id map         dropped silently -> the sound just vanishes

The current lexicon already trips the first: "y" appears in 106,417 entries,
nearly all foreign names, and the model has never produced it. So this is not a
hypothetical for some future Sloleks - it is a measurement of what ships today.

So: run this before shipping a lexicon the voice did not train on.

Exit status is 1 if any symbol falls outside the trained inventory, which makes
it usable as a release gate.
"""
import argparse
import collections
import importlib.util
import lzma
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INV = os.path.join(HERE, "..", "tts", "trained_phonemes.tsv")
PHONEMIZE = os.path.join(HERE, "..", "tts", "slovene_voice", "phonemize.py")


def _load_rewriter():
    """Borrow to_supported_inventory() from the phonemizer itself.

    The lexicon stores Sloleks notation, which is NOT what reaches the model:
    six symbols absent from piper's id table are rewritten first, and one of
    them is Slovene c-hacek at ~1% of all phonemes. Checking the raw lexicon
    therefore reports hundreds of thousands of false problems.

    Loaded from the module file directly, bypassing the package __init__, so
    this script needs neither onnxruntime nor numpy. Imported rather than
    copied so the two cannot drift apart.
    """
    spec = importlib.util.spec_from_file_location("_ph", PHONEMIZE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_supported_inventory

ap = argparse.ArgumentParser()
ap.add_argument("lexicon", help="TSV of form<TAB>IPA, optionally .xz")
ap.add_argument("--inventory", default=DEFAULT_INV)
ap.add_argument("--max-examples", type=int, default=6)
# A symbol seen a handful of times in training is not meaningfully trained.
# gamma occurs exactly ONCE in the whole corpus; treating it as supported is
# how a rare artefact gets blessed as fine.
ap.add_argument("--min-trained", type=int, default=50,
                help="symbols rarer than this in training are reported as weak")
a = ap.parse_args()

trained, counts = set(), {}
with open(a.inventory, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.rstrip("\n"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            trained.add(parts[0])
            counts[parts[0]] = int(parts[3])
print("  inventory: %d symbols the model has produced" % len(trained))
rewrite = _load_rewriter()
print("  applying the phonemizer's own symbol rewrites before comparing")

opener = lzma.open if a.lexicon.endswith(".xz") else open
seen = collections.Counter()
examples = collections.defaultdict(list)
rows = 0
with opener(a.lexicon, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        rows += 1
        form, ipa = parts[0], rewrite(parts[1])
        for ch in ipa:
            seen[ch] += 1
            if ch not in trained and len(examples[ch]) < a.max_examples:
                examples[ch].append(form)

print("  lexicon  : %d entries, %d distinct symbols" % (rows, len(seen)))

unknown = {s: n for s, n in seen.items() if s not in trained}
weak = {s: counts.get(s, 0) for s in seen
        if s in trained and counts.get(s, 0) < a.min_trained}

if unknown:
    print("\n  UNTRAINED SYMBOLS - the model has never produced these.")
    print("  Those present in piper's 166-symbol id map get an embedding that was")
    print("  never trained, so they render as an ARTEFACT. Those absent from it are")
    print("  dropped silently. Neither raises an error.")
    print("  %-6s %-10s %-30s %10s  %s" % ("sym", "codepoint", "name", "in lexicon", "examples"))
    for s, n in sorted(unknown.items(), key=lambda x: -x[1]):
        print("  %-6s U+%04X     %-30s %10d  %s"
              % (repr(s)[1:-1], ord(s), unicodedata.name(s, "?")[:30], n,
                 ", ".join(examples[s][:3])))
else:
    print("\n  no untrained symbols - this lexicon is safe for the current voice")

if weak:
    print("\n  WEAKLY TRAINED - present in training, but barely:")
    for s, n in sorted(weak.items(), key=lambda x: x[1]):
        print("  %-6s U+%04X  seen %d time(s) in the whole training set"
              % (repr(s)[1:-1], ord(s), n))

sys.exit(1 if unknown else 0)
