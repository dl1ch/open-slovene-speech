#!/usr/bin/env python3
"""Rewrite the voice config to espeak mode, so OFFICIAL piper can load it.

    tools/make_espeak_config.py model/sl_SI-artur-sloleks-medium.onnx.json

Writes <name>.espeak.onnx.json next to the input.

WHAT THIS IS FOR

The voice ships with "phoneme_type": "slovene", which upstream piper does not
know: it parses that field through an Enum, so loading raises ValueError in
wyoming-piper, the Home Assistant Piper add-on, or openedai-speech. This rewrites
it to "espeak" so piper loads it.

The SAME config has two very different outcomes depending on what feeds piper:

  WITH the phonemizer proxy (the light stack): the proxy sends text already
  converted to [[ raw phonemes ]] from the Sloleks lexicon, piper takes its
  raw-phoneme branch, espeak never runs, and the result is CORRECT. This is the
  config the deployment guide installs for the official piper container.

  WITHOUT the proxy (stock piper fed plain text): piper falls back to espeak's
  Slovene G2P, which mis-stresses ~74% of word forms. Usable in a pinch, wrong.

So this is not "degraded on purpose" -- it is the espeak-mode config the light
stack REQUIRES, which is only degraded when used without the lexicon in front.

WHAT YOU GIVE UP - MEASURED, NOT ESTIMATED

1. STRESS. Slovene lexical stress is not predictable from spelling. Against
   Sloleks 3.1 over 4,000 word forms, espeak-ng places it on the wrong syllable
   in about 74% of them. This is the whole reason the lexicon phonemizer exists.

2. UNTRAINED SYMBOLS. espeak's Slovene inventory is not the notation the model
   trained on. Measured over ordinary Slovene text, 6.5% of phoneme tokens come
   out as symbols the model has NEVER produced in training:

       r   espeak's trill, where the model trained the tap ɾ   - very frequent
       h   where the model trained x
       ˌ   secondary stress, which Sloleks does not mark
       ɲ   where the model trained the sequence nj

   These are present in piper's id map, so they are not dropped. They get an
   embedding that learned nothing, and render as an artefact. /r/ is common in
   Slovene, so this is audible.

3. REPRESENTATION DRIFT. Even where symbols are individually valid, espeak
   writes some sounds differently - "ts" as two characters where the model
   trained the single ʦ - so the sequence differs from anything seen in
   training.

Use the servers in serve/ if you can. Use this if you cannot, and know what it
costs.
"""
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit(__doc__.strip().splitlines()[2].strip())

src = sys.argv[1]
cfg = json.load(open(src, encoding="utf-8"))

was = cfg.get("phoneme_type")
cfg["phoneme_type"] = "espeak"
cfg.setdefault("espeak", {}).setdefault("voice", "sl")

# A marker so nobody later wonders why a voice sounds off. Piper ignores keys
# it does not know, so this travels with the file harmlessly.
cfg["_espeak_mode"] = {
    "original_phoneme_type": was,
    "note": ("espeak-mode config for the official piper container. CORRECT when "
             "fed [[ raw phonemes ]] by the Slovene phonemizer proxy (the light "
             "stack); if fed plain text it falls back to espeak and mis-stresses "
             "~74% of word forms. See the deployment guide."),
    "project": "https://github.com/dl1ch/open-slovene-speech",
}

base = src[:-len(".onnx.json")] if src.endswith(".onnx.json") else os.path.splitext(src)[0]
dst = base + ".espeak.onnx.json"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("  wrote %s" % dst)
print("  phoneme_type: %s -> espeak   (espeak voice: %s)"
      % (was, cfg["espeak"]["voice"]))
print()
print("  Stock piper will now load this voice. It will also mis-stress roughly")
print("  three quarters of Slovene word forms and render ~6.5% of phonemes as")
print("  untrained symbols. That is the trade, and nothing will warn you at")
print("  runtime - it simply sounds wrong to a Slovene speaker.")
