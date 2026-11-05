#!/usr/bin/env python3
"""Produce a config that lets STOCK piper load this voice. Degraded on purpose.

    tools/make_espeak_config.py model/sl_SI-artur-medium.onnx.json

Writes <name>.espeak.onnx.json next to the input.

WHAT THIS IS FOR

The voice ships with "phoneme_type": "slovene", which upstream piper does not
know: it parses that field through an Enum, so loading raises ValueError and the
voice will not start in wyoming-piper, the Home Assistant Piper add-on, or
openedai-speech.

Sometimes a worse voice inside the stack you already run beats a better voice in
a container you have to add. This rewrites the config to "espeak" so stock piper
will load and speak it.

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
cfg["_degraded"] = {
    "original_phoneme_type": was,
    "warning": ("Stock-piper compatibility config. Phonemised by espeak-ng, NOT "
                "by the Sloleks lexicon this voice was trained on. Expect "
                "misplaced lexical stress in ~74% of word forms and ~6.5% of "
                "phoneme tokens rendered as untrained symbols (notably /r/ and "
                "/h/). For correct pronunciation use the original config with "
                "the slovene_voice phonemizer."),
    "correct_usage": "https://github.com/dl1ch/slovene-speech",
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
