# Updating a component

The expensive thing here is training. Almost nothing else requires it, and the
parts most likely to change — the lexicon, the phonemizer, the runtimes — are
all on the cheap side of the line.

## What costs what

| component updates | retrain? | cost | what to do |
|---|---|---|---|
| **Sloleks** (pronunciations) | no | minutes | regenerate the table, check it, rebuild the bundle |
| **user_lexicon.tsv** (your fixes) | no | seconds | edit the file — this is what it is for |
| **english_words.txt** | no | seconds | replace the list |
| **espeak-ng** | no | minutes | verify the OOV rewrite still maps cleanly |
| **onnxruntime / numpy** | no | — | version bump only |
| **faster-whisper / CTranslate2** | no | ~10 min | re-convert from the published LoRA adapter |
| **CTranslate2 model format** | no | ~10 min | same — re-convert, do not retrain |
| **piper / VITS export format** | no | minutes | re-export from the checkpoint |
| **ARTUR** (new corpus release) | **yes** | ~22 GPU-h (ASR) | full LoRA fine-tune |
| **whisper base model** (a v4) | **yes** | ~22 GPU-h | the adapter is tied to the base weights |
| **the voice itself** (multi-speaker, more phonemes) | **yes** | days | see the roadmap |

The reason the middle rows are cheap is that **the model consumes phoneme IDs,
not text**. Everything that turns text into those IDs — lexicon, rewrites,
espeak fallback, English detection — sits outside the network and can be
replaced without touching it.

The reason re-conversion never needs retraining is that the **LoRA adapter is
published** alongside the converted model. Any future runtime format can be
produced from it.

## Updating the lexicon

The common case, and the one with a trap.

```bash
# 1. regenerate the lookup table from the new Sloleks release
#    (form<TAB>IPA, one per line)

# 2. CHECK IT BEFORE SHIPPING  <-- do not skip this
tools/check_lexicon.py new_lexicon.tsv

# 3. rebuild the bundle
LEX_SRC=/path/to/lexicon tools/build_portable.py
```

### Why step 2 exists

A new lexicon can transcribe a word with a symbol the voice has never produced.
Over the entire training set — 22,788 utterances — the model emitted exactly
**46 distinct symbols**, while its config declares 166. The gap is where the
trouble lives, and it fails in two different silent ways:

| symbol is… | result |
|---|---|
| in the id map but never trained | an embedding that learned nothing → **artefact** |
| absent from the id map | **dropped**, the sound simply vanishes |

Neither raises an error. Neither shows up in a loss curve. And a defect
affecting a few thousand words out of 2.9 million will not be found by
listening.

`tools/check_lexicon.py` compares a lexicon against `tts/trained_phonemes.tsv`,
the inventory extracted from the actual training cache, after applying the
phonemizer's own rewrite table. It exits non-zero on a problem, so it works as
a release gate.

### What the current lexicon already shows

This is not hypothetical. Running the checker on the shipped Sloleks 3.1 table:

```
y   U+0079   106,417 entries   aaliyah, aaliyaha, aaliyahe
q   U+0071    13,709 entries   abouyaaqoub, ...
ö   U+00F6     6,178 entries   ackeström, ...
```

`Aaliyah` phonemises to `aːliyax` — the `y` is in piper's id map, so it is not
dropped, but the model never trained it. Roughly 5% of lexicon entries contain
at least one such symbol, almost all of them foreign proper nouns. That is a
known limitation of the voice as shipped, and lifting it means retraining with a
wider phoneme inventory (roadmap 2.2), not swapping a lexicon.

## Updating a Python dependency

This project tracks current versions rather than the widest compatibility, so a
dependency bump can raise the Python minimum. That is acceptable only if the
bundle still behaves identically. Verify before merging:

```bash
tools/verify_bundle.py /path/to/slovene-voice
```

It checks that the lexicon and English list load with the expected counts, that
phonemisation is **byte-identical** to known-good output, and that synthesis
produces plausible audio. Exit status is non-zero on a regression.

Phonemes are the right thing to compare. Audio cannot be, because VITS sampling
is stochastic and two runs of identical input always differ.

Verified identical across Python 3.9.6 / 3.12.14 / 3.13.15, onnxruntime
1.19.2–1.31.1 and numpy 2.0.2–2.5.4.

## Updating espeak-ng

espeak handles only out-of-vocabulary words, about 1.34% of running text. Its
output is rewritten into Sloleks notation (`r→ɾ`, `h→x`, `tʃ→ʧ`, `ts→ʦ`, `w→u̯`)
so the model never sees two competing notations for one sound.

A new espeak version can emit a symbol that rewrite table does not cover. Check
after upgrading:

```bash
python3 -c "
from slovene_voice.phonemize import SlovenePhonemizer
p = SlovenePhonemizer()
print(p.phonemize('Zelenskij Ljubljana blockchain Schwarzenegger'))"
```

and compare against `tts/trained_phonemes.tsv`.

## Re-converting the recogniser

If CTranslate2 changes format, or you want a different quantisation, start from
the published adapter rather than from training:

```bash
# merge the LoRA adapter into the base model
python -c "
from transformers import WhisperForConditionalGeneration
from peft import PeftModel
m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
m = PeftModel.from_pretrained(m, 'models/slovene-whisper-lora').merge_and_unload()
m.save_pretrained('merged')"

ct2-transformers-converter --model merged --output_dir slovene-whisper-ct2 \
    --quantization int8_float16 --copy_files tokenizer.json preprocessor_config.json
```

`--copy_files` matters: without `tokenizer.json`, faster-whisper silently falls
back to downloading `whisper-tiny`'s tokenizer, and without
`preprocessor_config.json` it assumes 80 mel bins where large-v3 uses 128.

## What a retrain actually costs

Both models were trained on a single V100. Neither needs a cluster.

- **ASR**: ~22 GPU-hours per run. The recipe and the three data bugs worth
  avoiding are in [TRAINING.md](TRAINING.md).
- **TTS**: the published checkpoint is epoch 729 of a run holding ~8 epochs/h,
  so a full run is on the order of four days of continuous GPU.

Before retraining the ASR for a new ARTUR release, read the data-bugs section.
All three were invisible in the loss curve and were caught only by per-domain
evaluation.
