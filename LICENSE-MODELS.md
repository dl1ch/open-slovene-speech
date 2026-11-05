# Licence for the model weights and lexicon

The files distributed as **GitHub Release assets** — the TTS voice, the
speech-recognition model, the LoRA adapter, and the bundled pronunciation
lexicon — are licensed under

**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
<https://creativecommons.org/licenses/by-sa/4.0/>

This is **not** the same licence as the source code in this repository, which is
MIT. The split exists because the weights derive from corpora that carry
share-alike terms.

## Why

| artefact | derives from | upstream licence |
|---|---|---|
| speech-recognition model, LoRA adapter | ARTUR 1.0 (audio) | CC BY-SA 4.0 |
| TTS voice | `artur_studio_tts`, a subset of ARTUR 1.0 | CC BY-SA 4.0 (see note) |
| pronunciation lexicon (`sl_pronunciation.tsv.xz`) | Sloleks 3.1 | CC BY-SA 4.0 |

The lexicon is the clearest case: it is a direct transformation of Sloleks 3.1
into a lookup table, so it is unambiguously a derivative work and share-alike
plainly applies.

The weights are a harder question. Whether trained model parameters constitute a
derivative work of their training data is **genuinely unsettled** in law, and
reasonable people disagree. This project does not try to settle it. It applies
share-alike to the weights because that is the reading most consistent with the
terms the corpora were published under and with the intent behind publishing
them openly. Attribution is required under every reading of those licences.

Nothing here is legal advice. If your use is commercially sensitive, get your
own.

### Note on `artur_studio_tts`

The Hugging Face dataset card for `ppisljar/artur_studio_tts` is tagged
`cc-by-3.0`. That dataset is a derivative of ARTUR 1.0, which is CC BY-SA 4.0,
and a redistributor cannot remove the share-alike condition of the work they
are redistributing. This project therefore treats that data — and the voice
trained on it — as CC BY-SA 4.0.

## What you must do

Using these models commercially is fine. Modifying them is fine. Redistributing
them is fine. In exchange:

1. **Credit the corpus authors.** The full citations are in [NOTICE](NOTICE) and
   in the README. Crediting this repository alone is not sufficient — the
   corpora are the licensed work.
2. **Share alike.** If you distribute a modified model — fine-tuned, further
   trained, quantised, converted, distilled — license it under CC BY-SA 4.0 too.
3. **Say what you changed**, and keep these notices with the files.
