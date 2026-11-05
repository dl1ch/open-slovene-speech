# Slovene Speech

Open speech models for Slovene: a **text-to-speech voice** and a **speech
recognition model**, both trained on publicly available Slovene corpora and both
runnable on ordinary hardware.

| | model | result |
|---|---|---|
| **STT** | Whisper large-v3 + LoRA, fine-tuned on 853 h of ARTUR | **7.77% WER** (18.41% for stock Whisper) |
| **TTS** | Piper/VITS, 22.05 kHz, single speaker, lexicon-based phonemizer | 4–41× realtime on CPU, no GPU needed |

Slovene is a small language with excellent public language resources and almost
no ready-to-run models built from them. The corpora below already existed; this
repository is the training recipe, the inference code, and the resulting
weights, released under terms compatible with the data they came from.

**They are meant to slot into whatever you already run.** The recogniser is an
ordinary Whisper model — point faster-whisper, WhisperX, Speaches or the Home
Assistant Whisper add-on at it and it works. For the voice, and for Home
Assistant's Wyoming protocol, this repository ships the servers:
`docker compose up` gives you an OpenAI-compatible endpoint on `:8080` and a
Wyoming endpoint on `:10200`. See [Deploying](#deploying).

---

## Quick start

### Text to speech

```bash
pip install -r tts/requirements.txt       # onnxruntime + numpy
# download the model + lexicon (see Releases)
python -m slovene_voice "Pozdravljen, svet." out.wav
```

```python
from slovene_voice import SloveneVoice
v = SloveneVoice()
v.to_wav("V Ljubljani je triindvajset stopinj.", "out.wav")
```

No piper, no GPU, no special distribution. Python 3.9+ and onnxruntime are
the whole runtime.
Install `espeak-ng` as well — see [Why the phonemizer matters](#why-the-phonemizer-matters).

> **The two halves have different Python floors.** The portable voice runs on
> **3.9+** and is verified there. The serving stack in `serve/` needs **3.11+**,
> because onnxruntime 1.31 and python-multipart 0.0.33 require it. If you only
> want the voice as a library, the older floor is the one that applies.

### Speech to text

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel
m = WhisperModel("path/to/slovene-whisper-ct2", device="cuda", compute_type="int8_float16")
segments, info = m.transcribe("posnetek.wav", beam_size=2)
print(" ".join(s.text for s in segments))
```

1.5 GB on disk, ~2 GB VRAM. `beam_size=2` is the measured knee: 8.19% WER at
5.3× realtime, against 8.12% at 4.2× for beam 5 and 8.54% at 6.7× for greedy.

---

## System requirements

Both models are deliberately modest. The TTS runs on a Raspberry Pi; the STT is
the one that wants a GPU.

### Text to speech — CPU only

| | minimum | notes |
|---|---|---|
| CPU | any x86-64 or arm64 | no AVX-512, no GPU, no NPU |
| RAM | **1.5 GB free** | measured peak 1.06 GB — see below |
| disk | **80 MB** | the extracted bundle is 74 MB |
| Python | **3.9+** | `onnxruntime` + `numpy` are the only hard deps; verified on 3.9.6 |
| optional | `espeak-ng` | strongly recommended — without it OOV words are dropped |

**Memory is the real constraint, not disk.** Measured peak RSS on macOS:

```
baseline interpreter         8 MB
after import                32 MB
after loading the voice   1017 MB   (2.3 s)
after synthesising        1059 MB
```

Almost all of that is the 2,910,359-entry pronunciation lexicon held as a Python
dict. The `.tsv.xz` is read straight into memory and never decompressed to disk,
so the cost is RAM, not storage. A 1 GB machine will not fit it; 2 GB is
comfortable.

Speed varies a lot with the CPU. Measured: **~4× realtime** on the reference
server (an older x86-64 core, which is where the conservative figure in this
README comes from) and **41× realtime** on an Apple Silicon Mac — 3.96 s of
audio in 0.10 s. Assume the low end when sizing; you will probably do better.
A Raspberry Pi 5 is within range; a Pi 4 has not been tested.

Construction decompresses the lexicon and costs **2–3 seconds**, so build one
`SloveneVoice` and keep it. Do not construct one per utterance.

### Speech to text

**GPU path (recommended)** — CTranslate2 `int8_float16`:

| | minimum | notes |
|---|---|---|
| GPU | NVIDIA, **compute capability ≥ 7.0** | `int8_float16` needs Volta or newer; measured on a V100 |
| VRAM | **~2 GB** | model is 1.5 GB on disk |
| RAM | 4 GB | |
| disk | **2 GB** | |
| CUDA | cuBLAS + cuDNN | shipped in the `nvidia-*` pip wheels; no system CUDA needed |

Throughput at `beam_size=2`: **5.3× realtime**. Larger beams buy almost nothing
here — 8.12% WER at beam 5 against 8.19% at beam 2, for a fifth less speed.

**CPU path** — works, with `compute_type="int8"`, and needs no GPU at all:

| | minimum |
|---|---|
| CPU | x86-64 with AVX2, 4+ cores recommended |
| RAM | **4 GB** |
| disk | 2 GB |

Expect roughly realtime or slower. Fine for batch transcription, not for
streaming.

**Fine-tuning or running the LoRA adapter under `transformers`** is a different
budget: whisper-large-v3 is 1.55 B parameters, so inference in fp16 needs about
**4 GB VRAM** and the training run described in `docs/TRAINING.md` used a
16 GB V100 for ~22 GPU-hours per epoch pair.

### Optional: the text normalisation service

Needed only if you want digits and dates spoken correctly (`27. 9. 2026` →
*sedemindvajsetega devetega …*). It is heavy relative to everything else:
**~5 GB disk** and **~2 GB RAM**, because `classla` pulls in torch and neural
models, and it takes about **30 seconds to start**. This is exactly why it is
not part of the portable bundle.

> **Which of these are measured.** The STT VRAM, WER, beam trade-off and the
> TTS realtime factor are measured on the reference machine (NVIDIA V100,
> x86-64). The RAM and disk minimums are working figures rather than tight
> bounds — if you run close to them, measure on your own hardware.

---

## Results

### Speech recognition

Fine-tuned on 853 h of ARTUR. Evaluated on a **speaker-disjoint** test set of
1,000 utterances, 250 per domain:

| domain | stock Whisper large-v3 | this model |
|---|---|---|
| Artur-B — read speech | 12.09% | **4.16%** |
| Artur-N — private conversation | 17.50% | **5.68%** |
| Artur-J — public speech | 22.44% | **11.28%** |
| Artur-P — parliamentary | 25.84% | **11.56%** |
| **overall WER** | 18.41% | **7.77%** |
| **overall CER** | 6.82% | **2.41%** |

The deployed CTranslate2 `int8_float16` conversion measures 7.26% WER at beam 5;
quantisation did not cost accuracy on this test set.

### Text to speech

There is no honest single number for TTS quality, and this is worth stating
plainly: **round-trip WER is blind to the main failure mode.** Whisper
recognises a word whether or not its stress is correct, so a voice can sit at
the human WER floor while mis-stressing three quarters of what it says. The
voice here was judged by a native speaker; the training objective
(`val_mel` 0.3784 at epoch 729) is reported for reproducibility, not as a
quality claim.

---

## Why the phonemizer matters

This is the part worth reading even if you never run the model.

**Slovene lexical stress is not predictable from spelling.** It is a property of
the word, not of its letters, so a rule-based grapheme-to-phoneme converter
cannot get it right in principle. Measured against Sloleks 3.1 over 4,000 word
forms, **espeak-ng places the stress on the wrong syllable in ~74% of them** —
*ahniti* comes out *ah-NI-ti* instead of *AH-ni-ti*.

So this voice does not use rules. It looks each word up in **Sloleks 3.1**,
which ships a reference IPA transcription for 2.9 million word forms, and falls
back to espeak-ng only for what is missing — about 1.34% of real text.

Two consequences that trip people up:

**Shipping the `.onnx` alone is useless.** The model's input is phoneme IDs, not
text. The text→IDs half is the entire problem, and for this voice it is not
espeak. Feed it espeak phonemes and it still speaks, fluently, with most of the
stress wrong — and **nothing errors**. That is an easy regression to ship by
accident.

**espeak-ng is recommended, not optional.** Without it, out-of-vocabulary words
are *silently dropped* rather than mispronounced:

```
text           : Predsednik Zelenskij je obiskal Ljubljano.
with espeak    : Predsednik zelenski je obiskal ljubljano.
without espeak : Predsednik je obiskal ljubljano.
                            ^^^^^^^^^ the name is simply gone
```

`apt install espeak-ng` · `brew install espeak-ng` · `nix-shell -p espeak-ng`

---

## Limitations

Stated up front, because each one is a real boundary and not a to-do list.

- **One voice, one speaker.** The TTS is a single-speaker model from a 40 h
  studio subset. There is no speaker selection and no voice cloning.
- **Digits are not expanded.** ARTUR's training text contains no digits at all,
  so the model has never seen one. `27. 9. 2026` needs a normaliser to become
  *sedemindvajsetega devetega dva tisoč šestindvajset*, with case agreement. A
  normalisation service based on RSDO-DS2-NORM is included under `deploy/`, but
  it pulls in `classla` (~5 GB), so it is not part of the portable bundle.
- **English inside Slovene is Slovene-accented by necessity**, and detecting it
  at all requires `lexicon/english_words.txt` alongside the voice — without
  that file the detection loads empty and every English word falls through to
  Slovene letter-to-sound, with no error. Over all 22,768
  training utterances the model produced exactly **46 distinct phonemes**; 120
  of the 166 entries in its `phoneme_id_map` were never trained, and every core
  English vowel and the English rhotic are in the untrained set
  (ɹ æ ʊ ɚ ɑ ʌ ɜ ɐ θ ð). Feeding an untrained phoneme produces an artefact, not
  an accent, so English is mapped onto Slovene phones — *think→tink*,
  *better→beter* — which is also what a Slovene speaker actually produces.
- **Conversational and parliamentary ASR is roughly twice as hard as read
  speech** (11.3–11.6% vs 4.2% WER). The overall figure is an average over four
  very different domains; quote the per-domain number that matches your use.
- **Homographs are handled by lexicon lookup, not by context** in the portable
  bundle. They are 0.7% of forms, and 87% are resolvable by morphosyntactic tag,
  which the full deployment does via CLASSLA.

---

## What is in this repository

**No model weights.** The tree is ~500 KB of code and documentation. The weights
are Release assets, because the recogniser alone is 1.5 GB and the lexicon is
81 MB uncompressed.

```
tts/slovene_voice/   the TTS runtime: text -> phonemes -> ids -> audio
stt/transcribe.py    a minimal recogniser example
serve/openai_api.py  OpenAI-compatible API -> Open WebUI, LibreChat, ...
serve/wyoming_server.py   Wyoming protocol -> Home Assistant
Dockerfile           }  docker compose up api      (HTTP, port 8080)
docker-compose.yml   }  docker compose up wyoming  (HA, port 10200)
scripts/             fetch the weights from the Release
tools/build_portable.py   rebuild the release bundle (reference host only)
tools/check_lexicon.py    validate a lexicon against the voice's trained symbols
tools/make_espeak_config.py  config for stock piper (degraded — see DEPLOYMENT)
tts/trained_phonemes.tsv  the 46 symbols the model has actually produced
docs/                model cards, training recipe, deployment, roadmap
```

### Why is there code at all, if these are models?

For the recogniser, there barely is: it is a standard faster-whisper model and
`stt/transcribe.py` is a convenience, not a requirement. Point any Whisper
runtime at the weights and it works.

For the voice, the code **is half the model.** The `.onnx` file does not take
text — its input is phoneme IDs:

```
input          int64  [batch, phonemes]
input_lengths  int64  [batch]
scales         float  [3]
```

Something has to turn Slovene text into those IDs, and for this voice that is
not espeak: the model was trained on lexicon-derived pronunciations, because
espeak stresses ~74% of Slovene word forms wrongly. Pair the `.onnx` with an
espeak front-end and you get fluent speech with the stress mostly wrong, and no
error to tell you. `tts/slovene_voice/` is the matching front-end, and the
model is unusable without it.

That is also why the bundle drops piper entirely rather than forking it: the
`PhonemeType.SLOVENE` it would need does not exist upstream, so a stock piper —
or a stock `openedai-speech` container — cannot serve this voice.

---

## Deploying

Full guide: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

```bash
scripts/fetch-models.sh
docker compose up -d api        # OpenAI-compatible API on :8080
docker compose up -d wyoming    # Wyoming for Home Assistant on :10200
```

**Home Assistant** — add the Wyoming integration pointing at port 10200; a
Slovene STT and a Slovene TTS service appear for Assist pipelines.

**Open WebUI** — Settings → Audio, set both engines to OpenAI with base URL
`http://<host>:8080/v1`, models `slovene-stt` and `slovene-tts`.

**Any Whisper framework** — the recogniser is an ordinary CTranslate2 Whisper
model. faster-whisper, WhisperX, Speaches and the Home Assistant Whisper add-on
all take it as-is; a LoRA adapter is published for `transformers`/PEFT.

**The recogniser needs no server from here at all.** Mount the CTranslate2
directory into the official Whisper container and point `--model` at it:

```bash
docker run -it -p 10300:10300 -v ./data:/data \
  -v $PWD/models/slovene-whisper-ct2:/models/slovene:ro \
  rhasspy/wyoming-whisper --data-dir /data --model /models/slovene --language sl
```

> **The voice is different.** Its config declares `"phoneme_type": "slovene"`,
> which upstream piper does not know, so stock piper and the Home Assistant
> Piper add-on **refuse to load it** by default.
>
> `tools/make_espeak_config.py` produces a config they will accept, if a
> degraded voice in your existing stack beats adding a container. The cost is
> measured: **~74% of word forms mis-stressed** and **~6.5% of phoneme tokens
> rendered as symbols the model never trained**, notably every `/r/`. Nothing
> warns you at runtime — it sounds fluent and wrong. For correct pronunciation
> use `serve/`.

---

## Models

Weights are published as **GitHub Release assets** — part of this repository,
downloadable without a git clone, and permanent.

They are not committed into git itself, and that is deliberate rather than
tidiness: the recogniser is 1.5 GB, GitHub rejects any file over 100 MB, and a
binary in git history is there forever and is downloaded by every clone. Release
assets have a 2 GB per-file limit and cost nothing to skip.

| asset | download | on disk | what it is |
|---|---|---|---|
| `slovene-voice.tar.gz` | 68 MiB | **74 MB** | TTS bundle: ONNX voice + compressed Sloleks lexicon + runtime |
| `slovene-whisper-ct2.tar.gz` | 1.23 GiB | **1.5 GB** | STT model, CTranslate2 `int8_float16` |
| `slovene-whisper-lora.tar.gz` | 112 MiB | **124 MB** | the LoRA adapter alone, for use with transformers |

The two size columns differ because the assets are gzipped. `1.5 GB on disk`
elsewhere in this README is the extracted recogniser: `model.bin` alone is
1,479 MiB.

```bash
scripts/fetch-models.sh          # downloads and verifies checksums
```

---

## Updating a component

Most updates do **not** need retraining. The model consumes phoneme IDs, not
text, so the lexicon, the rewrites, espeak and the English detection all sit
outside the network and can be replaced in minutes. And because the LoRA adapter
is published alongside the converted recogniser, any future runtime format can
be produced without touching the GPU.

Retraining is only needed for a new ARTUR release, a new Whisper base model, or
a better voice. See **[docs/MAINTENANCE.md](docs/MAINTENANCE.md)** for the full
table and the procedures.

One trap worth knowing before swapping a lexicon: the model emitted exactly
**46 distinct symbols** across its entire training set, while its config
declares 166. A lexicon using anything outside those 46 fails silently — as an
artefact if the symbol is in the id map, as a dropped sound if it is not. Run
`tools/check_lexicon.py` first; it exits non-zero on a problem.

---

## Training

See **[docs/TRAINING.md](docs/TRAINING.md)** for the full recipe. The short version:

**STT** — LoRA r=32 on q/k/v/out projections of whisper-large-v3, 2 epochs,
batch 4 × grad-accum 8, lr 1e-4, fp16. About 22 GPU-hours per run on a single
V100.

**TTS** — Piper/VITS medium tier, 22.05 kHz, trained from the ARTUR studio
subset with lexicon-derived phonemes rather than espeak ones.

The training doc also records **three data bugs that cost two full training
days** and were invisible in the loss curve — packing utterances across file
boundaries, orphaned punctuation from the `.trs` parser, and a Whisper-specific
`decoder_start_token_id` trap that produced 291% WER. Each was caught only by
per-domain evaluation; a single overall WER would have hidden all three. If you
are fine-tuning Whisper on your own language, that section is the useful part of
this repository.

---

## What went into these models

Everything used to build these models, and the terms each piece carries. The
corpora are the substantive contribution — this project is a recipe applied to
them.

### Summary

| component | role | licence | redistributed here? |
|---|---|---|---|
| **ARTUR 1.0 (audio)** | ASR training data, 853 h | CC BY-SA 4.0 | no — fetch from CLARIN.SI |
| **ARTUR 1.0 (transcriptions)** | ASR targets | CC BY-SA 4.0 | no |
| **`artur_studio_tts`** | TTS training data, 40 h, 1 speaker | tagged CC BY 3.0, derives from CC BY-SA 4.0 | no |
| **Sloleks 3.1** | pronunciation lexicon, 2.9 M forms | CC BY-SA 4.0 | **yes** — in the TTS bundle |
| **RSDO-DS2-NORM** | digit/date → spoken form | Apache 2.0 | no — optional service |
| **RSDO-DS2-DENORM** | spoken → written form | Apache 2.0 | not yet used (see roadmap) |
| **whisper-large-v3** | ASR base model | MIT | as a fine-tuned derivative |
| **Piper / VITS** | TTS architecture + training code | MIT | `ids.py` logic reproduced |
| **faster-whisper / CTranslate2** | ASR inference runtime | MIT | no — pip dependency |
| **onnxruntime** | TTS inference runtime | MIT | no — pip dependency |
| **espeak-ng** | OOV fallback G2P (~1.34% of tokens) | **GPL-3.0** | no — separate process, CLI only |
| **CLASSLA** | morphosyntactic tagging, homographs | Apache 2.0 | no — optional service |
| **CMUdict-derived word list** | English-word detection (115,266 words) | BSD-2-Clause | list only, in the phonemizer |

**The one licence to look at twice is espeak-ng (GPL-3.0).** It is invoked as a
separate process through its command-line interface and never linked, so it
imposes no terms on this code and you install it yourself. If you were to link
it into a binary, that would be a different question.

### Training data, in full

#### ARTUR 1.0 (audio) — the ASR corpus

> Verdonik, Darinka; Bizjak, Andreja; Žgank, Andrej; Bernjak, Mitja; Antloga,
> Špela; Majhenič, Simona; Čakš, Peter; Pucer, Matevž; Cvetko, Mitja; Zelenik,
> Marijana; Pavlič, Jani; Dobrišek, Simon; Križaj, Janez; Strle, Gregor;
> Ivanovska, Marija; Grm, Klemen; Bajec, Marko; Lebar Bajec, Iztok; Jelovšek,
> Tjaša; Lokovšek, Jure; Longyka, Jure; Trojar, Mitja; Žganec Gros, Jerneja;
> Mihelič, Aleš; Vesnicer, Boštjan; Dretnik, Naum; Bordon, David, 2023,
> *ASR database ARTUR 1.0 (audio)*, Slovenian language resource repository
> CLARIN.SI, ISSN 2820-4042, http://hdl.handle.net/11356/1776.

**CC BY-SA 4.0.** Published 2023-02-27 by the Faculty of Electrical Engineering
and Computer Science, University of Maribor; the Faculty of Electrical
Engineering, University of Ljubljana; the Faculty of Computer and Information
Science, University of Ljubljana; Alpineon d.o.o.; and STA. Produced within the
[RSDO](https://rsdo.slovenscina.eu/en/speech-technologies) programme.

1,067 hours recorded, 884 transcribed, of which 853 were used here. Four
domains: read speech (B), private conversation (N), public speech (J),
parliamentary (P). The transcriptions are a separate CLARIN record,
http://hdl.handle.net/11356/1772. About 260 GB to download.

#### Sloleks 3.1 — the pronunciation lexicon

> Čibej, Jaka; Gantar, Kaja; Dobrovoljc, Kaja; Krek, Simon; Holozan, Peter;
> Erjavec, Tomaž; Romih, Miro; Arhar Holdt, Špela; Krsnik, Luka;
> Robnik-Šikonja, Marko, 2026, *Morphological Lexicon of Slovene Sloleks 3.1*,
> Slovenian language resource repository CLARIN.SI,
> http://hdl.handle.net/11356/2080.

**CC BY-SA 4.0.** Published by the Centre for Language Resources and
Technologies, University of Ljubljana. 2,910,337 word forms with reference IPA
from the RSDO G2P, giving 98.66% token coverage on real Slovene text.

This is the one third-party dataset **redistributed in this repository's
Release assets**, as a compressed lookup table inside the TTS bundle. It is
unambiguously a derivative work, so share-alike plainly applies to it.

#### artur_studio_tts — the TTS subset

A 40-hour single-speaker (G0911) studio subset derived from ARTUR's
`Artur-B-Studio` portion, prepared by **ppisljar** and distributed at
[huggingface.co/datasets/ppisljar/artur_studio_tts](https://huggingface.co/datasets/ppisljar/artur_studio_tts).
22,796 rows, resampled to 22.05 kHz.

> **Licence conflict, stated plainly.** That dataset card is tagged
> `cc-by-3.0`, but it is a derivative of ARTUR 1.0, which is CC BY-SA 4.0. A
> redistributor cannot drop the share-alike condition of the work being
> redistributed. This project therefore treats the TTS training data — and the
> voice trained on it — as **CC BY-SA 4.0**, the stricter upstream term. If you
> reuse that dataset directly, satisfy CC BY-SA 4.0.

### Resources considered and not yet used

Published, suitable, and on the roadmap rather than in the models:

| resource | would fix | licence |
|---|---|---|
| **SNES 1.0** ([11356/2003](http://hdl.handle.net/11356/2003)) | stress for out-of-vocabulary words, the 1.34% espeak currently guesses at | CC BY-SA 4.0 |
| **ILS 1.0** ([11356/2025](http://hdl.handle.net/11356/2025)) | pre-consonantal *l* — /l/ versus /u̯/ | CC BY-SA 4.0 |
| **RSDO-DS2-DENORM** ([11356/1743](http://hdl.handle.net/11356/1743)) | ASR output in written form — digits, dates, abbreviations | Apache 2.0 |

All three were located by harvesting the CLARIN.SI catalogue over OAI-PMH.
A practical note for anyone else searching it: **CLARIN.SI records are
catalogued in English only**, so Slovene keyword searches return almost nothing.

### Licence options, if you need different terms

Everything above is either permissive (MIT, Apache 2.0, BSD) or CC BY-SA 4.0.
There is no non-commercial restriction anywhere in this stack — **commercial use
is permitted throughout**. The only real constraint is share-alike, and it comes
entirely from ARTUR and Sloleks.

If share-alike does not work for you, the options are, honestly, limited:

1. **Comply.** Credit the corpus authors and release your derivative under
   CC BY-SA 4.0. This is what the data's authors asked for.
2. **Retrain on differently-licensed data.** The code here is MIT and the recipe
   is documented; point it at a corpus whose terms suit you. The models are the
   replaceable part.
3. **Ask.** CLARIN.SI lists the depositors for both corpora. Rights holders can
   grant terms a public licence does not.

What is *not* an option is relabelling the weights as permissive and hoping the
question never comes up — which, as the `artur_studio_tts` tag shows, is an easy
mistake to make by accident.

---

## Roadmap

See **[docs/ROADMAP.md](docs/ROADMAP.md)** for the full list with rationale. The
items most likely to matter:

- **Evaluate on a corpus neither model was trained on.** Every number in this
  README is in-domain against ARTUR. Until that is done, the comparison against
  stock Whisper flatters this model.
- **Written-form ASR output** via RSDO-DS2-DENORM — digits, dates and
  abbreviations currently come out as spoken words.
- **Multi-speaker TTS.** ARTUR has many speakers; only one was used.
- **Lift the 46-phoneme ceiling**, which is what currently makes faithful
  English impossible in the voice.
- **Stress for out-of-vocabulary words** using SNES 1.0, replacing the espeak
  guess.
- **whisper.cpp / GGML conversion** so the recogniser runs well on CPU and
  Apple Silicon.

Contributions to any of these are welcome; several are self-contained.

---

## Licence

This repository is deliberately split, because the code and the weights have
different origins.

| part | licence | why |
|---|---|---|
| source code (`tts/`, `stt/`, `scripts/`, `deploy/`) | **MIT** | original work; `tts/slovene_voice/ids.py` reproduces logic from [piper](https://github.com/rhasspy/piper) (MIT) and that notice is retained in [NOTICE](NOTICE) |
| model weights and the bundled lexicon (Release assets) | **CC BY-SA 4.0** | trained on and derived from ARTUR 1.0 and Sloleks 3.1, both CC BY-SA 4.0 |

**What CC BY-SA 4.0 means for you here:** you may use these models commercially,
modify them and redistribute them, provided you (a) credit the corpus authors
listed above and (b) release your adaptations under CC BY-SA 4.0 as well.

Whether model weights are legally a "derivative work" of their training data is
genuinely unsettled, and nothing here is legal advice. This project applies
share-alike to the weights as the interpretation most consistent with the
corpora's terms and with the spirit in which the data was published. Attribution
is required under any reading.

`espeak-ng` is GPLv3 and is invoked as a **separate process** via its CLI, never
linked, so it imposes no terms on this code. It is a runtime dependency you
install yourself.

---

## Credits

The models are the easy part. The hard part — a 1,000-hour transcribed speech
corpus and a 2.9-million-form pronunciation lexicon for a language of two
million speakers — was done by other people, over years, with public funding,
and then given away.

- the **ARTUR** team (27 authors, led by **Darinka Verdonik**) and the
  **Sloleks** team (10 authors, led by **Jaka Čibej**)
- the **RSDO** national programme (*Razvoj slovenščine v digitalnem okolju*),
  which funded both
- **CLARIN.SI** for hosting them openly and durably
- **ppisljar** for isolating and publishing the single-speaker studio subset
  that made a TTS voice possible at all
- **Michael Hansen** and contributors for [piper](https://github.com/rhasspy/piper)
- **OpenAI** for Whisper, and **SYSTRAN** for faster-whisper / CTranslate2
- **Luka Krsnik**, whose work on Slovene accentuation underpins the Sloleks
  pronunciation data this voice depends on

If you build something with these, consider citing the corpora rather than this
repository. They are the contribution.

---

## Citation

See [CITATION.cff](CITATION.cff). Please cite **ARTUR** and **Sloleks**
alongside, or instead of, this work.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pronunciation corrections are
particularly welcome — they go in `user_lexicon.tsv` and need no retraining.
