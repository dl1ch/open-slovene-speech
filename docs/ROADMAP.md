# Roadmap

What could be done next, why it matters, and roughly what it costs. Ordered by
what would change the most for users, not by what is easiest.

Effort is in person-days of focused work plus GPU time on a single 32 GB
datacenter GPU of compute capability 7.0. "Blocked on" means it genuinely cannot start until that is resolved.

---

## 0. Honesty first: evaluate outside ARTUR

**Status: the most important item here, and the one that could make the headline
numbers look worse.**

Every figure in this repository is measured on a held-out, speaker-disjoint
split of ARTUR — the same corpus the model trained on. That is a legitimate
measurement of in-domain performance and it is what the README says it is. But
stock Whisper has never seen ARTUR, so **the 18.41% → 7.77% comparison partly
measures domain adaptation rather than general Slovene ability**. A fair
comparison needs a corpus neither model trained on.

- **What:** evaluate both models on Slovene Common Voice, FLEURS, or a held-out
  broadcast set. Publish whatever comes out, including a smaller gap.
- **Effort:** 1–2 days, almost no GPU.
- **Why it is first:** everything else is tuning a number nobody has verified
  generalises.

---

## 1. Speech recognition

### 1.1 Written-form output (denormalisation) — ✅ done

Integrated. The servers run ASR output through RSDO-DS2-DENORM
([11356/1743](http://hdl.handle.net/11356/1743), Apache 2.0), an 8.9 MB
rule-based tool that turns *tisoč devetsto šestinšestdeset* into *1966*.

Measured on FLEURS, with denormalisation on: CER **7.12% → 4.55%** and WER
**14.53% → 12.72%**, both now below stock Whisper (6.25% / 20.92%). On the 156
digit-bearing utterances alone, CER went **20.74% → 6.49%**.

On by default in both servers and `stt/transcribe.py`; a no-op if the tool is
not installed. `--raw` / `SLOVENE_DENORMALIZE=0` turns it off.

Remaining: it is tuned for formal text, so conversational output is uncharacterised.

### 1.2 Punctuation and casing

Related but separate: Whisper produces some punctuation, inconsistently, and
ARTUR's transcription conventions are not subtitle conventions.

- **Effort:** 3–5 days including evaluation. Could be folded into 1.1.

### 1.3 Noise and far-field robustness

Not a focus of the fine-tune. Augmenting training with noise, reverberation and
codec artefacts is standard and would likely help real recordings more than any
further in-domain tuning.

- **What:** RIR + noise augmentation, re-run the LoRA fine-tune.
- **Effort:** 2 days of work, ~22 GPU-hours per run.
- **Measure it on 0** or the gain is unverifiable.

### 1.4 Smaller and faster models

large-v3 needs ~2 GB VRAM and a GPU for comfortable speed. Many uses would
accept a few points of WER for CPU-realtime.

- **What:** the same LoRA recipe on `medium` and `small`; also a distil-whisper
  style student.
- **Effort:** 1 day each plus GPU; the recipe is unchanged.
- **Value:** this is what makes the model usable on a laptop or a Pi.

### 1.5 whisper.cpp / GGML conversion

CTranslate2 covers CUDA well. GGML would cover Apple Silicon and plain CPU, and
is how most hobbyists would actually run this.

- **Effort:** 1–2 days, mostly conversion and verification.
- **Blocked on:** nothing.

### 1.6 Streaming

Everything here is batch. Real-time captioning needs chunked decoding with
context carry-over.

- **Effort:** 1–2 weeks, and it is a genuine engineering problem rather than a
  training one.

### 1.7 Dialect and speaker-group evaluation

The model card says performance on strong regional dialect, children's speech,
disordered speech and non-native speakers "has not been measured per group".
That gap should be closed rather than left as a disclaimer.

- **Blocked on:** finding or building labelled test sets. This is the kind of
  thing a university group is better placed to do than an individual.

---

## 2. Text to speech

### 2.1 Multi-speaker

ARTUR contains many speakers; exactly one was used, because a clean 40 h studio
subset existed for that one.

- **What:** VITS multi-speaker training over a wider slice of ARTUR.
- **Effort:** 1 week of data preparation, several days of GPU.
- **Also fixes:** the single-voice limitation, and reduces the impersonation
  concern that comes with one identifiable person's voice.

### 2.2 Lift the 46-phoneme ceiling

Over all 22,768 training utterances the model emitted exactly 46 distinct
symbols. Its id map declares 166, so **120 entries were never trained.**

Two different problems hide under "the phoneme limitation", and they have
different fixes. Checked against the current upstream (OHF-Voice/piper1-gpl)
and against this voice's own config:

| | in the id map? | trained? | what would fix it |
|---|---|---|---|
| ɹ æ ʊ ɚ ɑ ʌ ɜ ɐ θ ð (English) | **yes, all 10** | no | training data containing those sounds |
| ʧ ʣ ʤ ⁿ ˡ (Sloleks) | **no** | no | a larger id map at training time |

**The English row is not a piper limitation at all.** Every symbol is already
mapped; the model has simply never produced one, so it has an embedding that
learned nothing. No change to piper or to the map helps. Only Slovene speech
containing English sounds would, which is why faithful English needs a retrain
on new data and nothing less.

**The Sloleks row is fixable, and more cheaply than it looks.** The 166-symbol
map was never a hard limit of piper:

```python
# piper1-gpl, phoneme_ids.py
def phonemes_to_ids(phonemes, id_map=None):
    if not id_map:
        id_map = DEFAULT_PHONEME_ID_MAP
```

```python
# piper1-gpl, config.py
phoneme_id_map=config["phoneme_id_map"],
```

The map travels in each voice's config JSON; the 166-entry default is only a
fallback for voices that supply none, and there is no `MAX_PHONEMES` anywhere.
A retrained voice can declare whatever inventory it wants, including ʧ ʣ ʤ ⁿ ˡ
directly — which would remove the rewrite step (`ʧ→tʃ` and friends) entirely
and let the model learn Slovene č as one unit rather than two.

Checked on the current upstream: the default map is still 166 entries and still
lacks all five, so nothing improves by upgrading piper. The opportunity is in
the training config, not in the dependency.

#### What a retrain actually costs

Measured on the completed run (32 GB GPU, compute capability 7.0, batch 32,
16-mixed precision):

| | |
|---|---|
| throughput | **8.2–8.4 epochs/hour** (from checkpoint timestamps, two independent windows) |
| full run | 1,120 epochs → **~137 hours, 5.7 days** continuous |
| dataset | 22,768 utterances |

**Most of the cache survives a phoneme-map change**, which is what makes this
cheaper than it looks:

| cache | size | invalidated? |
|---|---|---|
| `spec.pt` (mel spectrograms) | 27.0 GiB | **no** |
| `audio.pt` | 13.5 GiB | **no** |
| `phonemes.pt` + `phonemes.txt` | 0.1 GiB | yes — regenerate |

Only the phoneme files change, and regenerating them is a lexicon lookup over
22,768 utterances: minutes of CPU, no GPU. The 40.5 GiB of audio and mel
features is untouched, because the recordings have not changed.

**Two routes, and they differ by two orders of magnitude:**

*From scratch* — **~5.7 days**. Needed if the symbol→id assignment changes for
existing symbols, because every embedding row then means something different.

*Fine-tune from `epoch=729`* — **hours, not days, but unmeasured**. Append the
five new symbols at ids 166–170 so every existing id keeps its meaning, grow
the embedding table from 166 to 171 rows, and carry the old rows over. The new
rows need not be random either: `ʧ` can be initialised from the mean of the
existing `t` and `ʃ` embeddings, which is exactly the sound the model already
produces for it. With that warm start, and `ʧ` at ~0.95% of all phonemes so it
appears in most utterances, convergence should take tens of epochs — call it
**3–10 hours**. That estimate is reasoning, not measurement; the embedding
surgery is mechanical but has not been done here.

**Is it worth doing alone? Probably not.** The model already renders Slovene č
correctly as `tʃ`; the gain is one fewer rewrite step and a slightly cleaner
inventory, not audibly better speech. The case for spending 5.7 days is
multi-speaker (2.1), English phonemes (this item), and digits (2.3) **together
in one run** — the phoneme-map extension is nearly free once that run is
happening anyway.

For comparison, the ASR side is far cheaper: ~22 GPU-hours for a full LoRA
fine-tune, and re-converting to a new CTranslate2 format needs no GPU at all.

- **Effort:** see the table above. The Sloleks half is nearly free once a
  retrain is happening; the English half needs new data and is substantial.
- **Honest view:** a Slovene-accented rendering of English is also what a
  Slovene speaker produces, so the English half may not be worth its cost.

### 2.3 Digits in training data

ARTUR's TTS text contains no digits at all, so the voice has never seen one.
Today that is handled by an external normaliser.

- **What:** synthesise digit-bearing training text with the normaliser, so the
  model learns the mapping directly.
- **Effort:** 3–5 days plus a retrain.
- **Value:** removes a ~5 GB dependency from the common path.

### 2.4 Stress for out-of-vocabulary words

Sloleks covers 98.66% of tokens. The remaining 1.34% — names, loanwords,
neologisms — falls through to espeak, which is exactly the component known to
stress ~74% of Slovene forms wrongly.

- **What:** use **SNES 1.0** ([11356/2003](http://hdl.handle.net/11356/2003),
  CC BY-SA 4.0), a dataset built for OOV stress prediction.
- **Effort:** 3–5 days.
- **Why it matters:** proper nouns are disproportionately OOV and
  disproportionately noticed.

### 2.5 Pre-consonantal *l*

Slovene *l* before a consonant is often /u̯/, not /l/. **ILS 1.0**
([11356/2025](http://hdl.handle.net/11356/2025), CC BY-SA 4.0) exists precisely
for this distinction.

- **Effort:** 2–3 days in the phonemizer. Cheap, and audible.

### 2.6 Homograph disambiguation in the portable bundle

Homographs are 0.7% of forms, and **87% are resolvable by morphosyntactic tag**.
The full deployment resolves them with CLASSLA; the portable bundle does not,
because CLASSLA costs ~5 GB.

- **What:** a small tagger, or shipping the most common homographs with
  context rules.
- **Effort:** 1 week for something lightweight.

### 2.7 Shrink the lexicon's memory footprint

The voice needs **~1.06 GB of RAM**, and roughly 980 MB of that is the Sloleks
lexicon held as a Python dict of 2.9 M entries. The model itself is 60 MB. That
ratio is what keeps this off small devices: a 1 GB Raspberry Pi cannot run it,
despite the bundle being 74 MB on disk.

- **What:** store the lexicon as an on-disk structure read without loading it
  whole — SQLite, a memory-mapped trie, or a sorted file with binary search.
  Lookups are single-word and random, which all three handle well.
- **Effort:** 2–3 days, no retraining, no change to the model.
- **Payoff:** a voice that fits comfortably in 256 MB, which would make it
  usable on hardware where a 60 MB ONNX model always should have fit.

### 2.8 Prosody beyond declarative

The voice is trained on studio-read declarative speech. Questions, emphasis and
long narration all sound flat.

- **Effort:** open-ended. Needs expressive training data that does not currently
  exist for Slovene in a free licence.

---

## 3. Packaging and reach

### 3.1 Publish to Hugging Face

A GitHub Release is not where people look for models.

- **What:** model repos with cards, plus the CC BY-SA 4.0 terms and the full
  ARTUR/Sloleks attribution carried over.
- **Effort:** 1 day.
- **Highest reach-per-hour item on this list.**

### 3.2 Prebuilt container images — ✅ partly done

`Dockerfile` and `docker-compose.yml` ship. What remains is publishing images to
GHCR so nobody has to build locally, and adding a CUDA image for the recogniser.

- **Effort:** half a day plus a release workflow.

### 3.3 OpenAI-compatible and Wyoming endpoints — ✅ done

`serve/openai_api.py` implements `/v1/audio/speech` and
`/v1/audio/transcriptions`; `serve/wyoming_server.py` implements the Wyoming
protocol for Home Assistant.

Remaining: streaming synthesis over Wyoming (`supports_synthesize_streaming`),
which would cut perceived latency in Assist pipelines on long replies.

- **Effort:** 2–3 days.

### 3.4 Get the phonemizer upstream — the way to drop the self-hosting

`PhonemeType.SLOVENE` does not exist in piper, so no piper-based stack can serve
this voice correctly. Upstreaming it would remove the need for any server in
this repository: `wyoming-piper`, the Home Assistant Piper add-on and
`openedai-speech` would all just work.

**The precedent is almost exactly this project.** Current piper
([OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)) ships
`PhonemeType.LITHUANIAN`, documented as *"espeak-ng IPA + pitch accent
dictionary"* — espeak output corrected by a lexical lookup, which is what the
Slovene phonemizer does. Its data is built from CC-BY-4.0 corpora, so a
CC BY-SA 4.0 lexicon is not obviously out of bounds either.

**The established pattern**, from reading the Lithuanian implementation:

```
src/piper/phonemize_lithuanian.py     the module
src/piper/lithuanian/lt_kirciai.tsv   the dictionary, plain TSV, package data
src/piper/lithuanian/LICENSE          third-party data ships its own licence
src/piper/lithuanian/SOURCE           and its provenance
```

```python
# voice.py — lazily imported, one instance cached per voice
if self.config.phoneme_type == PhonemeType.LITHUANIAN:
    from .phonemize_lithuanian import LithuanianPhonemizer
    phonemizer = getattr(self, "_lithuanian_phonemizer", None)
    if phonemizer is None:
        phonemizer = LithuanianPhonemizer(espeak_data_dir=self.espeak_data_dir)
        setattr(self, "_lithuanian_phonemizer", phonemizer)
    return phonemizer.phonemize(text)
```

Data files are listed by explicit filename in `setup.py`'s `package_data` — not
a glob — so adding a language means touching the packaging too.

**The one real obstacle is size.** Lithuanian's dictionary is 189k words and "a
few MB" of plain TSV. Sloleks is 2.9 M forms: 81 MB raw, 13 MB as `.xz`. That is
roughly fifteen times any data currently bundled, and there is no precedent in
the repo for compressed package data.

Measured here, pruning is not a free answer:

| lexicon | entries | token coverage |
|---|---|---|
| full Sloleks | 2,910,337 | **98.54%** |
| pruned to a 22.7k-utterance corpus | 50,718 | **88.08%** |

Dropping to Lithuanian's scale costs about ten points of coverage, which moves
12% of tokens onto espeak's wrong stress instead of 1.5%. A pruned list chosen
from a large Slovene corpus (Gigafida rather than this read-speech set) would do
considerably better at the same size, and that is the experiment worth running
before proposing anything.

**There is also a download precedent.** The Chinese phonemizer takes its data
from `self.download_dir / "g2pW"` rather than package data, so a large lexicon
fetched on first use is not without a model in the project.

- **Realistic plan:** (a) build a frequency-pruned lexicon from a proper corpus
  and measure where coverage lands at ~5 MB, (b) if that is good enough, follow
  the Lithuanian pattern exactly; if not, follow the Chinese download pattern.
- **Caveats:** piper1-gpl is GPL, so contributing means contributing under GPL,
  and the project has no `CONTRIBUTING.md` and is currently asking for
  maintainers, so review capacity is uncertain.
- **Publishing first helps.** A phonemizer cannot be upstreamed while its
  lexicon and voice are unpublished.

## 4. Deliberately not planned

- **Voice cloning.** This voice is one identifiable person who was recorded for
  a research corpus. Building cloning on top of that is not something this
  project will ship. See the model cards.
- **Removing the share-alike terms.** Not possible honestly, and not desirable:
  the corpora exist because they were publicly funded and openly licensed.
- **Chasing a better `val_mel`.** It is reported for reproducibility. It does not
  correlate with what a native speaker hears, which is the metric that matters —
  epoch 207 held the best loss for roughly 500 epochs while the voice kept
  improving audibly.

---

## How to help

Several of these are genuinely self-contained and need no GPU: **0** (external
evaluation), **1.5** (GGML conversion), **2.5** (pre-consonantal *l*), **3.1**
(Hugging Face), **3.2** (container). Each is a useful contribution on its own.

If you are a researcher with access to Slovene speech data that cannot be
redistributed, **item 0 is still possible on your side** — running the published
model against your held-out set and reporting the number would be valuable even
without sharing the data.
