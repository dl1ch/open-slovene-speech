# Training

Both models were trained on a single NVIDIA V100 (16 GB, sm_70). Neither needed
a cluster.

---

## Speech recognition

**Base.** `openai/whisper-large-v3` (1.55 B parameters).

**Data.** 853 hours of ARTUR 1.0, covering four domains: read speech (Artur-B),
private conversation (Artur-N), public speech (Artur-J) and parliamentary
proceedings (Artur-P).

**Method.** LoRA, rank 32, applied to the `q`, `k`, `v` and `out` projections.

| hyperparameter | value |
|---|---|
| epochs | 2 |
| batch size | 4 |
| gradient accumulation | 8 (effective batch 32) |
| learning rate | 1e-4 |
| precision | fp16 |
| cost | ~22 GPU-hours per run |

**Evaluation.** 1,000 utterances, 250 per domain, **speaker-disjoint** from
training. Speaker disjointness matters more than the size of the test set: with
the same speakers on both sides, the numbers improve and mean nothing.

**Deployment.** Converted to CTranslate2 `int8_float16`. Quantisation did not
cost accuracy on this test set — 7.26% WER converted, against 7.77% for the
unquantised adapter.

---

## Three data bugs that cost two training days

Each was invisible in the loss curve. Each was caught only by **per-domain**
evaluation. A single overall WER would have hidden all three, and that is the
transferable lesson: if you are fine-tuning on a corpus with distinct domains,
evaluate them separately or you will not see what you have broken.

### 1. Packing across file boundaries

Utterances were packed into 30-second windows — a 5.4× throughput win — but
grouped by *section* rather than by *source file*.

Artur-B is one isolated read sentence per file. So roughly five unrelated
sentences were concatenated into each window, and the model learned that
**another sentence always follows**. Given a single short utterance at
inference, it invented a continuation.

- read-speech WER: 12.09% → **19.06%**, i.e. worse than the stock model
- hypothesis/reference word ratio: **2.10** — it emitted twice the words present

**Fix:** pack only within the same audio file. Artur-B then never packs (one
segment per file), while Artur-J/N/P pack contiguous segments, which really are
continuous speech.

### 2. Orphaned punctuation from the `.trs` parser

Transcriber `.trs` files interleave `<Event>` tags — pauses, noises — inside a
turn. Joining the surrounding text fragments with a space produced strings like
`"ministrica. . Težava"`, and left **2,109 Artur-P utterances starting with a
stray full stop**.

The model learned it faithfully and began emitting `... ... . . . .` —
**3.14× over-generation**, with Artur-P stuck at 23.70% WER.

This one only became dominant in the second iteration: capping read speech (the
fix for bug 1) raised Artur-P's share of the data from 21% to 33%, which
promoted a latent bug into the leading error source. **Fixing one sampling
problem can expose another.**

### 3. The Whisper `decoder_start_token_id` trap

Not a data bug — a framework trap, and the most expensive of the three to
diagnose.

Whisper's tokenizer emits labels beginning with `<|startoftranscript|>`, which
**is** `decoder_start_token_id`. Whisper prepends that token itself. Leaving it
in the labels offsets every target by one position, so the model never learns
where `<|endoftext|>` belongs.

The symptom is bizarre and memorable: **291% WER**, transcribing the audio
correctly and then failing to stop. Strip the leading token from the labels.

---

## Text to speech

**Architecture.** Piper/VITS, medium tier, 22.05 kHz.

**Data.** `artur_studio_tts` — 40 hours of studio recordings from a single
speaker (G0911), derived from ARTUR's `Artur-B-Studio` portion. 22,768 cached
training utterances.

**The decisive choice: lexicon phonemes, not espeak phonemes.**

Slovene lexical stress is a property of the word, not of its spelling, so a
rule-based G2P cannot place it correctly in principle. Measured against Sloleks
3.1 over 4,000 forms, espeak-ng stresses **~74% of word forms wrongly**. Training
on Sloleks's reference IPA instead was confirmed audibly better by a native
speaker at **epoch 6** of the lexicon-trained run, against **epoch 91** of the
espeak-trained one.

**Checkpoint.** Epoch 729, `val_mel` 0.3784 — the best of the run. Epoch 207
(0.3818) held the record for roughly 500 epochs, long enough that "this is
buying nothing" looked like the honest read. It was true as a description and
wrong as a prediction: the adversarial loss was working through a rough patch.

### Three traps in wiring a custom phonemizer into Piper

1. **Wire it into both paths.** `train/vits/dataset.py` is training; `voice.py`
   is inference. Doing only the first gives a model that trains correctly and
   then emits a **zero-byte WAV** (`wave.Error: # channels not specified`),
   because `PiperVoice` produces no phonemes at all.
2. **`phoneme_id_map` holds 166 symbols and silently drops anything else.** Six
   Sloleks symbols are absent, including `ʧ` — Slovene *č*, about 1% of all
   phonemes. Rewrite them to supported equivalents: `ʧ→tʃ`, `ʣ→dz`, `ʤ→dʒ`,
   `ⁿ→n`, `ˡ→l`.
3. **Do not use round-trip ASR to judge prosody.** Whisper recognises a word
   whether or not its stress is correct. WER sat at the human floor while 74% of
   words were mis-stressed. Use a native speaker.

---

## Reproducing

The corpora are free to download from CLARIN.SI; see the README for handles and
citations. ARTUR is ~260 GB. CLARIN served it at about 60 MB/s over four
parallel range requests, so roughly 1.2 hours.

Note that CLARIN.SI records are **catalogued in English only** — searching in
Slovene returns almost nothing useful.
