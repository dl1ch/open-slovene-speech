# Model card — Slovene speech recognition

## Summary

| | |
|---|---|
| task | automatic speech recognition, Slovene |
| base model | `openai/whisper-large-v3` (1.55 B parameters) |
| method | LoRA, rank 32, on q/k/v/out projections |
| training data | ARTUR 1.0, 853 hours |
| deployed format | CTranslate2, `int8_float16`, 1.5 GB |
| licence | CC BY-SA 4.0 (see LICENSE-MODELS.md) |

## Performance

Speaker-disjoint test set, 1,000 utterances, 250 per domain:

| domain | stock large-v3 | this model |
|---|---|---|
| Artur-B — read speech | 12.09% | **4.16%** |
| Artur-N — private conversation | 17.50% | **5.68%** |
| Artur-J — public speech | 22.44% | **11.28%** |
| Artur-P — parliamentary | 25.84% | **11.56%** |
| **overall WER** | 18.41% | **7.77%** |
| **overall CER** | 6.82% | **2.41%** |

Decoding trade-off, measured on the deployed conversion:

| beam | WER | speed |
|---|---|---|
| greedy | 8.54% | 6.7× realtime |
| 2 | 8.19% | 5.3× realtime |
| 5 | 8.12% | 4.2× realtime |

Beam 2 is the knee and is what the reference deployment uses.

## Intended use

Transcribing Slovene speech: dictation, meeting and interview transcription,
subtitle generation, voice interfaces. It is a general-purpose Slovene ASR
model, not tuned to a single domain.

## Out of scope

- **Languages other than Slovene.** Fine-tuning has specialised it; it is worse
  than stock Whisper on other languages.
- **Speaker identification or diarisation.** It transcribes; it does not say who
  spoke.
- **Decisions about people.** Do not use transcripts as sole evidence in legal,
  medical, employment or disciplinary processes. At ~11% WER on conversational
  and parliamentary speech, roughly one word in nine is wrong.

## Limitations and biases

- **Domain gap is large.** Read speech is ~4% WER; spontaneous and
  parliamentary speech is ~11–12%. Quote the number for your domain, not the
  average.
- **The training corpus determines the accent and register coverage.** ARTUR is
  a broad Slovene corpus, but speakers heavily divergent from it — strong
  regional dialect, children, disordered speech, non-native speakers — are not
  well represented and will do worse. This has not been measured per group.
- **Noise and far-field audio** were not a focus. Expect degradation.
- **Numbers and formatting** come out as spoken, not written, unless you apply
  a denormaliser (RSDO-DS2-DENORM, Apache 2.0, see README).
- Whisper's known failure modes — hallucinating text over silence or noise,
  looping on degraded audio — are **inherited**. Fine-tuning does not remove
  them.

## Evaluation method

WER and CER over a held-out, speaker-disjoint split of ARTUR, 250 utterances per
domain. Because training and evaluation data come from the same corpus, these
figures measure in-domain performance. **Performance on audio unlike ARTUR is
not characterised by this table**, and a fair comparison against other systems
would need a corpus neither model was trained on.

## Training

Full recipe, including three data bugs worth avoiding, in
[TRAINING.md](TRAINING.md).

## Attribution

Trained on ARTUR 1.0 (CC BY-SA 4.0). The full 27-author citation is in
[NOTICE](../NOTICE) and must accompany redistribution.
