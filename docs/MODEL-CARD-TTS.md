# Model card — Slovene text-to-speech

## Summary

| | |
|---|---|
| task | text-to-speech, Slovene |
| architecture | Piper/VITS, medium tier |
| sample rate | 22.05 kHz |
| voice | single speaker (G0911), 40 h studio recordings |
| checkpoint | epoch 729, `val_mel` 0.3784 |
| size | 68 MiB download, 74 MB on disk, ~1.06 GB RAM when loaded |
| speed | 4–41× realtime on CPU, depending on the machine |
| licence | CC BY-SA 4.0 (see LICENSE-MODELS.md) |

## What makes this voice different

It does not use rule-based grapheme-to-phoneme conversion. Slovene lexical
stress is unpredictable from spelling, and espeak-ng places it wrongly on **~74%
of word forms** (measured against Sloleks 3.1 over 4,000 forms). This voice is
trained on, and expects at inference, **lexicon-derived pronunciations** from
Sloleks 3.1 — 2.9 million word forms, 98.66% token coverage on real text.

**Consequence: the `.onnx` file alone is not usable.** Its input is phoneme IDs,
not text, and the config declares `"phoneme_type": "slovene"`, which upstream
piper does not recognise — it raises `ValueError` and refuses to load. Forcing
the config to `espeak` makes it load and produces fluent speech with the stress
mostly wrong *and* with phoneme symbols the model never saw in training. Use the
bundled phonemizer.

## Quality evaluation

There is no reliable automatic metric here, and the reason is worth repeating:
**round-trip ASR is blind to stress errors.** Whisper recognises the word either
way, so WER sat at the human floor while three quarters of words were
mis-stressed. The voice was judged by a native speaker. `val_mel` is reported
for reproducibility, not as a quality claim.

## Intended use

Reading Slovene text aloud: screen readers and accessibility tooling, voice
assistants, audio versions of written material, language learning, hobby and
research projects.

## Out of scope

- **Impersonation.** This is one identifiable speaker's voice, recorded for a
  research corpus. Do not use it to imply that this person, or any person, said
  something they did not. Do not present synthesised audio as a real recording.
- **Voice cloning.** There is no mechanism here to clone another voice, and
  adding one against a real person's wishes would be an abuse of the corpus.
- **Safety-critical announcements** without human review.

## Limitations

- **One speaker, one style.** No speaker selection, no emotion control, no
  singing. Prosody is studio-read declarative; long dramatic text will sound
  flat.
- **Digits are not expanded.** ARTUR's training text contains no digits, so the
  model has never seen one. Expand them yourself or run the optional
  normalisation service — `27. 9. 2026` must become
  *sedemindvajsetega devetega dva tisoč šestindvajset*, with case agreement.
- **English and foreign words are Slovene-accented, by necessity**, and the
  detection needs `lexicon/english_words.txt` present — without it every
  English word falls through to Slovene letter-to-sound, silently. Over all
  22,768 training utterances the model produced exactly **46 distinct
  phonemes**; 120 of the 166 entries in its `phoneme_id_map` were never trained,
  and all core English vowels plus the English rhotic are in the untrained set
  (ɹ æ ʊ ɚ ɑ ʌ ɜ ɐ θ ð). An untrained phoneme yields an artefact, not an accent.
  English is therefore mapped onto Slovene phones — *think→tink*,
  *better→beter* — which is also what a Slovene speaker produces.
- **Out-of-vocabulary words need espeak-ng.** Without it they are *silently
  dropped*, which is worse than mispronouncing them. Install it.
- **Homographs** (0.7% of forms) are resolved by lexicon lookup in the portable
  bundle, not by context. 87% are resolvable by morphosyntactic tag, which the
  full deployment does via CLASSLA.

## Fixing a pronunciation

Wrong pronunciations do not need retraining. Add a line to `user_lexicon.tsv`:

```
beseda<TAB>bɛˈseːda
```

It overrides Sloleks. Corrections are welcome as pull requests.

## Attribution

Trained on a studio subset of ARTUR 1.0; pronunciations from Sloleks 3.1. Both
CC BY-SA 4.0. Full citations in [NOTICE](../NOTICE), and they must accompany
redistribution.
