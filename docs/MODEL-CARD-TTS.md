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
| speed | up to 60× realtime on CPU (measured, Python 3.12/3.13) |
| licence | CC BY-SA 4.0 (see LICENSE-MODELS.md) |

## What makes this voice different

It does not use rule-based grapheme-to-phoneme conversion. Slovene lexical
stress is unpredictable from spelling, and espeak-ng places it wrongly on **~74%
of word forms** (measured against Sloleks 3.1 over 4,000 forms). This voice is
trained on, and expects at inference, **lexicon-derived pronunciations** from
Sloleks 3.1 — 2.9 million word forms, 98.66% token coverage on real text.

**Consequence: the `.onnx` file alone is not usable.** Its input is phoneme IDs,
not text, and the config declares `"phoneme_type": "slovene"`, which upstream
piper does not recognise. The voice is served by pairing official piper with a
small lexicon phonemizer — either this repo's proxy (the light stack) or the
`slovene_voice` library. Fed plain text through stock piper's espeak G2P instead,
it mis-stresses ~74% of words. See the deployment guide.

## Provenance and relationship to the upstream voice

There is an existing Slovene Piper voice, `sl_SI-artur-medium`, in
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/main/sl/sl_SI/artur/medium).
This voice is a **sibling of it, not a derivative.** The two share the same
training audio (`artur_studio_tts`), the same single speaker, and the same
architecture; they differ in **one variable** — the upstream voice phonemizes
with espeak-ng, this one with the Sloleks lexicon.

- It was **not retrained from the upstream `.onnx`.** An `.onnx` is
  inference-only and cannot be trained further, and the upstream *training*
  checkpoint was never published. This model was trained independently.
- It was **warmstarted from the Piper Czech voice `cs_CZ-jirka`** (Michael
  Hansen / rhasspy), a standard related-language transfer-learning start for a
  medium VITS voice. The final weights are dominated by Slovene training but
  derive from that checkpoint; it is credited in [NOTICE](../NOTICE).
- It is **renamed** to `sl_SI-artur-sloleks-medium` precisely to end the naming
  collision with the upstream voice — the `sloleks` tag names the one thing that
  differs.

A documented A/B listening test showing where the phonemizer difference is
audible (and where it is not) is in
[docs/evaluation/stress-comparison.md](evaluation/stress-comparison.md).

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
  (ɹ æ ʊ ɚ ɑ ʌ ɜ ɐ θ ð). Those ten are *mapped* but never trained, so this is a
  training-data limit rather than a piper one — an untrained phoneme yields an
  artefact, not an accent, and no upgrade fixes it.
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
