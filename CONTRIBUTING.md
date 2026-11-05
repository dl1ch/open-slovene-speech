# Contributing

Contributions are welcome. The most valuable ones are probably not code.

## Pronunciation corrections — the easiest useful contribution

If the voice mispronounces a word, it almost certainly does not need
retraining. Add a line to `user_lexicon.tsv`:

```
beseda<TAB>bɛˈseːda
```

That overrides the Sloleks entry. Open a pull request with the word, the IPA you
expect, and a note on what it currently says. Corrections to proper nouns,
loanwords and technical terms are especially useful, since those are exactly
what Sloleks does not cover.

## Reporting a bad transcription or a bad pronunciation

Please include:

- the exact input text (TTS) or a description of the audio (STT)
- what you got, and what you expected
- for STT: the domain — read speech, conversation, broadcast, meeting
- for TTS: whether `espeak-ng` is installed, since its absence silently drops
  out-of-vocabulary words

## If you touch the phonemizer, regression-test pure Slovene

This is not boilerplate. A previous change defined `_VOWELS = set("aeiouy")` in
`phonemize.py`, shadowing an existing module-level
`_VOWELS = set("aeiouɛɔəɪ")` used by hiatus elision. Python resolves globals at
call time, so the later assignment won, silently removing ɛ ɔ ə ɪ from hiatus
elision and changing 8 of 300 pure-Slovene utterances. No error, no warning. It
was caught only by diffing phoneme output against a backup.

So: before and after any phonemizer change, dump phoneme IDs for a fixed set of
Slovene sentences and diff them. Prefer prefixed names in a module you did not
write.

## Equivalence testing

Audio comparison is the wrong test for this model — VITS sampling is stochastic,
so two runs of identical input differ. Compare **phoneme IDs**, which are
deterministic. That is how the portable bundle was verified against the piper
path (191/191 tokens byte-identical).

## Licensing of contributions

By contributing you agree that:

- **code** contributions are licensed MIT, matching the repository
- **lexicon and model** contributions are licensed CC BY-SA 4.0, matching the
  data they derive from

Do not contribute pronunciations copied from a dictionary or resource with
incompatible terms. Sloleks (CC BY-SA 4.0) is fine; a proprietary dictionary is
not.

## Scope

Things that fit here: pronunciation fixes, inference bugs, packaging and
portability, documentation, evaluation on corpora other than ARTUR, and
conversions to other runtimes.

Things that do not: voice cloning features, and anything whose purpose is to
make synthesised speech pass as a real recording of a real person. See the
model cards.
