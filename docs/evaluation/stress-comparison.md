# Stress comparison — lexicon voice vs. the upstream espeak voice

This documents a direct A/B listening test between two Slovene voices that share
the same training audio, the same speaker, and the same architecture, and differ
in **one variable only**: how text is turned into phonemes.

- **espeak voice** — the upstream [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/main/sl/sl_SI/artur/medium)
  `sl_SI-artur-medium`, which phonemizes with espeak-ng's rule-based Slovene G2P.
- **lexicon voice** — this project's `sl_SI-artur-sloleks-medium`, which
  phonemizes by looking each word up in **Sloleks 3.1** and falls back to
  espeak only for the ~1.34% of tokens Sloleks does not cover.

Because only the phonemizer differs, this is close to a controlled experiment:
whatever you hear is attributable to espeak-vs-lexicon phonemization, not to data,
speaker, or model size.

> **Note on provenance.** These are *sibling* voices, not one derived from the
> other. This voice was **not** retrained from the upstream `.onnx` (an `.onnx`
> is inference-only and not trainable, and the upstream training checkpoint was
> never published). It was trained independently on the same
> `artur_studio_tts` data, warmstarted from the Piper Czech voice
> `cs_CZ-jirka`. See [MODEL-CARD-TTS.md](../MODEL-CARD-TTS.md).

## Why automatic metrics can't settle this

Round-trip ASR (synthesise → transcribe → compare) is **blind to stress**:
Whisper recognises `žeLEZnica` and `železNIca` as the same word, so WER sits at
the human floor whether the stress is right or wrong. There is no automatic
metric for "is the stress on the correct syllable," so the arbiter is a native
speaker's ear. What *is* deterministic and checkable is the **phoneme divergence**
below — the input the two voices are actually fed.

## The phoneme divergence (verified, reproducible)

Primary stress is `ˈ`, placed immediately before the stressed syllable. The
"correct" column is the Sloleks reference transcription, corroborated by the
native listener. Reproduce any row with the commands at the bottom.

| word | espeak-ng (upstream voice) | Sloleks (this voice) | what differs |
|---|---|---|---|
| **glagola** | `ɡlaɡˈoːla` — gla-**GÓ**-la | `ˈɡlaːɡɔla` — **GLÁ**-go-la | **wrong syllable** (2nd → 1st) |
| **slovenščino** | `slɔʋɛnʃtʃˈiːnɔ` — -**ščí**-no | `slɔˈʋeːnʃʧinɔ` — slo-**VÉN**-ščino | **wrong syllable** + vowel quality (`ɛ`→`eː`) |
| **železnica** | `ʒɛlɛznˈiːtsa` — -**NÍ**-ca | `ʒɛˈleːzniʦa` — že-**LÉZ**-nica | **wrong syllable** (3rd → 2nd) |
| **prijateljstvo** | `prijatˈeːʎstʋɔ` — -**TÉLJ**-stvo | `pɾiˈjaːtɛlʲstʋɔ` — pri-**JÁ**-teljstvo | **wrong syllable** (3rd → 2nd) |
| **knjižnico** | `kɲiʒnˈiːtsɔ` — knji-**ŽNÍ**-co | `ˈknjiːʒniʦɔ` — **KNJÍŽ**-nico | **wrong syllable** (3rd → 1st) |
| **nedoločnik** | `nɛdɔlˈoːtʃnik` — ne-do-**LÓČ**-nik | `ˈnɛdɔˈloːʧnik` — **NE**-do-**LÓČ**-nik | main stress OK; espeak drops the initial stress |
| **osnovna** | `ɔsnˈoːwna` — os-**NÓV**-na | `ɔˈsnɔːu̯na` — os-**NÓV**-na | right syllable, **wrong vowel** (`oː`→`ɔː`) |
| **oblika** | `ɔblˈiːka` — ob-**LÍ**-ka | `ɔbˈliːka` — ob-**LÍ**-ka | identical (espeak correct here) |
| **gledališče** | `ɡlɛdalˈiːʃtʃɛ` — gleda-**LÍ**-šče | `ɡlɛdaˈliːʃʧɛ` — gleda-**LÍ**-šče | right syllable (only `ʃtʃ`→`ʃʧ` notation) |

On this set, espeak places the primary stress on the **wrong syllable for 6 of 9
words**, colours a vowel wrong on a 7th, and matches on 2.

**This set is adversarial by construction** — these are words chosen *because*
espeak mis-handles them, so 6/9 is not the error rate on running text. The
repo-wide figure, measured over 4,000 random Sloleks forms, is **~74% of word
forms mis-stressed**; everyday high-frequency words are more often among the ones
espeak gets right, which is exactly why the two voices sound similar on common
speech (see "Honest scope" below).

## The headline result — s4

The test sentence **s4** was *"Nedoločnik je osnovna oblika glagola."* The
listener, given the five pairs blind, called s4 "much, much better" in the
lexicon voice — and flagged **glagola** specifically as wrong in the espeak
voice. Note that glagola was **not** the marked target word for that line
(nedoločnik was); the listener caught it unprompted. The phoneme table explains
why: in s4, three of the four content words are off in the espeak voice —

- **nedoločnik** — loses its initial stress (`NE-do-LÓČ-nik` → `ne-do-LÓČ-nik`)
- **osnovna** — stressed vowel mis-coloured (`ɔː` → `oː`)
- **glagola** — stressed on the **wrong syllable** (`GLÁ-go-la` → `gla-GÓ-la`)

— and only **oblika** is clean. The lexicon voice gets all four right because it
reads them from Sloleks.

## Honest scope — where the lexicon helps, and where it doesn't

A fair earlier A/B on **everyday sentences** sounded nearly identical between the
two voices. That is a real and expected result, not a failure: for high-frequency
words the model has seen thousands of times, it learns the correct prosody from
the ARTUR **audio** during training and reproduces it even when the espeak input
marks the stress wrong — it effectively memorises the word. The phoneme marker
only becomes the deciding factor for words the model has **not** memorised:
less-common vocabulary, technical and grammatical terms, place names, inflected
forms. There, the audio can't rescue a wrong marker, and the lexicon's
correctness carries through.

So the honest claim is **not** "better across the board." It is:

> On words with non-initial lexical stress that espeak guesses and frequently
> misses — less-common and technical vocabulary — the lexicon voice is correct
> and the upstream espeak voice is not. On everyday high-frequency words the two
> are close.

## Audio samples

The ten WAVs from this test (`official_s1..5.wav`, `ours_s1..5.wav`, 22.05 kHz)
are published as the release asset **`stress-comparison-samples.tar.gz`** so you
can hear it without a GPU. `official_*` is the upstream espeak voice; `ours_*` is
the lexicon voice.

## Reproduce

```bash
# espeak's actual phonemisation of any word:
espeak-ng -v sl -q --ipa "glagola"          # -> ɡlaɡˈoːla  (gla-GÓ-la)

# the Sloleks reference for the same word (from the bundled lexicon):
xzgrep -iP '^glagola\t' slovene-voice/lexicon/sl_pronunciation.tsv.xz | cut -f2
#                                            # -> ˈɡlaːɡɔla  (GLÁ-go-la)
```

The divergence is in the data, not in anyone's opinion. Everything above the
"headline result" is mechanically checkable; the listening verdict is the one
part that is, by necessity, a human judgement.
