# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

First public release.

### Added
- Light serving stack: the models now run in **official** containers
  (`rhasspy/wyoming-whisper`, `rhasspy/wyoming-piper`), with the only custom
  code a ~90-line phonemizer proxy (`serve/wyoming_phonemize_proxy.py`,
  `Dockerfile.proxy`). It turns Slovene text into piper's `[[ raw phoneme ]]`
  form — verified byte-identical to the built-in phonemizer — so official piper
  renders the voice correctly. STT needs no custom code at all.
- Slovene TTS voice (Piper/VITS, 22.05 kHz, epoch 729) as a portable,
  piper-free bundle that runs on CPU anywhere Python and onnxruntime run.
- Slovene speech recognition: whisper-large-v3 + LoRA fine-tuned on 853 h of
  ARTUR, 7.77% WER against 18.41% stock, deployed as CTranslate2
  `int8_float16`.
- Lexicon-first phonemizer using Sloleks 3.1 (2.9 M word forms), with espeak-ng
  as fallback for the ~1.34% of tokens it does not cover.
- English-in-Slovene handling: English words are detected and mapped onto the
  46 phonemes the model actually learned, rather than onto untrained symbols.
  Requires `lexicon/english_words.txt`, which the bundle builder now ships and
  refuses to build without.
- Model cards for both models, including their out-of-scope uses.
- Training documentation covering three data bugs that cost two training days
  and were invisible in the loss curve.
- An OpenAI-compatible HTTP API (`/v1/audio/speech`,
  `/v1/audio/transcriptions`) for Open WebUI, LibreChat and similar.
- A Wyoming protocol server, so Home Assistant discovers both models as
  Assist-pipeline services.
- Dockerfile and compose file, CPU by default with a CUDA profile.
