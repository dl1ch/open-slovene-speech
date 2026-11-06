# Deployment

These are ordinary model files in standard formats. The recogniser works
anywhere Whisper works. The voice has one constraint, and it is the single most
important thing on this page.

---

## The short version

Both models run in **official containers**; the only custom code is one tiny
phonemizer. There is no heavy bespoke server to maintain.

| | container | custom code |
|---|---|---|
| **STT** | official `rhasspy/wyoming-whisper`, model mounted | **none** |
| **TTS synthesis** | official `rhasspy/wyoming-piper`, voice mounted | **none** |
| **TTS phonemization** | a ~90-line proxy (`Dockerfile.proxy`) | one small piece |

```bash
scripts/fetch-models.sh
docker compose up -d          # Home Assistant stack: STT :10300, TTS :10200
```

### Why one custom piece remains

The voice needs Slovene **lexicon** phonemization — espeak mis-stresses ~74% of
Slovene words, so the voice was trained on dictionary pronunciations. No TTS
server, piper included, ships a Slovene lexicon phonemizer, and piper has no
plugin hook to add one. So the lexicon lookup has to run *somewhere*.

The trick that keeps it tiny: piper accepts `[[ raw phonemes ]]` in its input,
and — **verified byte-identical** — feeding it our lexicon's output that way
produces exactly what a built-in Slovene phonemizer would. So the custom piece
is just text → `[[ipa]]`; the official piper does the synthesis. It holds the
lexicon and does string work: no model, no onnxruntime, no GPU.

When a `PhonemeType.SLOVENE` lands upstream in piper (roadmap 3.4), even this
disappears and you point Home Assistant straight at `wyoming-piper`.

## Compatibility

### Speech to text — works everywhere

| framework | format | status |
|---|---|---|
| **faster-whisper** | CTranslate2 | ✅ shipped as `slovene-whisper-ct2` |
| **wyoming-faster-whisper** (Home Assistant) | CTranslate2 | ✅ point it at the model directory |
| **Speaches** / faster-whisper-server | CTranslate2 | ✅ |
| **WhisperX** (alignment, diarisation) | CTranslate2 | ✅ |
| **transformers** / PEFT | HF + LoRA adapter | ✅ shipped as `slovene-whisper-lora` |
| **whisper.cpp** / GGML | GGUF | ⚠️ needs conversion — see roadmap 1.5 |
| **openai-whisper** (reference) | `.pt` | ⚠️ needs conversion |

### Text to speech

| setup | status |
|---|---|
| **official wyoming-piper + phonemizer proxy** (the light stack) | ✅ correct, verified against wyoming-piper 2.5.2 |
| **`slovene_voice` as a library** | ✅ correct pronunciation |
| **`serve/openai_api.py`** / **`serve/wyoming_server.py`** (optional all-in-one) | ✅ correct pronunciation |
| stock **piper** / **wyoming-piper** fed plain text (no proxy) | ⚠️ loads via `make_espeak_config.py`, then mis-stresses ~74% |
| **openedai-speech** | ⚠️ same — it uses piper underneath, no lexicon |

---

## Using official containers, mounting only the model

You do not have to run the servers in this repository for speech-to-text. You
do for text-to-speech. The asymmetry is worth stating precisely.

### Speech to text — yes, this works

`wyoming-faster-whisper` (the image behind Home Assistant's Whisper add-on)
accepts a local directory as `--model`. The value is passed through untouched —
only the `tiny|base|small|medium[.-]int8` aliases are rewritten — and
`faster_whisper.WhisperModel` checks `os.path.isdir()` before deciding to
download. So a mounted CTranslate2 directory is loaded from disk:

```bash
docker run -it -p 10300:10300 \
  -v /path/to/data:/data \
  -v $PWD/models/slovene-whisper-ct2:/models/slovene:ro \
  rhasspy/wyoming-whisper \
  --data-dir /data --model /models/slovene --language sl
```

Then add it in Home Assistant as a Wyoming service on port **10300**.

Four things that will bite you:

- **Use the in-container path** (`/models/slovene`), not the host path. If the
  directory does not exist inside the container, the string is treated as a
  Hugging Face model ID and the container tries to download it.
- **`--data-dir` is required** even though nothing is downloaded.
- **Keep `tokenizer.json` in the directory.** If it is missing, faster-whisper
  silently falls back to downloading `openai/whisper-tiny`'s tokenizer, which is
  not the right one.
- **Check `preprocessor_config.json` is present.** large-v3 uses 128 mel bins
  rather than the 80 of earlier Whisper models, and that file is what says so.

The same mounted directory works with Speaches, WhisperX, and
`openai-whisper-asr-webservice`, since they all sit on faster-whisper. Note that
the Speaches docs describe mounting a Hugging Face cache rather than a model
directory, so check its configuration before assuming.

### Text to speech — official piper plus the phonemizer proxy

The official `wyoming-piper` renders the voice correctly, **provided the text
reaches it already phonemized** as `[[ ipa ]]`. The proxy does exactly that, so
the compose file wires them together:

```
Home Assistant ──text──▶ tts proxy ──[[ipa]]──▶ wyoming-piper ──audio──▶ back
                         (lexicon)               (official)
```

`scripts/fetch-models.sh` prepares `models/piper-voice/` — the voice ONNX plus
an **espeak-mode config** (via `tools/make_espeak_config.py`) so piper takes its
raw-phoneme branch. You do not edit anything by hand.

**Verified end to end.** The whole chain was tested against the real
`wyoming-piper` software (2.5.2, which the official image packages): it loads
the mounted custom voice and reports `Ready`, the proxy turns text into the
right `[[ipa]]`, piper maps that to byte-identical phoneme IDs, and audio of the
correct length comes back — matching a direct render within VITS's stochastic
variance. The only thing not exercised was the Docker *packaging* itself (no
container runtime was available on the test host); the software inside the image
is exactly what was tested.

**Loading the voice into stock piper WITHOUT the proxy** still works but is
wrong: `tools/make_espeak_config.py` lets stock piper load it, and it will then
phonemize with espeak and mis-stress ~74% of words. Only the proxy path feeds it
lexicon phonemes.

## Home Assistant

```bash
scripts/fetch-models.sh
docker compose up -d
```

That brings up three services — official whisper, official piper, and the tiny
phonemizer proxy. Then in Home Assistant: **Settings → Devices & Services → Add
Integration → Wyoming Protocol**, twice:

| add a Wyoming service at | gives you |
|---|---|
| `host:10300` | Slovene speech-to-text |
| `host:10200` | Slovene text-to-speech |

Both appear for any Assist pipeline under **Settings → Voice assistants**.

### STT alone, via the existing Whisper add-on

If you already run the **Whisper** add-on (wyoming-faster-whisper), skip our
whisper container entirely: drop `slovene-whisper-ct2/` into its model directory
and select it as a custom model. Fully supported — it is a normal Whisper model.

For TTS, the **Piper add-on cannot** serve this voice on its own: it has no
Slovene lexicon phonemizer. Use the `docker compose` stack above, whose piper
container *is* the official piper — just fronted by the phonemizer proxy.

### Single-container alternative

If you would rather run one container than three, `serve/openai_api.py` (the
`openai` compose profile) and `serve/wyoming_server.py` are self-contained
servers that do everything in-process with no official containers. Heavier, but
one moving part. The light stack above is the recommended default.

### A note on Assist pipelines

Set the pipeline language to Slovene. The recogniser is pinned to `sl` in these
servers deliberately: it is specialised to Slovene, and letting Whisper
auto-detect on a short or noisy command is a reliable way to get the wrong
language.

---

## Open WebUI

Open WebUI speaks the OpenAI audio API, which is what `serve/openai_api.py`
implements.

```bash
scripts/fetch-models.sh
docker compose up -d api
```

In Open WebUI, **Settings → Audio**:

| setting | value |
|---|---|
| STT Engine | OpenAI |
| STT Base URL | `http://<host>:8080/v1` |
| STT Model | `slovene-stt` |
| TTS Engine | OpenAI |
| TTS Base URL | `http://<host>:8080/v1` |
| TTS Model | `slovene-tts` |
| TTS Voice | `artur` |

The API key field is not checked by this server; put anything in it.

If Open WebUI runs in its own container, `localhost` will not reach the host —
use the host's LAN address, or `host.docker.internal` where that is available.

---

## Anything else that speaks the OpenAI audio API

LibreChat, AnythingLLM, custom clients, and Home Assistant's OpenAI
integrations all work against the same two endpoints:

```bash
# text -> speech
curl -s http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"slovene-tts","input":"Pozdravljen, svet.","response_format":"wav"}' \
  -o out.wav

# speech -> text
curl -s http://localhost:8080/v1/audio/transcriptions \
  -F file=@posnetek.wav -F model=slovene-stt
```

`response_format` accepts `wav` and `mp3`. mp3 needs ffmpeg; if it is missing
the server returns a 503 saying so rather than serving WAV bytes mislabelled as
mp3, which causes silent playback failures downstream.

---

## As a Python library

No server at all:

```python
from slovene_voice import SloveneVoice
v = SloveneVoice()                       # ~3 s: decompresses the lexicon
v.to_wav("Danes je lep dan.", "out.wav") # keep the instance, do not rebuild it

from faster_whisper import WhisperModel
m = WhisperModel("models/slovene-whisper-ct2", device="cuda",
                 compute_type="int8_float16")
segments, _ = m.transcribe("posnetek.wav", language="sl", beam_size=2)
print(" ".join(s.text for s in segments))
```

---

## With transformers (the LoRA adapter)

For further fine-tuning, or if you want the unquantised model:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base, "models/slovene-whisper-lora")
model = model.merge_and_unload()         # optional: fold the adapter in
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
```

Budget about 4 GB of VRAM for fp16 inference this way — considerably more than
the 2 GB the CTranslate2 build needs.

---

## Environment variables

| variable | default | meaning |
|---|---|---|
| `SLOVENE_VOICE_MODEL` | bundle-relative | path to the `.onnx` |
| `SLOVENE_LEXICON` | bundle-relative | Sloleks lookup table (`.tsv.xz`) |
| `SLOVENE_USER_LEXICON` | bundle-relative | your pronunciation overrides |
| `SLOVENE_ENGLISH_WORDS` | bundle-relative | English detection list; **absent = feature silently off** |
| `SLOVENE_ENGLISH` | `1` | `0` disables embedded-English handling |
| `SLOVENE_NORMALIZE` | `1` | `0` disables the digit-expansion HTTP call |
| `SLOVENE_NORMALIZER_URL` | `http://127.0.0.1:8081/normalize` | normaliser endpoint |
| `SLOVENE_STT_MODEL` | `/models/slovene-whisper-ct2` | recogniser directory |
| `SLOVENE_STT_DEVICE` | `cuda` | `cpu` works, roughly realtime |
| `SLOVENE_STT_COMPUTE` | `int8_float16` | use `int8` on CPU |
| `SLOVENE_STT_BEAM` | `2` | the measured knee |
| `SLOVENE_DENORMALIZE` | `1` | `0` keeps ASR numbers as words |
| `SLOVENE_DENORMALIZER_DIR` | `models/slovene-denormalizer` | the RSDO tool |
| `SLOVENE_ENABLE_TTS` / `_STT` | `1` | run half the API only |

### Digits

Neither server expands digits by default (`SLOVENE_NORMALIZE=0` in the
container). The voice has never seen a digit — ARTUR's training text contains
none — so `27. 9. 2026` needs expanding to
*sedemindvajsetega devetega dva tisoč šestindvajset*, with case agreement.

Either expand them yourself before calling, or run a normalisation service
based on RSDO-DS2-NORM (Apache 2.0, http://hdl.handle.net/11356/1742) and point
`SLOVENE_NORMALIZER_URL` at it. It depends on `classla`, which pulls in torch
and neural models — about 5 GB — which is why it is not in the container.
