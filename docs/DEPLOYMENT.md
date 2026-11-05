# Deployment

These are ordinary model files in standard formats. The recogniser works
anywhere Whisper works. The voice has one constraint, and it is the single most
important thing on this page.

---

## Read this first: the voice needs its own phonemizer

The recogniser is a plain Whisper fine-tune. Drop it into any Whisper runtime
and it behaves exactly as you expect.

**The voice is different.** It was trained on lexicon-derived pronunciations,
not espeak ones, because espeak-ng places Slovene lexical stress on the wrong
syllable in about **74% of word forms**.

Its config declares `"phoneme_type": "slovene"`, which upstream piper does not
know. Piper parses that field with `PhonemeType(config.get("phoneme_type", ...))`,
so an unknown value raises `ValueError` — **stock piper refuses to load this
voice at all.** That is the good case: it fails loudly at startup rather than
sounding subtly wrong.

The bad case is editing the config to say `espeak` to make it load. It will then
run, and be wrong in two ways at once: the stress will be misplaced, and the
phoneme *symbols* will be espeak's rather than the Sloleks notation the model was
trained on (`ʧ`, `ʦ`, `u̯`, `ɾ` …), which the model has never seen. Don't.

Use the servers in `serve/`, which carry the phonemizer.

---

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

| framework | status |
|---|---|
| **`serve/openai_api.py`** (this repo) | ✅ correct pronunciation |
| **`serve/wyoming_server.py`** (this repo) | ✅ correct pronunciation |
| **`slovene_voice` as a library** | ✅ correct pronunciation |
| stock **piper** / **wyoming-piper** | ❌ refuses to load (`ValueError: 'slovene' is not a valid PhonemeType`) |
| **openedai-speech** | ❌ same — it uses piper underneath |

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

### Text to speech — no

There is no mount-only path for this voice. `wyoming-piper` can load custom
voices from `--download-dir`, but it runs stock piper, which cannot phonemise
Slovene from a lexicon and refuses to load a voice whose `phoneme_type` it does
not recognise. See the warning at the top of this page.

This may not be permanent. Upstream piper has added `PhonemeType.LITHUANIAN`,
documented as *"espeak-ng IPA + pitch accent dictionary"* — architecturally the
same idea as this voice. A `SLOVENE` type upstream would make the voice usable
in any piper-based stack; see roadmap 3.4.

---

## Home Assistant

Two options. The Wyoming server is the better one because it is the only way to
get the voice pronounced correctly.

### Option A — Wyoming (recommended, both STT and TTS)

```bash
scripts/fetch-models.sh
docker compose up -d wyoming
```

Then in Home Assistant: **Settings → Devices & Services → Add Integration →
Wyoming Protocol**, host = the machine running it, port = `10200`.

Both a speech-to-text and a text-to-speech service appear, and can be selected
in any Assist pipeline under **Settings → Voice assistants**.

Run only one half if you want:

```bash
python3 serve/wyoming_server.py --uri tcp://0.0.0.0:10200 --no-stt   # voice only
python3 serve/wyoming_server.py --uri tcp://0.0.0.0:10200 --no-tts   # recogniser only
```

### Option B — the existing add-ons

If you already run the **Whisper** add-on (wyoming-faster-whisper), you can
point it at this model instead of converting anything: place
`slovene-whisper-ct2/` in its model directory and select it as a custom model.
That is a fully supported path — the recogniser is a normal Whisper model.

For TTS, the **Piper** add-on is *not* a supported path for this voice. See the
warning at the top.

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
