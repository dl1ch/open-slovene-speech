#!/usr/bin/env bash
# Download the model weights from the GitHub Release.
#
# Weights are not in the git tree: the STT model is 1.5 GB and the lexicon is
# 81 MB uncompressed. They are Release assets, under CC BY-SA 4.0 (see
# LICENSE-MODELS.md), while this script and the rest of the code are MIT.
#
#   scripts/fetch-models.sh            # everything
#   scripts/fetch-models.sh tts        # just the voice (69 MB)
#   scripts/fetch-models.sh stt        # just the recogniser (1.5 GB)
set -euo pipefail

REPO="${SLOVENE_SPEECH_REPO:-dl1ch/open-slovene-speech}"
TAG="${SLOVENE_SPEECH_TAG:-latest}"
DEST="${SLOVENE_SPEECH_DEST:-$(cd "$(dirname "$0")/.." && pwd)/models}"
WHAT="${1:-all}"

mkdir -p "$DEST"

assets=()
case "$WHAT" in
  tts) assets=(slovene-voice.tar.gz) ;;
  stt) assets=(slovene-whisper-ct2.tar.gz slovene-denormalizer.tar) ;;
  lora) assets=(slovene-whisper-lora.tar.gz) ;;
  denorm) assets=(slovene-denormalizer.tar) ;;
  # stt pulls the denormaliser too: it is 8.9 MB and turns spoken-form numbers
  # into written form, which is what most ASR uses want.
  all) assets=(slovene-voice.tar.gz slovene-whisper-ct2.tar.gz slovene-denormalizer.tar) ;;
  *) echo "usage: $0 [all|tts|stt|lora|denorm]" >&2; exit 2 ;;
esac

fetch () {
  local name="$1"
  if command -v gh >/dev/null 2>&1; then
    echo "==> $name (gh)"
    gh release download "$TAG" --repo "$REPO" --pattern "$name" --dir "$DEST" --clobber
  else
    local url
    if [ "$TAG" = "latest" ]; then
      url="https://github.com/$REPO/releases/latest/download/$name"
    else
      url="https://github.com/$REPO/releases/download/$TAG/$name"
    fi
    echo "==> $name (curl)"
    curl -fL --progress-bar -o "$DEST/$name" "$url"
  fi
}

for a in "${assets[@]}"; do fetch "$a"; done

# Checksums are published as one file covering every asset in the release.
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "SHA256SUMS" \
     --dir "$DEST" --clobber 2>/dev/null || true
else
  curl -fsL -o "$DEST/SHA256SUMS" \
    "https://github.com/$REPO/releases/latest/download/SHA256SUMS" 2>/dev/null || true
fi

# sha256sum is the GNU/Linux tool, shasum the macOS/perl one. Most Linux
# containers have neither guaranteed, so pick whichever exists rather than
# assuming - an unverified download is worse than a slow one.
if command -v sha256sum >/dev/null 2>&1; then
  SHACHECK="sha256sum -c -"
elif command -v shasum >/dev/null 2>&1; then
  SHACHECK="shasum -a 256 -c -"
else
  SHACHECK=""
fi

if [ -f "$DEST/SHA256SUMS" ] && [ -n "$SHACHECK" ]; then
  echo "==> verifying"
  ( cd "$DEST"
    # check only what we actually downloaded, and fail loudly if it mismatches
    for a in "${assets[@]}"; do
      if grep -q "  $a\$" SHA256SUMS; then
        grep "  $a\$" SHA256SUMS | $SHACHECK
      else
        echo "    no checksum published for $a" >&2
      fi
    done )
elif [ -f "$DEST/SHA256SUMS" ]; then
  echo "    WARNING: neither sha256sum nor shasum found - NOT verified" >&2
else
  echo "    WARNING: no SHA256SUMS in the release - contents unverified" >&2
fi

echo "==> unpacking"
for a in "${assets[@]}"; do
  case "$a" in
    *.tar.gz) tar xzf "$DEST/$a" -C "$DEST" ;;
    *.tar)    tar xf  "$DEST/$a" -C "$DEST" ;;
  esac
done
# The RSDO denormaliser extracts to Slovene_denormalizator-1.0/; the wrapper
# looks for it as slovene-denormalizer/. Rename, and note the runtime deps.
if [ -d "$DEST/Slovene_denormalizator-1.0" ]; then
  rm -rf "$DEST/slovene-denormalizer"
  mv "$DEST/Slovene_denormalizator-1.0" "$DEST/slovene-denormalizer"
  echo "    denormaliser needs: pip install nltk flask flask_restx"
fi

# Set up the voice for the OFFICIAL piper container: the ONNX plus an espeak-mode
# config, so piper takes its raw-phoneme branch for the [[ ]] input the proxy
# sends. This is what lets the light stack use official wyoming-piper.
if [ -d "$DEST/slovene-voice" ]; then
  _tools="$(cd "$(dirname "$0")/.." && pwd)/tools"
  mkdir -p "$DEST/piper-voice"
  cp "$DEST/slovene-voice/model/sl_SI-artur-medium.onnx" "$DEST/piper-voice/"
  if command -v python3 >/dev/null 2>&1 && [ -f "$_tools/make_espeak_config.py" ]; then
    python3 "$_tools/make_espeak_config.py" \
      "$DEST/slovene-voice/model/sl_SI-artur-medium.onnx.json" >/dev/null
    cp "$DEST/slovene-voice/model/sl_SI-artur-medium.espeak.onnx.json" \
       "$DEST/piper-voice/sl_SI-artur-medium.onnx.json"
    echo "    piper voice ready in $DEST/piper-voice (espeak/raw-phoneme mode)"
  else
    echo "    WARNING: could not build the piper espeak config (need python3 + tools/)" >&2
  fi
fi

cat <<EOF

Done. Models are in: $DEST

  TTS   export SLOVENE_VOICE_MODEL=$DEST/slovene-voice/model/sl_SI-artur-medium.onnx
        export SLOVENE_LEXICON=$DEST/slovene-voice/lexicon/sl_pronunciation.tsv.xz
  STT   WhisperModel("$DEST/slovene-whisper-ct2", compute_type="int8_float16")

These weights are CC BY-SA 4.0 and derive from ARTUR 1.0 and Sloleks 3.1.
If you redistribute them or anything trained from them, keep the attribution in
NOTICE and license your version under CC BY-SA 4.0 as well.
EOF
