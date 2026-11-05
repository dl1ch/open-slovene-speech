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

REPO="${SLOVENE_SPEECH_REPO:-dl1ch/slovene-speech}"
TAG="${SLOVENE_SPEECH_TAG:-latest}"
DEST="${SLOVENE_SPEECH_DEST:-$(cd "$(dirname "$0")/.." && pwd)/models}"
WHAT="${1:-all}"

mkdir -p "$DEST"

assets=()
case "$WHAT" in
  tts) assets=(slovene-voice.tar.gz) ;;
  stt) assets=(slovene-whisper-ct2.tar.gz) ;;
  lora) assets=(slovene-whisper-lora.tar.gz) ;;
  all) assets=(slovene-voice.tar.gz slovene-whisper-ct2.tar.gz) ;;
  *) echo "usage: $0 [all|tts|stt|lora]" >&2; exit 2 ;;
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
  tar xzf "$DEST/$a" -C "$DEST"
done

cat <<EOF

Done. Models are in: $DEST

  TTS   export SLOVENE_VOICE_MODEL=$DEST/slovene-voice/model/sl_SI-artur-medium.onnx
        export SLOVENE_LEXICON=$DEST/slovene-voice/lexicon/sl_pronunciation.tsv.xz
  STT   WhisperModel("$DEST/slovene-whisper-ct2", compute_type="int8_float16")

These weights are CC BY-SA 4.0 and derive from ARTUR 1.0 and Sloleks 3.1.
If you redistribute them or anything trained from them, keep the attribution in
NOTICE and license your version under CC BY-SA 4.0 as well.
EOF
