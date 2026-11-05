# Slovene speech: OpenAI-compatible API and/or Wyoming for Home Assistant.
#
#   docker build -t slovene-speech .
#   docker run -p 8080:8080 -v ./models:/models slovene-speech
#
# CPU by default. For GPU recognition, build with:
#   docker build --build-arg BASE=nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04 .
ARG BASE=python:3.11-slim
FROM ${BASE}

# espeak-ng is NOT optional in practice: it is the fallback for words missing
# from Sloleks (~1.34% of running text), and without it those words are silently
# DROPPED from the output rather than mispronounced.
# ffmpeg is needed only to answer requests for mp3; wav works without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY serve/requirements.txt /app/serve/requirements.txt
RUN python3 -m pip install --no-cache-dir -r serve/requirements.txt

COPY tts/ /app/tts/
COPY serve/ /app/serve/
COPY stt/ /app/stt/
COPY NOTICE LICENSE LICENSE-MODELS.md /app/

# The package lives under tts/, and the voice resolves its model and lexicon
# relative to that directory.
ENV PYTHONPATH=/app/tts \
    SLOVENE_VOICE_MODEL=/models/slovene-voice/model/sl_SI-artur-medium.onnx \
    SLOVENE_LEXICON=/models/slovene-voice/lexicon/sl_pronunciation.tsv.xz \
    SLOVENE_USER_LEXICON=/models/slovene-voice/lexicon/user_lexicon.tsv \
    SLOVENE_ENGLISH_WORDS=/models/slovene-voice/lexicon/english_words.txt \
    SLOVENE_STT_MODEL=/models/slovene-whisper-ct2 \
    SLOVENE_STT_DEVICE=cpu \
    SLOVENE_STT_COMPUTE=int8 \
    SLOVENE_NORMALIZE=0

EXPOSE 8080 10200
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["uvicorn", "serve.openai_api:app", "--host", "0.0.0.0", "--port", "8080"]
