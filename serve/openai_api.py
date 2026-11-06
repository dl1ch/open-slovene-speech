#!/usr/bin/env python3
"""OpenAI-compatible audio API for the Slovene voice and recogniser.

    uvicorn serve.openai_api:app --host 0.0.0.0 --port 8080

Implements the two endpoints most software already speaks:

    POST /v1/audio/speech            text  -> audio   (Open WebUI, many clients)
    POST /v1/audio/transcriptions    audio -> text

That covers Open WebUI, LibreChat, AnythingLLM, Home Assistant's OpenAI
integrations, and anything else written against the OpenAI audio API. For Home
Assistant's native Wyoming protocol use serve/wyoming_server.py instead.

WHY NOT A STOCK openedai-speech CONTAINER: this voice does not use espeak. It
was trained on lexicon-derived pronunciations, and feeding it espeak phonemes
produces fluent Slovene with the stress wrong in about three quarters of words,
with no error raised. Any server for this voice must carry the phonemizer in
slovene_voice/, which is why this file exists.

Either half can be disabled, so this runs as a TTS-only box with no GPU:

    SLOVENE_ENABLE_STT=0    # skip loading faster-whisper
    SLOVENE_ENABLE_TTS=0    # skip loading the voice
"""
import io
import os
import subprocess
import tempfile
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

ENABLE_TTS = os.environ.get("SLOVENE_ENABLE_TTS", "1") != "0"
ENABLE_STT = os.environ.get("SLOVENE_ENABLE_STT", "1") != "0"
# Numbers leave the model as words; the wrapper turns them into written
# form and is a no-op if the tool is not installed.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "stt"))
from denormalize import denormalize as _denormalize
STT_MODEL = os.environ.get("SLOVENE_STT_MODEL", "/models/slovene-whisper-ct2")
STT_DEVICE = os.environ.get("SLOVENE_STT_DEVICE", "cuda")
STT_COMPUTE = os.environ.get("SLOVENE_STT_COMPUTE", "int8_float16")
# beam 2 is the measured knee: 8.19% WER at 5.3x realtime, against 8.12% at
# 4.2x for beam 5. Larger beams cost speed and buy almost nothing.
STT_BEAM = int(os.environ.get("SLOVENE_STT_BEAM", "2"))

app = FastAPI(title="Slovene speech", version="1.0")
_voice = None
_asr = None


def voice():
    """Load the voice once. Construction decompresses the lexicon (~3 s)."""
    global _voice
    if _voice is None:
        from slovene_voice import SloveneVoice
        _voice = SloveneVoice()
    return _voice


def asr():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        _asr = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE)
    return _asr


@app.get("/health")
def health():
    return {"status": "ok", "tts": ENABLE_TTS, "stt": ENABLE_STT}


@app.get("/v1/models")
def models():
    out = []
    if ENABLE_TTS:
        out.append({"id": "slovene-tts", "object": "model", "owned_by": "slovene-speech"})
    if ENABLE_STT:
        out.append({"id": "slovene-stt", "object": "model", "owned_by": "slovene-speech"})
    return {"object": "list", "data": out}


def _to_mp3(wav_bytes):
    """WAV -> MP3 via ffmpeg. Clients that ask for mp3 usually mean it."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
             "-f", "mp3", "-b:a", "96k", "pipe:1"],
            input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return p.stdout
    except (OSError, subprocess.CalledProcessError):
        # Serving WAV bytes under an mp3 content-type is how you get silent
        # playback failures downstream. Say what actually happened instead.
        raise HTTPException(
            status_code=503,
            detail="mp3 requested but ffmpeg is unavailable; request "
                   "response_format=wav or install ffmpeg")


@app.post("/v1/audio/speech")
async def speech(payload: dict):
    if not ENABLE_TTS:
        raise HTTPException(status_code=404, detail="TTS disabled")
    text = (payload.get("input") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'input' is required")
    fmt = (payload.get("response_format") or "mp3").lower()

    v = voice()
    pcm = v.synthesize(text)            # int16 samples, no temp file needed
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(v.sample_rate)
        w.writeframes(pcm.tobytes())
    data = buf.getvalue()

    if fmt in ("wav", "pcm"):
        return Response(content=data, media_type="audio/wav")
    if fmt in ("mp3", "mpeg"):
        return Response(content=_to_mp3(data), media_type="audio/mpeg")
    raise HTTPException(status_code=400, detail="unsupported response_format: %s" % fmt)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("slovene-stt"),
    language: str = Form("sl"),
    response_format: str = Form("json"),
):
    if not ENABLE_STT:
        raise HTTPException(status_code=404, detail="STT disabled")
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        t.write(await file.read())
        path = t.name
    try:
        # language is pinned to Slovene: this model is specialised, and letting
        # Whisper auto-detect on a short or noisy clip picks the wrong one.
        segments, _info = asr().transcribe(path, language=language or "sl",
                                           beam_size=STT_BEAM)
        text = _denormalize(" ".join(s.text.strip() for s in segments).strip())
    finally:
        os.unlink(path)

    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    return JSONResponse({"text": text})
