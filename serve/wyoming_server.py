#!/usr/bin/env python3
"""Wyoming protocol server — Home Assistant's native speech integration.

    python serve/wyoming_server.py --uri tcp://0.0.0.0:10200          # both
    python serve/wyoming_server.py --uri tcp://0.0.0.0:10200 --no-stt # voice only

Home Assistant discovers Wyoming services and uses them for Assist pipelines:
speech-to-text, text-to-speech, or both. Add it in HA under
Settings -> Devices & Services -> Add Integration -> Wyoming Protocol, and give
it the host and port this is listening on.

WHY NOT wyoming-piper: Home Assistant's piper add-on runs STOCK piper, which has
no PhonemeType.SLOVENE. Pointed at this voice it would phonemise with espeak,
and espeak places Slovene lexical stress on the wrong syllable in about 74% of
word forms. It would speak, fluently, and sound wrong to any Slovene speaker,
and nothing would error. That is the whole reason this server exists rather than
a voice file dropped into the piper add-on.

The recogniser has no such constraint - it is an ordinary Whisper model, so
wyoming-faster-whisper can serve it instead of this, if you prefer.
"""
import argparse
import asyncio
import io
import logging
import os
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import (
    AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice,
    TtsVoiceSpeaker,
)
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger("slovene-wyoming")

# Numbers leave the model as words; denormalise to written form.
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "stt"))
from denormalize import denormalize as _denormalize

# CC BY-SA 4.0: the attribution has to travel with the model, including here,
# where Home Assistant surfaces it in the UI.
ATTR = Attribution(
    name="ARTUR 1.0 (Verdonik et al., 2023) / Sloleks 3.1 (Cibej et al., 2026), CC BY-SA 4.0",
    url="https://github.com/dl1ch/open-slovene-speech",
)


class SloveneHandler(AsyncEventHandler):
    def __init__(self, *args, info, voice=None, asr=None, beam=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = info
        self.voice = voice
        self.asr = asr
        self.beam = beam
        self._audio = bytearray()
        self._rate = 16000
        self._width = 2
        self._channels = 1

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        # ---- text to speech ------------------------------------------------
        if Synthesize.is_type(event.type):
            if self.voice is None:
                _LOGGER.warning("synthesize requested but TTS is disabled")
                return True
            text = Synthesize.from_event(event).text
            pcm = await asyncio.get_running_loop().run_in_executor(
                None, self.voice.synthesize, text)
            rate = self.voice.sample_rate
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            # 1024 samples per chunk keeps latency low without flooding the link
            step = 1024 * 2
            raw = pcm.tobytes()
            for i in range(0, len(raw), step):
                await self.write_event(
                    AudioChunk(rate=rate, width=2, channels=1,
                               audio=raw[i:i + step]).event())
            await self.write_event(AudioStop().event())
            return True

        # ---- speech to text ------------------------------------------------
        if Transcribe.is_type(event.type):
            self._audio = bytearray()
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._rate, self._width, self._channels = start.rate, start.width, start.channels
            self._audio = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            self._audio.extend(AudioChunk.from_event(event).audio)
            return True

        if AudioStop.is_type(event.type):
            if self.asr is None:
                await self.write_event(Transcript(text="").event())
                return True
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(self._channels)
                w.setsampwidth(self._width)
                w.setframerate(self._rate)
                w.writeframes(bytes(self._audio))
            buf.seek(0)
            text = await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe, buf)
            await self.write_event(Transcript(text=text).event())
            return True

        return True

    def _transcribe(self, buf):
        segments, _ = self.asr.transcribe(buf, language="sl", beam_size=self.beam)
        text = " ".join(s.text.strip() for s in segments).strip()
        return _denormalize(text)


def build_info(tts: bool, stt: bool) -> Info:
    tts_programs, asr_programs = [], []
    if tts:
        tts_programs.append(TtsProgram(
            name="slovene-tts",
            description="Slovene voice (VITS, Sloleks phonemizer)",
            attribution=ATTR, installed=True, version="1.0",
            voices=[TtsVoice(
                name="sl_SI-artur-medium",
                description="Slovene, single speaker, 22.05 kHz",
                attribution=ATTR, installed=True, version="1.0",
                languages=["sl", "sl_SI"],
                speakers=[TtsVoiceSpeaker(name="artur")],
            )],
        ))
    if stt:
        asr_programs.append(AsrProgram(
            name="slovene-stt",
            description="Slovene ASR (whisper-large-v3 fine-tuned on ARTUR)",
            attribution=ATTR, installed=True, version="1.0",
            models=[AsrModel(
                name="slovene-whisper-ct2",
                description="7.77% WER on ARTUR, speaker-disjoint",
                attribution=ATTR, installed=True, version="1.0",
                languages=["sl"],
            )],
        ))
    return Info(tts=tts_programs, asr=asr_programs)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10200"))
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--stt-model", default=os.environ.get(
        "SLOVENE_STT_MODEL", "/models/slovene-whisper-ct2"))
    ap.add_argument("--stt-device", default=os.environ.get("SLOVENE_STT_DEVICE", "cuda"))
    ap.add_argument("--stt-compute", default=os.environ.get("SLOVENE_STT_COMPUTE", "int8_float16"))
    ap.add_argument("--beam", type=int, default=int(os.environ.get("SLOVENE_STT_BEAM", "2")))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO)

    voice = asr = None
    if not a.no_tts:
        from slovene_voice import SloveneVoice
        _LOGGER.info("loading voice (decompressing lexicon, ~3 s)")
        voice = SloveneVoice()
    if not a.no_stt:
        from faster_whisper import WhisperModel
        _LOGGER.info("loading recogniser from %s", a.stt_model)
        asr = WhisperModel(a.stt_model, device=a.stt_device, compute_type=a.stt_compute)

    if voice is None and asr is None:
        raise SystemExit("both --no-tts and --no-stt given: nothing to serve")

    info = build_info(tts=voice is not None, stt=asr is not None)
    server = AsyncServer.from_uri(a.uri)
    _LOGGER.info("listening on %s", a.uri)
    await server.run(lambda r, w: SloveneHandler(
        r, w, info=info, voice=voice, asr=asr, beam=a.beam))


if __name__ == "__main__":
    asyncio.run(main())
