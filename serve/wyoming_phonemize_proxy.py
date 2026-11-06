#!/usr/bin/env python3
"""Tiny Wyoming TTS proxy: Slovene lexicon phonemisation in front of piper.

    python serve/wyoming_phonemize_proxy.py \
        --uri tcp://0.0.0.0:10200 --piper-uri tcp://piper:10200

This is the whole of the custom TTS code. It does NOT synthesise audio — the
official wyoming-piper container does that. All this does is turn Slovene text
into piper's `[[ raw phoneme ]]` form, using the Sloleks lexicon, and relay the
request through. Verified: the phoneme IDs are byte-identical to the voice's own
phonemizer, so piper renders exactly what it would with a built-in Slovene type.

Home Assistant talks to THIS; this talks to piper. The voice mounted into piper
must be configured phoneme_type="espeak" (tools/make_espeak_config.py), so piper
takes its raw-phoneme branch and never runs espeak's G2P.

Why this exists at all: piper has no Slovene lexicon phonemizer and no plugin
hook, and no other TTS server does either. Until a PhonemeType.SLOVENE lands
upstream (then this disappears and you point HA straight at wyoming-piper), the
lexicon lookup has to run somewhere. This is the smallest place to put it.
"""
import argparse
import asyncio
import logging
import os
import sys

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tts"))
from slovene_voice.raw_phonemes import to_raw

_LOGGER = logging.getLogger("slovene-phonemize-proxy")
ATTR = Attribution(name="ARTUR 1.0 / Sloleks 3.1, CC BY-SA 4.0",
                   url="https://github.com/dl1ch/open-slovene-speech")


class ProxyHandler(AsyncEventHandler):
    def __init__(self, *args, piper_host, piper_port, info, **kwargs):
        super().__init__(*args, **kwargs)
        self.piper_host = piper_host
        self.piper_port = piper_port
        self.info = info

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if not Synthesize.is_type(event.type):
            return True

        syn = Synthesize.from_event(event)
        raw = to_raw(syn.text)                    # <-- the only real work here
        _LOGGER.debug("text %r -> %r", syn.text, raw)

        # forward to the official piper, relay its audio back unchanged
        async with AsyncTcpClient(self.piper_host, self.piper_port) as piper:
            await piper.write_event(Synthesize(text=raw, voice=syn.voice).event())
            while True:
                ev = await piper.read_event()
                if ev is None:
                    break
                await self.write_event(ev)
                if AudioStop.is_type(ev.type):
                    break
        return True


def build_info():
    return Info(tts=[TtsProgram(
        name="slovene-tts",
        description="Slovene voice (lexicon phonemizer + piper)",
        attribution=ATTR, installed=True, version="1.0",
        voices=[TtsVoice(name="sl_SI-artur-sloleks-medium",
                         description="Slovene, single speaker, 22.05 kHz",
                         attribution=ATTR, installed=True, version="1.0",
                         languages=["sl", "sl_SI"],
                         speakers=[TtsVoiceSpeaker(name="artur")])],
    )])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("PROXY_URI", "tcp://0.0.0.0:10200"))
    ap.add_argument("--piper-uri", default=os.environ.get("PIPER_URI", "tcp://127.0.0.1:10201"))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO)

    # parse tcp://host:port
    hp = a.piper_uri.replace("tcp://", "").split(":")
    host, port = hp[0], int(hp[1])
    info = build_info()
    _LOGGER.info("warming the phonemizer (lexicon load ~2 s)")
    to_raw("Pozdravljen.")
    server = AsyncServer.from_uri(a.uri)
    _LOGGER.info("proxy on %s -> piper at %s:%s", a.uri, host, port)
    await server.run(lambda r, w: ProxyHandler(
        r, w, piper_host=host, piper_port=port, info=info))


if __name__ == "__main__":
    asyncio.run(main())
