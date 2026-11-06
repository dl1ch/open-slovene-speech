"""python -m slovene_voice "besedilo" out.wav"""
import sys

from .synth import SloveneVoice

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    dur = SloveneVoice().to_wav(text, out)
    print("%s  %.2fs" % (out, dur))
