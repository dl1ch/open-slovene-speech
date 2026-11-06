"""Slovene voice package.

SloveneVoice is imported lazily: the phonemizer modules (phonemize,
raw_phonemes) must be usable without numpy or onnxruntime, so the tiny
phonemizer proxy image can import them without the synthesis stack.
"""
__all__ = ["SloveneVoice"]


def __getattr__(name):           # PEP 562
    if name == "SloveneVoice":
        from .synth import SloveneVoice
        return SloveneVoice
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
