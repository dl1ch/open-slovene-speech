"""Render Slovene text as piper [[ raw phoneme ]] blocks.

This is the key to serving the voice from the OFFICIAL piper container instead
of a custom server. Piper lets text carry `[[ <phonemes> ]]` blocks that bypass
espeak; verified byte-identical phoneme IDs to this voice's own phonemizer. So a
tiny text->text step can run the Slovene lexicon lookup and hand piper fully
phonemised input, and official wyoming-piper does the synthesis.

    from slovene_voice.raw_phonemes import to_raw
    to_raw("Črna večerja.")  ->  "[[ ˈtʃəɾna ʋɛˈtʃeːɾja. ]]"

The voice must then be configured with phoneme_type "espeak" (make_espeak_config
writes that), so piper takes the raw-phoneme branch.
"""
from .phonemize import SlovenePhonemizer

_PHON = None


def _phonemizer():
    global _PHON
    if _PHON is None:
        _PHON = SlovenePhonemizer()
    return _PHON


def to_raw(text):
    """Slovene text -> a single piper [[ ... ]] raw-phoneme string."""
    if not text or not text.strip():
        return text
    sents = _phonemizer().phonemize(text)
    phon = "".join("".join(s) for s in sents)
    if not phon.strip():
        return text
    return "[[ %s ]]" % phon
