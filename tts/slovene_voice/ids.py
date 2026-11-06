"""Phoneme strings -> model input ids.

Reproduced from piper's phoneme_ids.py. The model was trained on this exact
interleaving, so it is not a free choice: BOS, PAD, then every phoneme followed
by PAD, then EOS. Getting it wrong yields audio that is not silent but is wrong,
which is the worst failure mode to debug.

The id map is NOT a fixed property of piper. It travels in each voice's config
JSON, and piper falls back to a 166-symbol default only when a voice supplies
none. There is no cap on its size. A retrained voice may therefore declare any
inventory it likes - see docs/ROADMAP.md 2.2.
"""
import logging

_LOGGER = logging.getLogger(__name__)

PAD, BOS, EOS = "_", "^", "$"


def phonemes_to_ids(phonemes, id_map):
    ids = list(id_map[BOS]) + list(id_map[PAD])
    for p in phonemes:
        if p not in id_map:
            # Upstream piper warns here rather than dropping in silence, and so
            # do we. A phoneme absent from the map simply vanishes from the
            # audio, which is invisible unless you diff phoneme output.
            _LOGGER.warning("phoneme missing from the id map, dropped: %r", p)
            continue
        ids.extend(id_map[p])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids
