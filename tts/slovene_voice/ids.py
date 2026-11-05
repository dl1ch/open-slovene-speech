"""Phoneme strings -> model input ids.

Reproduced from piper's phoneme_ids.py. The model was trained on this exact
interleaving, so it is not a free choice: BOS, PAD, then every phoneme followed
by PAD, then EOS. Getting it wrong yields audio that is not silent but is wrong,
which is the worst failure mode to debug.
"""
PAD, BOS, EOS = "_", "^", "$"


def phonemes_to_ids(phonemes, id_map):
    ids = list(id_map[BOS]) + list(id_map[PAD])
    for p in phonemes:
        if p not in id_map:
            continue  # unmapped symbol: skip, exactly as piper does
        ids.extend(id_map[p])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids
