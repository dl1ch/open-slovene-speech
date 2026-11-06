#!/usr/bin/env python3
"""Build a portable, piper-free bundle of the Slovene voice.

The ONNX model takes phoneme IDs, not text, so shipping the .onnx alone is not
enough - the text->IPA->ids half has to travel with it. That half is currently a
patch to piper (PhonemeType.SLOVENE does not exist upstream), which is exactly
what makes the voice non-portable today.

NOT A SCRIPT MOST PEOPLE NEED. It rebuilds the published TTS bundle from a
PATCHED piper installation - the one carrying PhonemeType.SLOVENE, which does
not exist upstream. Without that installation it has nothing to read. It is
here as provenance: it records exactly how the released bundle was produced.

To just USE the voice, download the bundle (scripts/fetch-models.sh tts); you
do not need this file. All inputs come from PIPER_PKG, MODEL_SRC and LEX_SRC;
there are no defaults, so nothing depends on one machine's directory layout.

This extracts the three pieces that actually matter and drops piper entirely:

  phonemize.py  derived from the patched piper module, with two substitutions -
                espeak reached through the espeak-ng CLI rather than piper's
                compiled bridge (verified to emit identical IPA), and lexicon
                paths resolved relative to the package, with .xz support.
  ids.py        BOS/PAD/phoneme-PAD.../EOS interleaving, reproduced from
                piper's phoneme_ids.py.
  synth.py      onnxruntime session + WAV writer.

Runtime dependencies drop to onnxruntime + numpy, with espeak-ng optional
(Sloleks covers 98.66% of real text; espeak handles the 1.34% remainder).
"""
import lzma
import os
import shutil
import subprocess
import sys

# No defaults that assume one machine's layout: every input is an explicit
# environment variable, and the script says what is missing rather than
# silently reading from somewhere that happens to exist.
SRC = os.environ.get("PIPER_PKG")        # the PATCHED piper package directory
MODEL_SRC = os.environ.get("MODEL_SRC")  # holds sl_SI-artur-sloleks-medium.onnx[.json]
LEX_SRC = os.environ.get("LEX_SRC")      # holds sl_pronunciation.tsv et al.
OUT = os.environ.get("BUNDLE_OUT", "./slovene-voice")
PKG = os.path.join(OUT, "slovene_voice")
_missing = [n for n, v in (("PIPER_PKG", SRC), ("MODEL_SRC", MODEL_SRC),
                           ("LEX_SRC", LEX_SRC)) if not v]
if _missing:
    raise SystemExit(
        "set %s. This script rebuilds the release bundle from a PATCHED piper "
        "install and is not needed to use the voice - download the bundle "
        "instead (scripts/fetch-models.sh tts)." % ", ".join(_missing))

for d in (PKG, os.path.join(OUT, "model"), os.path.join(OUT, "lexicon")):
    os.makedirs(d, exist_ok=True)

# --- 1. phonemize.py -------------------------------------------------------
src = open(os.path.join(SRC, "phonemize_slovene.py"), encoding="utf-8").read()

# Lexicon paths: package-relative, env-overridable, and .xz aware.
# The source file may already carry an earlier env-var patch, so match the whole
# assignment by regex rather than by literal text.
import re as _re
src, n = _re.subn(
    r"DEFAULT_LEXICON_PATH\s*=\s*Path\((?:[^()]|\([^()]*\))*\)",
    '_HERE = Path(__file__).resolve().parent.parent\n'
    'DEFAULT_LEXICON_PATH = Path(os.environ.get(\n'
    '    "SLOVENE_LEXICON", str(_HERE / "lexicon" / "sl_pronunciation.tsv.xz")))',
    src, count=1)
assert n == 1, "DEFAULT_LEXICON_PATH assignment not matched"
src, n = _re.subn(
    r'"SLOVENE_USER_LEXICON",\s*"[^"]*"',
    '"SLOVENE_USER_LEXICON", str(_HERE / "lexicon" / "user_lexicon.tsv")',
    src, count=1)
assert n == 1, "SLOVENE_USER_LEXICON default not matched"
src, n = _re.subn(
    r'"SLOVENE_ENGLISH_WORDS",\s*"[^"]*"',
    '"SLOVENE_ENGLISH_WORDS", str(_HERE / "lexicon" / "english_words.txt")',
    src, count=1)
# Copying english_words.txt into the bundle is not enough on its own: the
# source module defaults to an absolute path in the service tree, so a bundle
# unpacked anywhere else looks in the wrong place, loads an empty word list and
# disables embedded-English detection without saying anything.
assert n == 1, "SLOVENE_ENGLISH_WORDS default not matched"

# load_lexicon must transparently read .xz so the bundle can ship 12 MB not 81.
# ONLY load_lexicon (the first occurrence, 4-space indent). load_user_lexicon
# has the same line at 8 spaces inside a try: block - replacing there with a
# 4-space-indented insert empties the try and is a SyntaxError. The user lexicon
# is plain text anyway and needs no .xz support.
src, n = _re.subn(
    r'^    with open\(path, "r", encoding="utf-8"\) as f:',
    '    opener = lzma.open if str(path).endswith(".xz") else open\n'
    '    with opener(path, "rt", encoding="utf-8") as f:',
    src, count=1, flags=_re.M)
assert n == 1, "load_lexicon open() not matched"

# espeak: CLI instead of piper's compiled bridge. Verified byte-identical IPA on
# OOV words. This is what removes the last piper import.
# Drop the piper import and the bridge instantiation wherever they sit.
src, n = _re.subn(r"^\s*from piper\.phonemize_espeak import EspeakPhonemizer\s*\n",
                  "", src, count=1, flags=_re.M)
assert n == 1, "EspeakPhonemizer import not matched"
src, n = _re.subn(r"^\s*self\._espeak = EspeakPhonemizer\(\)\s*\n",
                  "", src, count=1, flags=_re.M)
assert n == 1, "EspeakPhonemizer instantiation not matched"
src = src.replace(
    '            raw = "".join(self._espeak.phonemize(ESPEAK_VOICE, word)[0])',
    '            raw = _espeak_cli(word)')

# Insert the CLI helper just above the class.
helper = '''

def _espeak_cli(word: str) -> str:
    """OOV fallback through the espeak-ng binary.

    piper reaches espeak through a compiled bridge (espeakbridge.so); this uses
    the CLI so the bundle needs no compiled extension. Confirmed to emit
    identical IPA to the bridge across OOV words including foreign names and
    loanwords. If espeak-ng is absent the word is dropped rather than raising -
    it is 1.34% of tokens, and losing one word beats losing the utterance.
    """
    try:
        out = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", ESPEAK_VOICE, word],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


'''
src = src.replace("\nclass SlovenePhonemizer:", helper + "class SlovenePhonemizer:")
src = src.replace("import re\n", "import re\nimport lzma\nimport subprocess\n", 1)
# Only IMPORTS matter. The module's prose legitimately mentions Piper when
# explaining where the 166-symbol table comes from; that should stay.
assert not _re.search(r"^\s*(from|import)\s+piper", src, _re.M), \
    "a piper import survived"
open(os.path.join(PKG, "phonemize.py"), "w", encoding="utf-8").write(src)

# --- 2. ids.py -------------------------------------------------------------
open(os.path.join(PKG, "ids.py"), "w", encoding="utf-8").write('''"""Phoneme strings -> model input ids.

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
''')

# --- 3. synth.py -----------------------------------------------------------
open(os.path.join(PKG, "synth.py"), "w", encoding="utf-8").write('''"""Slovene TTS with no piper dependency: text -> WAV via onnxruntime."""
import json
import os
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .ids import phonemes_to_ids
from .phonemize import SlovenePhonemizer

_HERE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.environ.get(
    "SLOVENE_VOICE_MODEL", str(_HERE / "model" / "sl_SI-artur-sloleks-medium.onnx"))


class SloveneVoice:
    def __init__(self, model_path=None, lexicon_path=None):
        model_path = str(model_path or DEFAULT_MODEL)
        self.config = json.load(open(model_path + ".json", encoding="utf-8"))
        # onnxruntime picks sensible thread counts; CPU is plenty at ~4x realtime.
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.id_map = self.config["phoneme_id_map"]
        self.sample_rate = self.config["audio"]["sample_rate"]
        inf = self.config.get("inference") or {}
        # Chosen by listening test, carried in the voice config.
        self.noise_scale = float(inf.get("noise_scale", 0.5))
        self.length_scale = float(inf.get("length_scale", 0.96))
        self.noise_w = float(inf.get("noise_w", 0.8))
        self.phonemizer = SlovenePhonemizer(lexicon_path)

    def synthesize(self, text, length_scale=None):
        """Returns int16 mono PCM at self.sample_rate."""
        scales = np.array(
            [self.noise_scale,
             self.length_scale if length_scale is None else length_scale,
             self.noise_w], dtype=np.float32)
        chunks = []
        for phonemes in self.phonemizer.phonemize(text):
            ids = phonemes_to_ids(phonemes, self.id_map)
            if len(ids) <= 3:  # BOS+PAD+EOS only: nothing to say
                continue
            audio = self.session.run(None, {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": scales,
            })[0].squeeze()
            audio = audio / max(np.abs(audio).max(), 1e-8) * 0.95
            chunks.append((audio * 32767).astype(np.int16))
            # Sentence gap, matching piper's own inter-sentence silence.
            chunks.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.int16))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def to_wav(self, text, path, length_scale=None):
        pcm = self.synthesize(text, length_scale)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return len(pcm) / self.sample_rate
''')

open(os.path.join(PKG, "__init__.py"), "w", encoding="utf-8").write(
    'from .synth import SloveneVoice\n\n__all__ = ["SloveneVoice"]\n')

open(os.path.join(PKG, "__main__.py"), "w", encoding="utf-8").write('''"""python -m slovene_voice "besedilo" out.wav"""
import sys

from .synth import SloveneVoice

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    dur = SloveneVoice().to_wav(text, out)
    print("%s  %.2fs" % (out, dur))
''')

# --- 4. assets -------------------------------------------------------------
for f in ("sl_SI-artur-sloleks-medium.onnx", "sl_SI-artur-sloleks-medium.onnx.json"):
    shutil.copy2(os.path.join(MODEL_SRC, f), os.path.join(OUT, "model", f))
shutil.copy2(os.path.join(LEX_SRC, "user_lexicon.tsv"),
             os.path.join(OUT, "lexicon", "user_lexicon.tsv"))
# The English word list is what makes embedded-English detection work. It was
# not copied here for the first bundles, because the English support was added
# to the service venv after this builder was written. A missing list is not an
# error at runtime - it loads as empty and every English word falls through to
# Slovene letter-to-sound - so the feature appeared to exist while never once
# running. Fail loudly here instead.
_ew = os.path.join(LEX_SRC, "english_words.txt")
if os.path.exists(_ew):
    shutil.copy2(_ew, os.path.join(OUT, "lexicon", "english_words.txt"))
else:
    raise SystemExit(
        "english_words.txt missing from %s - the bundle would ship with "
        "embedded-English detection silently disabled. Point LEX_SRC at a "
        "lexicon directory that has it, or set SLOVENE_ENGLISH=0 deliberately."
        % LEX_SRC)
with open(os.path.join(LEX_SRC, "sl_pronunciation.tsv"), "rb") as fi, \
        lzma.open(os.path.join(OUT, "lexicon", "sl_pronunciation.tsv.xz"), "wb",
                  preset=6) as fo:
    shutil.copyfileobj(fi, fo, 1 << 20)

# Copy the repo's declared minimums rather than restating them here. These
# were hardcoded, and had already drifted from tts/requirements.txt - the
# bundle shipped onnxruntime>=1.17 while the repo said >=1.31.1.
_req = os.environ.get("TTS_REQUIREMENTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tts", "requirements.txt"))
if os.path.exists(_req):
    shutil.copy2(_req, os.path.join(OUT, "requirements.txt"))
else:
    raise SystemExit("cannot find %s - the bundle would ship invented "
                     "dependency versions" % _req)
print("built:", OUT)
