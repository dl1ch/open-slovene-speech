"""Slovene phonemizer: Sloleks reference lexicon, espeak-ng only as fallback.

WHY THIS EXISTS
espeak-ng's Slovene is rule-based, but Slovene lexical stress is unpredictable
from spelling. Measured against Sloleks 3.1 on 4,000 word forms, espeak places
the stress on the wrong syllable in ~74% of them - e.g. "ahniti" comes out
ah-NI-ti instead of AH-ni-ti. Round-trip ASR cannot see this (Whisper still
recognises the word), but any Slovene speaker hears it immediately.

Sloleks ships a reference IPA transcription per word form, produced by the RSDO
G2P. This module looks the word up and only falls back to espeak for OOV
tokens. Measured token coverage on the ARTUR training text: 98.6%.

NOTATION
Sloleks is the canonical inventory here, because it supplies ~99% of tokens.
espeak's fallback output is rewritten into Sloleks notation (r->ɾ, h->x, tʃ->ʧ,
ts->ʦ, w->u̯, ...) so the model never sees two competing phoneme systems for
the same sound - that would be worse than either system alone.

Output matches the other Piper phonemizers: one list of single-codepoint
phonemes per sentence, NFD-normalised.
"""
from __future__ import annotations

import logging
import os
import re
import lzma
import subprocess
import subprocess
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union

_LOGGER = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH = Path(os.environ.get(
    "SLOVENE_LEXICON", str(_HERE / "lexicon" / "sl_pronunciation.tsv.xz")))
ESPEAK_VOICE = "sl"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_CLEAN = re.compile(r"[^\w']+", re.UNICODE)
_KEEP_PUNCT = set(',.!?;:"')

# espeak-ng -> Sloleks. Multi-character sequences are applied first, longest
# first, so "tʃ" becomes ʧ before the bare "t" and "ʃ" are touched.
_SEQ_MAP = [
    ("tʃ", "ʧ"),
    ("ts", "ʦ"),
    ("dz", "ʣ"),
    ("dʒ", "ʤ"),
]
_CHAR_MAP = {
    "r": "ɾ",   # espeak trill -> Sloleks tap
    "h": "x",   # both are /x/; Sloleks is consistent, espeak is not
    "ʑ": "ʒ",
    "w": "u̯",   # the l-vocalisation glide, written u̯ in Sloleks
    "ɲ": "nj",
    "ʎ": "lj",
    "ˌ": "",    # Sloleks marks primary stress only
}


# Piper maps phonemes to ids through a fixed 166-symbol table. Six symbols
# Sloleks uses are absent from it, and an unmapped phoneme is silently dropped -
# which would have mangled every word containing "č" (ʧ alone is ~1% of all
# phonemes in this corpus). Each is rewritten to symbols the table does have.
# The choice per sound is arbitrary but must be STABLE: the model only needs one
# consistent representation, not a phonetically ideal one.
_TO_SUPPORTED = [
    ("ʧ", "tʃ"),      # 0.95% of phonemes - Slovene č
    ("ʣ", "dz"),
    ("ʤ", "dʒ"),
    ("ⁿ", "n"),       # nasal release
    ("ˡ", "l"),       # lateral release
    ("/", ""),
]


def to_supported_inventory(ipa: str) -> str:
    """Rewrite symbols missing from Piper's phoneme_id_map."""
    for src, dst in _TO_SUPPORTED:
        ipa = ipa.replace(src, dst)
    return ipa


def espeak_to_sloleks(ipa: str) -> str:
    """Rewrite espeak-ng IPA into the Sloleks inventory."""
    out = unicodedata.normalize("NFC", ipa)
    for src, dst in _SEQ_MAP:
        out = out.replace(src, dst)
    return "".join(_CHAR_MAP.get(ch, ch) for ch in out)


def load_lexicon(path: Union[str, Path]) -> Dict[str, str]:
    """wordform -> IPA. The file is 'form <TAB> ipa <TAB> frequency'."""
    lex: Dict[str, str] = {}
    opener = lzma.open if str(path).endswith(".xz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                lex[parts[0]] = parts[1]
    _LOGGER.info("Loaded %d Slovene pronunciations from %s", len(lex), path)
    return lex


# ---------------------------------------------------------------------------
# TEXT NORMALISATION (RSDO-DS2-NORM, via a local service)
#
# "3. maja ob 14.30" must be spoken as "tretjega maja ob štirinajstih trideset",
# with correct case agreement. That is what RSDO-DS2-NORM does, and it depends
# on classla, which pins torch<=2.6 - incompatible with the torch==2.14.0+cu126
# this host needs for sm_70 kernels. Hence a separate venv behind an HTTP call
# rather than an import.
#
# It also matters that ARTUR's training text contains NO digits at all, so the
# model has never seen one. Normalising at inference puts the input back into
# the distribution the voice was actually trained on.
#
# Failure is always non-fatal: if the service is down or slow, the original text
# is used. A voice that mispronounces a date beats a voice that does not speak.
NORMALIZER_URL = os.environ.get("SLOVENE_NORMALIZER_URL", "http://127.0.0.1:8081/normalize")
NORMALIZE = os.environ.get("SLOVENE_NORMALIZE", "1") != "0"
_NORM_CACHE: Dict[str, str] = {}
# BACKOFF, NOT A LATCH. This used to be a permanent _NORM_DEAD flag: one failure
# and number expansion was off for the life of the process, silently. That fired
# in the real world - slovene-text takes ~30 s to load its classla models, so a
# TTS restart races it, the first request times out, and every later request
# then skipped normalisation with no further log line. Dates and decimals simply
# vanished from the audio (digits have no phonemes), and nothing said why.
# Now a failure only suppresses retries for a short while, so the service heals
# itself once the normaliser finishes starting.
_NORM_RETRY_AFTER = [0.0]   # unix time before which we do not retry
_NORM_BACKOFF = 30.0        # seconds


def normalize_via_service(text: str, timeout: float = 15.0) -> str:
    # 15 s, not 5: the normaliser answers in ~0.13 s once warm, but its first
    # call after a restart has to finish loading classla. 5 s was under that.
    import time as _t
    if not NORMALIZE or not text.strip():
        return text
    if _t.monotonic() < _NORM_RETRY_AFTER[0]:
        return text
    if text in _NORM_CACHE:
        return _NORM_CACHE[text]
    try:
        import json as _json
        import urllib.request as _u
        req = _u.Request(NORMALIZER_URL,
                         data=_json.dumps({"text": text}).encode("utf-8"),
                         headers={"Content-Type": "application/json"})
        with _u.urlopen(req, timeout=timeout) as r:
            out = _json.loads(r.read().decode("utf-8")).get("text") or text
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("normalizer unavailable (%s); using raw text, "
                        "retrying in %.0fs", exc, _NORM_BACKOFF)
        _NORM_RETRY_AFTER[0] = _t.monotonic() + _NORM_BACKOFF
        return text
    _NORM_CACHE[text] = out
    return out


# ---------------------------------------------------------------------------
# VOWEL HIATUS ACROSS WORD BOUNDARIES
#
# "za blaženje energetske" phonemises as "...njɛ ɛnɛɾ...": the first word ends in
# ɛ and the next begins with ɛ, so the model articulates BOTH and it is heard as
# "blaženjee". A native listener flagged it as the single most irritating artefact
# in the voice.
#
# (An earlier attempt to justify leaving it alone measured seconds-per-phoneme
# over whole utterances and found no effect - but that metric is far too coarse to
# see one locally stretched vowel, so it proved nothing.)
#
# Fix: when a word-final vowel is IDENTICAL to the next word's initial vowel,
# drop one. That is elision, which is what speakers do with a repeated vowel.
# Only exact repeats are touched; differing vowels in hiatus are left alone.
ELIDE_HIATUS = os.environ.get("SLOVENE_ELIDE_HIATUS", "1") != "0"
_VOWELS = set("aeiouɛɔəɪ")
_STRESS = "ˈˌ"


def elide_hiatus(ipa: str) -> str:
    """Collapse a repeated vowel across a word boundary: 'njɛ ɛnɛ' -> 'njɛ nɛ'."""
    if not ELIDE_HIATUS or " " not in ipa:
        return ipa
    words = ipa.split(" ")
    out = [words[0]]
    for w in words[1:]:
        prev = out[-1]
        # last phone of the previous word, ignoring length/punctuation
        tail = prev.rstrip("ː" + _STRESS)
        # U+032F (combining inverted breve) marks a GLIDE: u̯ is the l-vocalisation
        # /w/, not a vowel. "napovedau̯ ukrepe" must NOT elide to "napovedau̯ krepe".
        if tail.endswith("\u032f"):
            out.append(w)
            continue
        last = ""
        for ch in reversed(tail):
            if ch.isalpha() or ch in _VOWELS:
                last = ch
                break
        # first phone of this word, skipping stress marks
        j = 0
        while j < len(w) and w[j] in _STRESS:
            j += 1
        first = w[j] if j < len(w) else ""
        if j + 1 < len(w) and w[j + 1] == "\u032f":
            first = ""            # word begins with a glide, not a vowel
        if last and last == first and last in _VOWELS:
            stressed_second = j > 0                 # the 2nd word starts with ˈ or ˌ
            long_second = (j + 1 < len(w) and w[j + 1] == "ː")
            if stressed_second or long_second:
                # The second word's vowel carries the stress (or is long), so it is
                # the one that must survive - dropping it would strand the stress
                # mark on a consonant, or leave a dangling length mark. Drop the
                # first word's final (unstressed) vowel instead.
                k = len(prev) - 1
                while k >= 0 and prev[k] not in _VOWELS:
                    k -= 1
                if k >= 0:
                    out[-1] = prev[:k] + prev[k + 1:]
            else:
                # drop the repeated vowel from the second word, keeping its stress
                w = w[:j] + w[j + 1:]
        out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------------------
# USER LEXICON - override or extend Sloleks without retraining.
#
# Sloleks holds STANDARD Slovene. Colloquial text ("vidm", "nosjo", "reku")
# misses it entirely - 12.7% OOV on song lyrics versus 1.3% on prose - and those
# words fall through to espeak, which mis-stresses ~74% of what it sees.
#
# File: lexicon/user_lexicon.tsv beside the voice (SLOVENE_USER_LEXICON
# overrides the path), one entry per line:
#
#     surface <TAB> replacement
#
# The replacement is read one of two ways:
#   * looks like IPA (contains ˈ ː ɛ ɔ ə ʋ ɾ ...)  -> used as the pronunciation
#   * otherwise                                    -> treated as a STANDARD
#     spelling and looked up in Sloleks, so "vidm -> vidim" inherits the
#     reference pronunciation and its correct stress for free.
#
# Reloaded per process; lines starting with # are ignored.
USER_LEXICON_PATH = os.environ.get(
    "SLOVENE_USER_LEXICON", str(_HERE / "lexicon" / "user_lexicon.tsv"))
_IPA_MARKERS = set("ˈˌːɛɔəʋɾʃʒʧʦɪŋɱʲu\u032f")


def load_user_lexicon(path, sloleks):
    """surface -> IPA, resolving standard-spelling aliases through Sloleks."""
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = [p for p in line.split("\t") if p != ""]
                if len(parts) < 2:
                    continue
                surface, repl = parts[0].strip().lower(), parts[1].strip()
                if not surface or not repl:
                    continue
                if any(ch in _IPA_MARKERS for ch in repl):
                    out[surface] = repl                      # direct pronunciation
                else:
                    ipa = sloleks.get(repl.lower())           # standard-form alias
                    if ipa:
                        out[surface] = ipa
    except FileNotFoundError:
        return out
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("user lexicon unreadable (%s)", exc)
        return out
    _LOGGER.info("Loaded %d user lexicon entries from %s", len(out), path)
    return out




# ---------------------------------------------------------------------------
# EMBEDDED ENGLISH
#
# Slovene technical and business speech is full of English words. Sent through
# espeak's SLOVENE voice they get Slovene letter-to-sound rules and come out
# wrong ("blockchain" -> "bloc s hajn"). Sent through espeak's ENGLISH voice
# they come out in phonemes this voice has never been trained on. Neither is
# usable on its own, so: phonemise with en-us, then fold the result into the
# 46-symbol inventory the model actually knows.
#
# The mapping is deliberately the substitution a Slovene speaker makes:
#   θ -> t   "think" -> tink        æ -> ɛ   "bag" -> beg
#   ð -> d   "this"  -> dis         ʌ -> a   "cup" -> kap
#   ɹ -> ɾ   (Slovene tap)          ɚ -> əɾ  "better" -> beter
# ---------------------------------------------------------------------------
ENGLISH_ENABLED = os.environ.get("SLOVENE_ENGLISH", "1") != "0"
# Package-relative, like DEFAULT_LEXICON_PATH above. This was an absolute path
# into the reference host's service tree, because the English support was first
# written as a patch to a venv rather than to the portable package. Anywhere
# else the file was simply missing, the word list came back empty, and every
# English word quietly fell through to Slovene letter-to-sound - the feature
# silently absent rather than broken.
ENGLISH_WORDS_PATH = os.environ.get(
    "SLOVENE_ENGLISH_WORDS", str(_HERE / "lexicon" / "english_words.txt"))
_ENGLISH_VOICE = "en-us"
# NOT _VOWELS: that name is already taken at module level by the IPA vowel
# set used for hiatus elision, set("aeiouɛɔəɪ"). Redefining it here
# silently dropped ɛ ɔ ə ɪ from elision and broke 8 of 300 Slovene
# utterances, with no error anywhere.
_ENGLISH_VOWEL_LETTERS = set("aeiouy")

# Longest-first: diphthongs must be rewritten before their components, or the
# ʊ in "oʊ" is consumed by the bare-ʊ rule and the glide is lost.
ENGLISH_MULTI = [
    ("aɪ", "ai̯"),
    ("eɪ", "ɛi̯"),
    ("ɔɪ", "ɔi̯"),
    ("aʊ", "au̯"),
    ("oʊ", "ou̯"),
    ("ɚ", "əɾ"),   # r-coloured schwa: "better" -> beter
    ("ɜː", "ɛɾ"),  # "bird" -> berd; espeak drops the r, Slovene restores it
    ("ɜ", "ɛɾ"),
]

ENGLISH_SINGLE = {
    "ɹ": "ɾ",   # English approximant -> Slovene tap
    "r": "ɾ",
    "æ": "ɛ",   # "bag" -> beg
    "ʌ": "a",   # "cup" -> kap
    "ɑ": "a",
    "ɐ": "a",
    "ʊ": "u",
    "ᵻ": "ɪ",
    "θ": "t",   # "think" -> tink
    "ð": "d",   # "this"  -> dis
    "v": "ʋ",   # Slovene has ʋ, not v
    "h": "x",   # Slovene h is /x/
    "ˌ": "",    # secondary stress: never trained, and Slovene has one stress
    "ʔ": "",    # glottal stop: not a Slovene phoneme
    "̩": "",     # syllabic mark: not trained
}


def english_to_sloleks(ipa: str) -> str:
    """Fold espeak en-us IPA into the symbols this voice was trained on."""
    for src, dst in ENGLISH_MULTI:
        ipa = ipa.replace(src, dst)
    return "".join(ENGLISH_SINGLE.get(ch, ch) for ch in ipa)


def load_english_words(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            words = {w.strip().lower() for w in f if w.strip()}
    except OSError as exc:
        _LOGGER.warning("English wordlist unreadable (%s); embedded English "
                        "will fall back to Slovene rules", exc)
        return frozenset()
    _LOGGER.info("Loaded %d English words from %s", len(words), path)
    return frozenset(words)


def looks_english(word: str, english: frozenset) -> bool:
    """See the module docstring: length and vowel guards are load-bearing."""
    if not ENGLISH_ENABLED or len(word) < 3:
        return False
    w = word.lower()
    if not (_ENGLISH_VOWEL_LETTERS & set(w)):   # vowel-less -> acronym, spell it out instead
        return False
    return w in english



def _espeak_voice(word: str, voice: str) -> str:
    """espeak-ng IPA for one word in a given voice, empty string on failure."""
    try:
        out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, word],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()

def _espeak_cli_sl(word: str) -> str:
    return _espeak_voice(word, ESPEAK_VOICE)



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


class SlovenePhonemizer:
    def __init__(self, lexicon_path: Optional[Union[str, Path]] = None) -> None:
        self.lexicon = load_lexicon(lexicon_path or DEFAULT_LEXICON_PATH)
        # user entries win over Sloleks so a bad reference form can be corrected
        self.user_lexicon = load_user_lexicon(USER_LEXICON_PATH, self.lexicon)
        self.lexicon.update(self.user_lexicon)
        self.english = load_english_words(ENGLISH_WORDS_PATH)
        self.english_hits = 0
        self.hits = 0
        self.misses = 0

    # -- one word ------------------------------------------------------------
    def phonemize_word(self, word: str) -> str:
        ipa = self.lexicon.get(word.lower())
        if ipa is not None:
            self.hits += 1
            return ipa
        self.misses += 1
        # Sloleks did not have it. Before falling back to Slovene letter-to-sound
        # rules, check whether this is an English word - those are common in
        # Slovene technical speech and Slovene rules mangle them.
        if looks_english(word, self.english):
            self.english_hits += 1
            raw = _espeak_voice(word, _ENGLISH_VOICE)
            if raw:
                return english_to_sloleks(raw)
        try:
            raw = _espeak_cli_sl(word)
        except Exception:  # noqa: BLE001 - never let one token kill a run
            return ""
        return espeak_to_sloleks(raw)

    # -- one sentence --------------------------------------------------------
    def phonemize_sentence(self, sentence: str) -> str:
        pieces: List[str] = []
        for token in sentence.split():
            word = _WORD_CLEAN.sub("", token)
            punct = "".join(c for c in token if c in _KEEP_PUNCT)
            if not word:
                if punct and pieces:
                    pieces[-1] += punct
                continue
            # Anything with a digit goes to espeak whole, so its number
            # expansion still applies (ARTUR's training text has no digits, so
            # this path only matters at inference time).
            if any(ch.isdigit() for ch in word):
                try:
                    raw = _espeak_cli_sl(word)
                    pieces.append(espeak_to_sloleks(raw) + punct)
                except Exception:  # noqa: BLE001
                    pass
                continue
            pieces.append(self.phonemize_word(word) + punct)
        return " ".join(p for p in pieces if p)

    # -- public API ----------------------------------------------------------
    def phonemize(self, text: str) -> List[List[str]]:
        text = normalize_via_service(text)
        result: List[List[str]] = []
        for sentence in _SENTENCE_SPLIT.split(text.strip()):
            if not sentence.strip():
                continue
            ipa = to_supported_inventory(elide_hiatus(self.phonemize_sentence(sentence)))
            if ipa:
                result.append(list(unicodedata.normalize("NFD", ipa)))
        return result

    @property
    def coverage(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
