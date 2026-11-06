"""Convert spoken-form Slovene back to written form, as an ASR post-step.

Our recogniser emits numbers, dates and measures as words, because that is how
ARTUR transcribes them: *dva tisoč dva* rather than *2002*. For subtitles, notes
and anything a person reads, the written form is wanted. Measured on FLEURS this
takes CER from 7.12% to 4.55% and WER from 14.53% to 12.72% — below stock
Whisper on both.

The work is done by RSDO-DS2-DENORM (CLARIN.SI, Apache 2.0, a separate rule-based
tool), which this module wraps. It is optional: if the tool is not installed,
`denormalize` returns its input unchanged, so the recogniser still works.

The wrapped package opens its data files relative to the working directory and
is not importable as an ordinary module, so this loads it once, from its own
directory, and keeps the process CWD unchanged.
"""
import logging
import os
import threading

_LOGGER = logging.getLogger(__name__)
_LOCK = threading.Lock()
_FN = None          # the real denormalize callable, or False once we give up

# Where the RSDO package lives. fetch-models.sh unpacks it beside the models;
# the container sets this explicitly.
DIR = os.environ.get(
    "SLOVENE_DENORMALIZER_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "slovene-denormalizer"))
ENABLED = os.environ.get("SLOVENE_DENORMALIZE", "1") != "0"


def _load():
    """Import the RSDO denormaliser once. Returns its callable or False."""
    global _FN
    if _FN is not None:
        return _FN
    with _LOCK:
        if _FN is not None:
            return _FN
        if not ENABLED or not os.path.isdir(DIR):
            if ENABLED and not os.path.isdir(DIR):
                _LOGGER.info("denormaliser not found at %s; numbers stay as "
                             "words (set SLOVENE_DENORMALIZE=0 to silence)", DIR)
            _FN = False
            return _FN
        import sys
        cwd = os.getcwd()
        try:
            # the package reads data/*.pickle relative to cwd, so run from there
            os.chdir(DIR)
            if DIR not in sys.path:
                sys.path.insert(0, DIR)
            from denormalizer import denormalize as _dn
            _FN = _dn
            _LOGGER.info("loaded RSDO denormaliser from %s", DIR)
        except Exception as e:           # a missing dep or a bad tree: degrade
            _LOGGER.warning("denormaliser failed to load (%s); numbers stay as "
                            "words", e)
            _FN = False
        finally:
            os.chdir(cwd)
    return _FN


def denormalize(text):
    """Spoken-form Slovene -> written form. Returns `text` unchanged on any error."""
    if not text or not text.strip():
        return text
    fn = _load()
    if not fn:
        return text
    cwd = os.getcwd()
    try:
        os.chdir(DIR)                    # data files are resolved per call too
        out = fn(text)
        if isinstance(out, dict):
            return out.get("denormalized_string", text) or text
        return str(out) if out else text
    except Exception as e:
        _LOGGER.warning("denormalize failed on one input (%s); left unchanged", e)
        return text
    finally:
        os.chdir(cwd)
