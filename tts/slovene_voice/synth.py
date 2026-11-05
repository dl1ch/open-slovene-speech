"""Slovene TTS with no piper dependency: text -> WAV via onnxruntime."""
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
    "SLOVENE_VOICE_MODEL", str(_HERE / "model" / "sl_SI-artur-medium.onnx"))


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
