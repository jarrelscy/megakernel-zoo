"""An audio projector: speech encoder frames -> Inkling input embeddings, so Inkling transcribes from the audio itself
(Engine.submit(..., embeds=...)). The projector is a small MLP trained on one speech encoder's output; its weights and
prompt are in a folder named by MK_PROJECTOR (not in this repository):
  config.json         {"encoder_dim": 1024, "stack": 2, "layers": 3, "instruction": "...", "audio_prefix": "Audio:",
                       "audio_suffix": "\\nTranscript:\\n", "tokens_per_second": 6, "frames_per_second": 12.5}
  model.safetensors   net.0/net.2/net.4 .weight and .bias (Linear layers, GELU between them) and scale
It stacks `stack` consecutive frames into one vector, runs the MLP, multiplies by the learned scale, and rounds to
BF16 (the trained model received BF16 embeddings). The rows go into the prompt as they are: Inkling's own embedding
norm is not applied to them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np


class Projector:
    def __init__(self, folder: str):
        d = Path(folder)
        self.cfg = json.loads((d / "config.json").read_text())
        w = mx.load(str(d / "model.safetensors"))
        self.stack, self.dim = int(self.cfg["stack"]), int(self.cfg["encoder_dim"])
        self.layers = [(w[f"net.{i}.weight"].astype(mx.float32), w[f"net.{i}.bias"].astype(mx.float32)) for i in (0, 2, 4)]
        self.scale = float(w["scale"].item())
        assert self.layers[0][0].shape[1] == self.dim * self.stack and self.layers[-1][0].shape[0] == 2304
        mx.eval([a for pair in self.layers for a in pair])

    def rows(self, frames: np.ndarray) -> np.ndarray:
        """frames [T, encoder_dim] -> embeddings float32 [ceil(T / stack), 2304] (BF16 values)."""
        x = mx.array(np.asarray(frames, dtype=np.float32))
        pad = (-x.shape[0]) % self.stack
        if pad:
            x = mx.concatenate([x, mx.zeros((pad, x.shape[1]), dtype=mx.float32)])
        x = x.reshape(-1, self.dim * self.stack)
        for i, (wt, b) in enumerate(self.layers):
            x = x @ wt.T + b
            if i < len(self.layers) - 1:
                x = 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))     # nn.GELU(): the exact form
        x = (x * self.scale).astype(mx.bfloat16).astype(mx.float32)
        return np.array(x)
