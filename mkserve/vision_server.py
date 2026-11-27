"""Inkling 2's vision encoder as a local service, for mkserve (MK_VISION_URL).

POST /embed {"image": base64 bytes (PNG, JPEG, WebP...), "max_side": optional int}
  -> {"rows": base64 float16 [N, 2304], "count": N, "grid": [1, rows, cols], "seconds": float}

The rows are what the language model takes at the image's N placeholder tokens: the vision tower's merged and
projected patch features (transformers' Inkling2VisionModel, the reference implementation), one per 40 x 40 px tile.
The image is first scaled down so its long side is at most max_side (default 1280, about 1,000 tokens at most);
a smaller image is left as it is.

Run with the environment that has transformers and torch:
    <python with transformers, torch, torchvision, pillow> mkserve/vision_server.py [--model-dir DIR] [--port 8791]
Listens on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

# checkpoint names (model.visual.*) -> transformers' Inkling2VisionModel names
RENAMES = [
    ("backbone.embeddings.position_embedding", "embeddings.position_embedding.weight"),
    ("backbone.embeddings.", "embeddings."),
    ("backbone.ln_pre.", "pre_layernorm."),
    ("backbone.layers.", "layers."),
    (".ln_1.", ".layernorm_before."),
    (".ln_2.", ".layernorm_after."),
    (".ls_1.gamma", ".lambda_1"),
    (".ls_2.gamma", ".lambda_2"),
    (".mlp.c_fc.", ".mlp.fc1."),
    (".mlp.c_proj.", ".mlp.fc2."),
    ("proj_in.", "merger.linear_1."),
    ("proj_out.", "merger.linear_2."),
]


def rename(key: str) -> str:
    key = key[len("model.visual."):]
    for old, new in RENAMES:
        key = key.replace(old, new)
    return key


def load(model_dir: Path, device: str, dtype: torch.dtype) -> tuple[Inkling2VisionModel, Inkling2ImageProcessor]:
    cfg = json.loads((model_dir / "config.json").read_text())["vision_config"]
    cfg.pop("model_type", None)
    cfg.setdefault("rope_parameters", {"rope_type": "axial", "rope_theta": 10000.0})
    config = Inkling2VisionConfig(**cfg)
    config._attn_implementation = "sdpa"
    model = Inkling2VisionModel(config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for k, f in index.items() if k.startswith("model.visual.")})
    state = {}
    for name in files:
        with safe_open(model_dir / name, framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[rename(key)] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise SystemExit(f"vision weights do not match: missing {missing[:8]}, unexpected {unexpected[:8]}")
    model.to(device=device, dtype=dtype).eval()
    return model, Inkling2ImageProcessor()


class Encoder:
    def __init__(self, model_dir: Path):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float32
        self.model, self.processor = load(model_dir, self.device, self.dtype)
        self.lock = threading.Lock()

    def embed(self, data: bytes, max_side: int) -> tuple[np.ndarray, list[int]]:
        image = Image.open(io.BytesIO(data))
        image.load()
        scale = max_side / max(image.size)
        if scale < 1:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.Resampling.LANCZOS)
        inputs = self.processor(images=[image], return_tensors="pt")
        with self.lock, torch.inference_mode():
            out = self.model(pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
                             image_grid_thw=inputs["image_grid_thw"])
            rows = out.pooler_output.float().cpu().numpy()
        return rows, inputs["image_grid_thw"][0].tolist()


ENCODER: Encoder | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def reply(self, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self.reply(200, {"status": "ok", "device": ENCODER.device})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            data = base64.b64decode(body["image"])
            t0 = time.time()
            rows, grid = ENCODER.embed(data, int(body.get("max_side") or 1280))
        except Exception as exc:  # noqa: BLE001  a bad image is the client's error, not the server's end
            return self.reply(400, {"error": f"{type(exc).__name__}: {exc}"})
        self.reply(200, {"rows": base64.b64encode(rows.astype("<f2").tobytes()).decode(), "count": int(rows.shape[0]),
                         "dim": int(rows.shape[1]), "grid": grid, "seconds": round(time.time() - t0, 3)})


def main() -> None:
    global ENCODER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass"))
    ap.add_argument("--port", type=int, default=8791)
    args = ap.parse_args()
    t0 = time.time()
    ENCODER = Encoder(Path(args.model_dir))
    print(f"vision encoder on {ENCODER.device} loaded in {time.time() - t0:.1f}s; http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
