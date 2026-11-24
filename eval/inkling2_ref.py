#!/usr/bin/env python
"""Inkling-2-Small (`model_type: inkling2`) and Inkling-2-Mini (`inkling2_loop`) text decoder in plain PyTorch, for the reference and for checking engines.

There is no Hugging Face implementation of this model (transformers 5.22 and main know only Inkling v1). The public
definitions of the function are the general engines' model code; the specification used here is SGLang's pull
request sgl-project/sglang#50186 at commit 237ba6baf2f2 (python/sglang/srt/models/
inkling2.py and inkling_common/inkling2_*.py, kernels/ops/inkling2/*), together with the checkpoint's
hf_quant_config.json. This file started from the plain-PyTorch transcription of that model in NovaSky-AI/SkyRL pull
request #2730 (skyrl/backends/skyrl_train/patches/megatron/inkling2/{reference,quant}.py at a7a8ba5e, Apache-2.0),
extended with a high-precision mode and batching over sequences of equal length.

Two precisions:

  fork   the serving engines' rounding points: norms, attention residuals, short convolutions, the attention's gate
         and sink and the MoE activation in FP32, rounded to BF16 where the engines store BF16; linears BF16 x BF16
         -> BF16 except router and unembedding (FP32 out); attention in FP32. This is a BF16 path of the same kind
         as the Hugging Face BF16 path of the other models.
  high   the reference: the discrete steps exactly as the model defines them and everything between them in
         float64 (`--dtype float32` for a cheaper check). Discrete steps: the stored weight codes and scales
         (NVFP4: E2M1 codes x E4M3 block scale per 16 inputs x per-expert amax / outer_scale_denominator; MXFP8: E4M3
         x UE8M0 per 32 inputs), the quantization of each expert GEMM's INPUT activations that the checkpoint
         requires (NVFP4 experts of layers 0-39: per-token amax, blocks of 16, E4M3 block scale chosen by the "four
         over six" lower-MAE rule, E2M1 values; MXFP8 experts of layers 40-47: blocks of 32, UE8M0 scale, E4M3
         values), and the top-8 expert choice. The quantizers take the float32 rounding of their high-precision
         input and use float32 for their scale arithmetic, as the engines' kernels do (a quantizer is a format
         definition); the products of the quantized operands, their per-token and per-expert outer scales and
         everything after are float64.

Inkling-2-Mini (`model_type: inkling2_loop`, config text_config.loop_config): the same layers, with layers
core_start..core_end-1 (6..17) applied num_passes (3) times with shared weights, 48 applications per token from 24
layers. The specification is the open pull request huggingface/transformers#51050 ("Add Inkling2 Mini") at head
e916f57dfd7e (models/inkling2_loop/modular_inkling2_loop.py): each pass has its own K/V and
short-convolution state (Config.applications, cache slots); passes after the first start again from the pre-loop
hidden and block states and mix four messages (query, key, value, MLP inputs) into each loop layer's inputs. That
pull request dequantizes the experts on load and does NOT quantize expert input activations, although the
checkpoint's hf_quant_config.json lists them (NVFP4 per-token for layers 0-19, MXFP8 for 20-23): for the looped model
`act_quant` is therefore off by default (Inkling2.act_quant; checked: float32 logits equal transformers#51050's float32
logits to rel. L2 1.4e-6 median, <job>, scripts/mini/ref_check.py). "fork" for the looped model is
a BF16 path of the same kind as the PR's BF16 model, not bit-identical to it.

Both: logits are 30 * tanh(z / 30) of the unembedding; only the first `unpadded_vocab_size` (200058) entries are
real tokens and only those are returned. The vision tower and mtp.safetensors are not used (text decoding).

    python eval/inkling2_ref.py --model-dir DIR --ids ids.npy --positions pos.npy --modes fork,high --out-dir OUT
    python eval/inkling2_ref.py --model-dir DIR --greedy-prompts prompts.json --total 2000 --ids ids.npy --out-dir OUT

The second form makes token sequences the way eval/make_reference.py does with the Hugging Face BF16 model: the
prompts (a JSON list of token-id lists of one length, eval/natural_prompts.py) and their greedy continuation in fork
precision (Decoder, all sequences in step), the EOS tokens of config.json / generation_config.json banned so that
no sequence ends. --deadline S stops early (the sequences are then shorter than --total; the log says so).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PREFIX = "model.llm."
ATTN_RES_EPS = 1e-6
L2_CLIP_EPS = 1e-6
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BF16 = torch.bfloat16
SPEC = {"sglang_pr": "sgl-project/sglang#50186", "sglang_commit": "237ba6baf2f2",
        "transcription": "NovaSky-AI/SkyRL#2730 @ a7a8ba5e (reference.py, quant.py), Apache-2.0"}
SPEC_LOOP = {"transformers_pr": "huggingface/transformers#51050", "transformers_commit": "e916f57dfd7e",
             "files": "src/transformers/models/inkling2_loop/modular_inkling2_loop.py, src/transformers/core_model_loading.py"}


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window_size: int
    local_layer_ids: list
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    log_scaling_n_floor: float
    log_scaling_alpha: float
    use_value_norm_no_gamma: bool
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    attn_res_normalize_values: bool
    n_routed_experts: int
    num_experts_per_tok: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    use_gate_bias: bool
    final_logit_softcapping: float
    loop: tuple | None = None  # Inkling-2-Mini (inkling2_loop): (core_start, core_end, num_passes), else None

    @classmethod
    def load(cls, model_dir) -> "Config":
        text = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        names = [k for k in cls.__dataclass_fields__ if k != "loop"]
        cfg = cls(**{k: text[k] for k in names})
        lc = text.get("loop_config")
        if lc:
            # a variable-loop checkpoint (is_varloop) has the same tensors (1,244, the same shapes); it was trained to run
            # at several pass counts, and runs at the num_passes its config states
            cfg.loop = (lc["core_start"], lc["core_end"], lc["num_passes"])
            assert cfg.loop[0] % cfg.n_layers_per_attn_res_block == 0 and cfg.loop[1] % cfg.n_layers_per_attn_res_block == 0
        # the swa_* fields equal the global ones in this checkpoint; the code below assumes it
        for k in ("head_dim", "num_attention_heads", "num_key_value_heads"):
            assert text.get("swa_" + k, getattr(cfg, k)) == getattr(cfg, k), k
        assert not text.get("pre_norm_pre_affine") and not text.get("final_norm_pre_affine")
        return cfg

    def is_local(self, i: int) -> bool:
        return i in self.local_layer_ids

    def applications(self) -> list:
        """The decoder-layer applications of one token, in order: (layer, pass, cache slot). Without a loop, layer i
        once with slot i. With loop (s, e, P): layers 0..s-1, then layers s..e-1 P times (pass 0 uses slots s..e-1,
        pass p >= 1 slots L + (p - 1)(e - s) + (i - s): every pass has its own K/V and convolution state, as
        transformers#51050's cache_offset), then layers e..L-1."""
        L = self.num_hidden_layers
        if not self.loop:
            return [(i, 0, i) for i in range(L)]
        s, e, P = self.loop
        out = [(i, 0, i) for i in range(s)]
        for p in range(P):
            out += [(i, p, i if p == 0 else L + (p - 1) * (e - s) + (i - s)) for i in range(s, e)]
        return out + [(i, 0, i) for i in range(e, L)]

    def cache_slots(self) -> int:
        return len(self.applications()) if not self.loop else self.num_hidden_layers + (self.loop[2] - 1) * (self.loop[1] - self.loop[0])


class Prec:
    """Rounding points. `sd`: dtype of tensors the engines store as BF16; `cd`: dtype of the engines' FP32 math."""

    def __init__(self, mode: str, dtype: torch.dtype = torch.float64):
        self.mode = mode
        self.fork = mode == "fork"
        self.sd = BF16 if self.fork else dtype
        self.cd = torch.float32 if self.fork else dtype


def rms_norm(x, weight, eps, p: Prec):
    u = x.to(p.cd)
    y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.to(p.cd)
    return y.to(p.sd)


def linear(x, w, p: Prec, fp32_out: bool = False):
    if not p.fork:
        return x.to(p.cd) @ w.to(p.cd).t()
    if fp32_out:
        return x.float() @ w.float().t()
    return F.linear(x.to(p.sd), w.to(p.sd))


# ---------------------------------------------------------------------------------------------- attention residuals

def block_attn_res(blocks, partial, queries, normalize_values, p: Prec):
    """Softmax over depth of the RMS-normalised sources; one mixture per row of `queries`. Returns `cd` tensors."""
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    addend = None
    if isinstance(partial, tuple):
        partial, addend = partial
    values = torch.stack([*blocks, partial]).to(p.cd)  # [n, ..., H]
    if addend is not None:
        values[-1] += addend.to(p.cd)
    hidden = values.shape[-1]
    inv_rms = torch.rsqrt(values.square().mean(-1) + ATTN_RES_EPS)  # [n, ...]
    q = queries.to(values.device).to(p.cd) * hidden ** -0.5  # [Q, H]
    logits = torch.einsum("n...d,qd->q...n", values, q) * inv_rms.movedim(0, -1).unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)  # [Q, ..., n]
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return tuple(torch.einsum("q...n,n...d->q...d", probs, values).unbind(0))


class AttnResState:
    def __init__(self, model: "Inkling2", h0):
        self.m, self.partial, self.blocks = model, h0, []

    def to(self, device):
        self.partial = self.partial.to(device)
        self.blocks = [b.to(device) for b in self.blocks]

    def boundary(self, i):
        return i % self.m.cfg.n_layers_per_attn_res_block == 0

    def attention_input(self, i, n):
        if not self.blocks:
            return (self.partial,) * n
        return block_attn_res(self.blocks, self.partial, self.m.w["attn_res.attn_queries.%d" % (i - 1)],
                              self.m.cfg.attn_res_normalize_values, self.m.p)

    def mlp_input(self, i, attn_out):
        if self.boundary(i):
            blocks, partial = [*self.blocks, self.partial], attn_out
        else:
            blocks, partial = self.blocks, (self.partial, attn_out)
        return block_attn_res(blocks, partial, self.m.w["attn_res.mlp_queries.%d" % i], self.m.cfg.attn_res_normalize_values, self.m.p)[0]

    def finalize(self, i, attn_out, mlp_out):
        p = self.m.p
        if self.boundary(i):
            self.blocks.append(self.partial)
            self.partial = attn_out + mlp_out  # BF16 add in the engines
        else:
            self.partial = (self.partial.to(p.cd) + attn_out.to(p.cd) + mlp_out.to(p.cd)).to(p.sd)

    def output(self):
        return block_attn_res(self.blocks, self.partial, self.m.w["attn_res.output_query"], self.m.cfg.attn_res_normalize_values, self.m.p)[0]


# ---------------------------------------------------------------------------------------------- loop messages (Mini)
# transformers#51050 (Inkling2LoopMessageAttentionResidual, Inkling2LoopMessageGate): in passes p >= 1 over the loop
# layers, every loop layer l gets four messages (query, key, value, MLP input) by attention over depth of the PREVIOUS
# pass's blocks past the pre-loop ones (the first block it appended is the pre-loop output, the same in every pass, and
# is skipped) and its final partial sum; each of the layer's four inputs x becomes (1 - tanh(F g)) x + tanh(G g) msg
# with g = D [unit_rms(x), unit_rms(msg)] (D rank 64, BF16 in the engines; F, G float32).

def message_gate(m: "Inkling2", i: int, which: str, x, msg):
    p, w, eps = m.p, m.layers[i], m.cfg.rms_norm_eps
    xf, mf = x.to(p.cd), msg.to(p.cd)
    unit = lambda v: v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)  # noqa: E731
    g = linear(torch.cat([unit(xf), unit(mf)], dim=-1), w[f"controllers.{which}.down_weight"], p).to(p.cd)
    forget = torch.tanh(g @ w[f"controllers.{which}.forget_weight"].to(p.cd).t())
    gain = torch.tanh(g @ w[f"controllers.{which}.gain_weight"].to(p.cd).t())
    return (1 - forget) * xf + gain * mf


def messages(m: "Inkling2", prev_blocks, prev_partial, loop_layer: int):
    """(query, key, value, mlp) messages of loop layer `loop_layer` from the previous pass (`cd` tensors)."""
    return block_attn_res(prev_blocks, prev_partial, m.w["loop_queries"][loop_layer], True, m.p)


# ---------------------------------------------------------------------------------------------- short convolutions
# Tensors are [R, T, D] (R sequences of T tokens); the shift never crosses from one sequence into another.

def _shift(x, lag):
    if lag == 0:
        return x
    return torch.cat([x.new_zeros(*x.shape[:-2], lag, x.shape[-1]), x[..., :-lag, :]], dim=-2)


def static_sconv(x, weight, p: Prec):
    """Depthwise causal conv, weight [D, 1, W] (tap W-1 = current token), identity residual, x (W+1)^-1/2."""
    w = weight.squeeze(1).to(p.cd)
    width = w.shape[1]
    xf = x.to(p.cd)
    acc = torch.zeros_like(xf)
    for tap in range(width):
        acc = acc + _shift(xf, width - 1 - tap) * w[:, tap]
    return ((acc + xf) * (width + 1) ** -0.5).to(p.sd)


def dynamic_sconv(x, static_w, weight_proj, source, p: Prec):
    """Global-layer K/V conv: per-token, per-KV-head taps static_w[lag, c] + (source @ weight_proj^T)[t, c//d, lag]
    (static_w [W, D], row 0 = current token), identity residual, x (2W+1)^-1/2."""
    width, dim = static_w.shape
    n_heads = weight_proj.shape[0] // width
    dyn = linear(source, weight_proj, p).view(*source.shape[:-1], n_heads, width)
    dyn = dyn.to(p.cd).repeat_interleave(dim // n_heads, dim=-2)  # [R, T, D, W]
    xf = x.to(p.cd)
    acc = torch.zeros_like(xf)
    for lag in range(width):
        acc = acc + _shift(xf, lag) * (static_w[lag].to(p.cd) + dyn[..., lag])
    return ((acc + xf) * (2 * width + 1) ** -0.5).to(p.sd)


# ---------------------------------------------------------------------------------------------- attention

def l2_clip(x, threshold, p: Prec, scale=None, output_scale=None):
    """x [..., heads, d]: scale per head, then clip each head vector's L2 norm to `threshold`."""
    v = x.to(p.cd)
    if scale is not None:
        v = v * scale.to(p.cd)
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    out = (v * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(p.sd)
    if output_scale is not None:
        out = (out.to(p.cd) * output_scale[..., None, None].to(p.cd)).to(p.sd)
    return out


def dense_attention(q, k, v, rel, window, p: Prec, chunk: int = 256):
    """One sequence. Causal GQA with the sheared relative bias rel[i, h, i - j] (0 outside [0, extent)) and an
    optional sliding window (keys with i - j < window). q [T, Hq, d], k/v [T, Hkv, d], rel [T, Hq, E].
    Returns out [T, Hq, d] and lse [T, Hq] in `cd`."""
    T, Hq, d = q.shape
    group = Hq // k.shape[1]
    extent = rel.shape[-1]
    kf = k.to(p.cd).repeat_interleave(group, dim=1).transpose(0, 1)
    vf = v.to(p.cd).repeat_interleave(group, dim=1).transpose(0, 1)
    j = torch.arange(T, device=q.device)
    outs, lses = [], []
    for start in range(0, T, chunk):
        i = torch.arange(start, min(start + chunk, T), device=q.device)
        qf = q[i].to(p.cd).transpose(0, 1)
        scores = (qf @ kf.transpose(1, 2)) * d ** -0.5
        dist = i[:, None] - j[None, :]
        in_band = (dist >= 0) & (dist < extent)
        idx = dist.clamp(0, extent - 1).unsqueeze(1).expand(-1, Hq, -1)
        bias = rel[i].to(p.cd).gather(-1, idx).transpose(0, 1)
        scores = scores + bias.masked_fill(~in_band.unsqueeze(0), 0.0)
        visible = dist >= 0
        if window is not None:
            visible &= dist < window
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        outs.append((torch.exp(scores - lse.unsqueeze(-1)) @ vf).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs), torch.cat(lses)


def attention(m: "Inkling2", i: int, xq, xk, xv, positions):
    cfg, p, w = m.cfg, m.p, m.layer(i)
    R, T, _ = xq.shape
    H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    local = cfg.is_local(i)
    q = linear(xq, w["attn.wq_du.weight"], p)
    r = linear(xq, w["attn.wr_du.weight"], p)
    gate = linear(xq, w["attn.gate_attn.weight"], p)
    sink = linear(xq, w["attn.attn_sink_proj.weight"], p)
    k = linear(xk, w["attn.wk_dv.weight"], p)
    v = linear(xv, w["attn.wv_dv.weight"], p)
    if local:
        k = static_sconv(k, w["attn.k_sconv.weight"], p)
        v = static_sconv(v, w["attn.v_sconv.weight"], p)
        tau = None
    else:
        k = dynamic_sconv(k, w["attn.k_sconv.static_w"], w["attn.k_sconv.weight_proj"], xk, p)
        v = dynamic_sconv(v, w["attn.v_sconv.static_w"], w["attn.v_sconv.weight_proj"], xv, p)
        pos = positions.to(p.cd)
        tau = (1.0 + cfg.log_scaling_alpha * torch.log(torch.clamp((pos + 1) / cfg.log_scaling_n_floor, min=1.0)))
        tau = tau.expand(R, T)
    q = l2_clip(q.view(R, T, H, D), cfg.qk_l2_clip_threshold, p, w["attn.q_l2_clip.scale"], tau)
    k = l2_clip(k.view(R, T, KV, D), cfg.qk_l2_clip_threshold, p)
    v = v.view(R, T, KV, D)
    if cfg.use_value_norm_no_gamma:
        v = rms_norm(v, None, cfg.rms_norm_eps, p)
    r = l2_clip(r.view(R, T, H, cfg.d_rel), cfg.rel_l2_clip_threshold, p, w["attn.rel_l2_clip.scale"], tau)
    rel = torch.einsum("rthd,de->rthe", r, w["attn.rel_logits_proj.proj"].to(p.sd))  # BF16 einsum in fork mode
    window = cfg.sliding_window_size if local else None
    outs = []
    for s in range(R):
        out, lse = dense_attention(q[s], k[s], v[s], rel[s], window, p)
        out = out * torch.sigmoid(gate[s].to(p.cd).view(T, H, D))
        out = out * torch.sigmoid(lse - sink[s].to(p.cd)).unsqueeze(-1)
        outs.append(out.reshape(T, H * D).to(p.sd))
    return linear(torch.stack(outs), w["attn.wo_ud.weight"], p)


# ---------------------------------------------------------------------------------------------- experts

def _e2m1_table(device):
    mag = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=device)
    return torch.cat([mag, -mag])


def quant_nvfp4(x):
    """NVFP4 activation quantization of the engines' kernels (per-row amax, 1x16 blocks, the lower-MAE of the /6 and
    /4 E4M3 block scales, E2M1 values with round-to-nearest-even). Input: float32. Returns (codes x block scale,
    exact in float32, [..., K]) and the per-row outer scale amax / 1536 (float32)."""
    x = x.float()
    fmax = torch.finfo(torch.float32).max
    amax = x.abs().amax(-1)
    encoder = (1536.0 / amax).clamp(max=fmax)
    decoder = 1.0 / encoder
    outer = amax * (1.0 / 1536.0)
    blocks = x.unflatten(-1, (-1, 16))
    block_max = blocks.abs().amax(-1)
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    boundaries = (grid[:-1] + grid[1:]) / 2
    best_vals = best_err = None
    for inverse_max in (1.0 / 6.0, 0.25):
        scale = (block_max * (encoder[..., None] * inverse_max)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        inverse = (1.0 / (scale * decoder[..., None])).clamp(max=fmax)
        values = blocks * inverse[..., None]
        magnitude = values.abs()
        code = torch.bucketize(magnitude, boundaries)
        ties = magnitude == boundaries[code.clamp(max=6)]
        code = code + (ties & (code % 2 == 1)).long()  # cvt.rn: ties to the even code
        rounded = grid[code] * torch.where(values.signbit(), -1.0, 1.0)
        vals = rounded * scale[..., None]  # exact: 3-bit x 4-bit significands
        err = (vals * outer[..., None, None] - blocks).abs().sum(-1)
        if best_vals is None:
            best_vals, best_err = vals, err
        else:
            choose = err < best_err  # a strictly better /4 wins, as the kernel
            best_vals = torch.where(choose[..., None], vals, best_vals)
    return best_vals.flatten(-2), outer


def quant_mxfp8(x):
    """MXFP8 activation quantization (blocks of 32, UE8M0 scale rounded up, E4M3 values). Input float32. Returns the
    dequantized values in float64 (exact)."""
    groups = x.float().unflatten(-1, (-1, 32))
    raw = groups.abs().amax(-1) * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0 ** -127, torch.zeros_like(exponent), exponent.clamp(max=254))
    scale = torch.exp2(exponent.double() - 127.0).clamp(min=2.0 ** -127)
    inverse = torch.where(exponent == 0, torch.full_like(scale, 2.0 ** 127), 1.0 / scale).float()
    values = (groups * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).double()
    return (values * scale[..., None]).flatten(-2)


def dequant_weights(stored: dict, which: str, es, dtype):
    """Experts `es` (1-D index tensor) of `which` (w13 [2I, H], gate/up rows interleaved; or w2 [H, I]) in `dtype`:
    [len(es), N, K]. Exact for float64; for float32 one rounding of (code x block scale) x (amax / denominator)."""
    wt = stored[which + "_weight"][es]
    if wt.dtype == torch.uint8:
        codes = torch.stack([wt & 0xF, wt >> 4], dim=-1).flatten(-2).long()
        vals = _e2m1_table(wt.device).to(dtype)[codes]
        del codes
        scale = stored[which + "_weight_scale"][es].to(torch.float32).to(dtype)
        vals = (vals.unflatten(-1, (-1, 16)) * scale.unsqueeze(-1)).flatten(-2)
        factor = stored[which + "_weight_amax"][es].double() / stored[which + "_weight_outer_scale_denominator"].double()
        if dtype == torch.float64:
            return vals * factor.view(-1, 1, 1)
        return (vals.double() * factor.view(-1, 1, 1)).to(dtype)
    if wt.dtype == torch.float8_e4m3fn:
        exponent = torch.exp2(stored[which + "_weight_scale"][es].double() - 127.0).unsqueeze(-1)
        return (wt.double().unflatten(-1, (-1, 32)) * exponent).flatten(-2).to(dtype)
    return wt.to(dtype)


def weight_format(stored: dict, which: str) -> str:
    return {torch.uint8: "nvfp4", torch.float8_e4m3fn: "mxfp8"}.get(stored[which + "_weight"].dtype, "bf16")


def expert_matmul(fmt: str, W, x, p: Prec, act_quant: bool = True):
    """x [N, K] (sd) times one expert's dequantized weight W [M, K] (float32 for fork precision, `cd` for high),
    with the input quantization the checkpoint prescribes (act_quant=False is a study switch only: the model is not
    defined without it)."""
    if not act_quant:
        fmt = "bf16"
    if p.fork:  # the engines: quantized operands in FP32, product rounded to BF16
        if fmt == "nvfp4":
            vals, outer = quant_nvfp4(x.float())
            xq = vals * outer[..., None]
        elif fmt == "mxfp8":
            xq = quant_mxfp8(x.float()).float()
        else:
            return linear(x, W.to(p.sd), p)
        return (xq @ W.float().t()).to(p.sd)
    if fmt == "nvfp4":
        vals, outer = quant_nvfp4(x.to(torch.float32))
        return (vals.to(p.cd) @ W.t()) * outer.to(p.cd)[..., None]
    if fmt == "mxfp8":
        return quant_mxfp8(x.to(torch.float32)).to(p.cd) @ W.t()
    return linear(x, W, p)


def moe(m: "Inkling2", i: int, x, record=None):
    cfg, p, w = m.cfg, m.p, m.layer(i)
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    logits = linear(x, w["mlp.gate.weight"], p, fp32_out=True).to(p.cd)
    selection = logits.sigmoid()
    if cfg.use_gate_bias:
        selection = selection + w["mlp.gate.bias"].to(p.cd)
    ids = selection.topk(cfg.num_experts_per_tok, dim=-1).indices
    if getattr(m, "force_routes", None) is not None:  # a study tool: another path's expert choices (same token order)
        ids = m.force_routes[i].reshape(ids.shape).to(ids.device)
    if record is not None:
        record[i] = ids.view(*shape[:-1], -1).cpu()
    chosen = logits.gather(-1, ids).sigmoid()
    weights = chosen / chosen.sum(-1, keepdim=True) * cfg.route_scale
    stored = m.experts[i]
    aq = getattr(m, "act_quant", True)
    wdt = torch.float32 if p.fork else p.cd
    fmt13, fmt2 = weight_format(stored, "w13"), weight_format(stored, "w2")
    out = torch.zeros(x.shape, dtype=p.cd, device=x.device)
    used = ids.unique()
    chunk = getattr(m, "expert_chunk", 8 if wdt == torch.float64 else 16)
    for c in range(0, len(used), chunk):
        es = used[c:c + chunk]
        W13, W2 = dequant_weights(stored, "w13", es, wdt), dequant_weights(stored, "w2", es, wdt)
        for j, e in enumerate(es.tolist()):
            token, slot = (ids == e).nonzero(as_tuple=True)
            h = expert_matmul(fmt13, W13[j], x[token], p, aq)
            g, u = h[:, 0::2].to(p.cd), h[:, 1::2].to(p.cd)
            act = cfg.mlp_gate_softcap * torch.tanh(g / cfg.mlp_gate_softcap) * torch.sigmoid(g)
            act = act * (cfg.mlp_up_softcap * torch.tanh(u / cfg.mlp_up_softcap))
            act = (act * weights[token, slot, None]).to(p.sd)
            out.index_add_(0, token, expert_matmul(fmt2, W2[j], act, p, aq).to(p.cd))
        del W13, W2
    return out.to(p.sd).view(shape)


# ---------------------------------------------------------------------------------------------- model

class Inkling2:
    """Weights stay in the checkpoint's own storage (BF16 / F32 / packed experts), spread over `devices` by layer;
    every computation upcasts what it needs. forward(ids [R, T]) -> final hidden [R, T, H]."""

    def __init__(self, model_dir, devices=None, mode="high", dtype=torch.float64):
        from safetensors import safe_open

        self.cfg = cfg = Config.load(model_dir)
        self.p = Prec(mode, dtype)
        devices = devices or [torch.device("cuda", d) for d in range(torch.cuda.device_count())]
        L = cfg.num_hidden_layers
        per = math.ceil(L / len(devices))
        self.layer_device = [devices[min(i // per, len(devices) - 1)] for i in range(L)]
        self.first, self.last = devices[0], devices[-1]
        self.w: dict = {}
        self.layers = [dict() for _ in range(L)]
        self.experts = [dict() for _ in range(L)]
        index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        by_file = defaultdict(list)
        for name, file in index.items():
            if name.startswith(PREFIX):
                by_file[file].append(name)
        for file, names in sorted(by_file.items()):
            with safe_open(str(Path(model_dir) / file), framework="pt") as f:
                for name in names:
                    key = name[len(PREFIX):]
                    if key.startswith("layers."):
                        i = int(key.split(".")[1])
                        rest = key.split(".", 2)[2]
                        t = f.get_tensor(name).to(self.layer_device[i])
                        if rest.startswith("mlp.experts."):
                            self.experts[i][rest[len("mlp.experts."):]] = t
                        else:
                            self.layers[i][rest] = t
                    else:
                        dev = self.last if key.startswith(("unembed.", "norm.")) else self.first
                        self.w[key] = f.get_tensor(name).to(dev)
        for i in range(L):
            assert "w13_weight" in self.experts[i] and "w2_weight" in self.experts[i], i
        if cfg.loop:  # [2n, H] (query, mlp) and (key, value) pairs per loop layer -> [n, 4, H]: query, key, value, mlp
            qm = self.w.pop("middle_cycle_cross_pass_queries").unflatten(0, (-1, 2))
            kv = self.w.pop("middle_cycle_cross_pass_kv_queries").unflatten(0, (-1, 2))
            assert qm.shape[0] == cfg.loop[1] - cfg.loop[0], qm.shape
            self.w["loop_queries"] = torch.stack([qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]], dim=1)
            for i in range(*cfg.loop[:2]):
                for which in ("attn", "k", "v", "mlp"):
                    c = {k: self.layers[i][f"controllers.{which}.{k}_weight"].shape for k in ("down", "forget", "gain")}
                    assert c["down"][1] == 2 * cfg.hidden_size and c["forget"] == c["gain"] == (cfg.hidden_size, c["down"][0]), (i, which, c)
        self.act_quant = not cfg.loop  # Mini: the transformers#51050 function (experts dequantized, inputs not quantized)

    def layer(self, i):
        return self.layers[i]

    def layer_inputs(self, i, state, msgs):
        """The attention's query, key and value inputs of layer i (after the norms; with the loop messages in passes
        p >= 1)."""
        cfg, p, w = self.cfg, self.p, self.layers[i]
        local = cfg.is_local(i)
        streams = state.attention_input(i, 1 if local else 3)
        if local:
            if msgs is None:
                xq = xk = xv = rms_norm(streams[0], w["attn_norm.weight"], cfg.rms_norm_eps, p)
            else:
                xq, xk, xv = (rms_norm(message_gate(self, i, which, streams[0], msg), w["attn_norm.weight"], cfg.rms_norm_eps, p)
                              for which, msg in (("attn", msgs[0]), ("k", msgs[1]), ("v", msgs[2])))
        else:
            if msgs is not None:
                streams = [message_gate(self, i, which, x, msg) for which, x, msg in zip(("attn", "k", "v"), streams, msgs[:3])]
            xq = rms_norm(streams[0], w["q_input_norm.weight"], cfg.rms_norm_eps, p)
            xk = rms_norm(streams[1], w["k_input_norm.weight"], cfg.rms_norm_eps, p)
            xv = rms_norm(streams[2], w["v_input_norm.weight"], cfg.rms_norm_eps, p)
        return xq, xk, xv

    def mlp_input(self, i, state, attn, msgs):
        x = state.mlp_input(i, attn)
        if msgs is not None:
            x = message_gate(self, i, "mlp", x, msgs[3])
        return rms_norm(x, self.layers[i]["mlp_norm.weight"], self.cfg.rms_norm_eps, self.p)

    def run(self, state, layer_fn):
        """Every application of a decoder layer to `state` in order: layer_fn(i, state, msgs, slot). The loop
        (transformers#51050 Inkling2LoopTextModel.forward): the pre-loop state is kept; each pass after the first starts
        again from it and gets messages from the previous pass's blocks (past the pre-loop ones and the first one the
        pass appended) and its final partial sum."""
        cfg = self.cfg
        if not cfg.loop:
            for i, _, slot in cfg.applications():
                layer_fn(i, state, None, slot)
            return
        s, e, P = cfg.loop
        L = cfg.num_hidden_layers
        for i in range(s):
            layer_fn(i, state, None, i)
        pre_blocks, pre_partial = list(state.blocks), state.partial
        for i in range(s, e):
            layer_fn(i, state, None, i)
        for pas in range(1, P):
            prev_blocks, prev_partial = state.blocks[len(pre_blocks) + 1:], state.partial
            state.blocks, state.partial = list(pre_blocks), pre_partial
            for l, i in enumerate(range(s, e)):
                state.to(self.layer_device[i])
                msgs = messages(self, [b.to(self.layer_device[i]) for b in prev_blocks], prev_partial.to(self.layer_device[i]), l)
                layer_fn(i, state, msgs, L + (pas - 1) * (e - s) + l)
        for i in range(e, L):
            layer_fn(i, state, None, i)

    @torch.no_grad()
    def hidden(self, ids, record=None, taps=None, features=None):
        """ids [R, T] -> final hidden [R, T, H]. `record` (dict): expert choices per application slot; `taps`
        (list): the partial sum after every application (cd), for a layer-by-layer comparison. `features` (dict
        {application index: None}): filled with the running attention-residual partial block at the START of those
        applications (0-based, loop passes counted; the draft head's target_layer_ids, eval/inkling2_mtp.py), [R, T, H]
        in the stored dtype; the key len(applications) gets the final-norm output."""
        cfg, p = self.cfg, self.p
        ids = ids.to(self.first)
        R, T = ids.shape
        positions = torch.arange(T, device=self.first)
        h0 = rms_norm(self.w["embed.weight"][ids], self.w["embed_norm.weight"], cfg.rms_norm_eps, p)
        state = AttnResState(self, h0)
        visit = [0]

        def layer(i, state, msgs, slot):
            dev, w = self.layer_device[i], self.layers[i]
            state.to(dev)
            if features is not None and visit[0] in features:
                features[visit[0]] = state.partial.to(p.sd).clone()
            visit[0] += 1
            xq, xk, xv = self.layer_inputs(i, state, msgs)
            attn = static_sconv(attention(self, i, xq, xk, xv, positions.to(dev)), w["attn_sconv.weight"], p)
            mlp_in = self.mlp_input(i, state, attn, msgs)
            rec = {} if record is not None else None
            mlp = static_sconv(moe(self, i, mlp_in, rec), w["mlp_sconv.weight"], p)
            if record is not None:
                record[slot] = rec[i]
            state.finalize(i, attn, mlp)
            if taps is not None:
                taps.append(state.partial.float().cpu())

        self.run(state, layer)
        state.to(self.last)
        out = rms_norm(state.output(), self.w["norm.weight"], cfg.rms_norm_eps, p)
        if features is not None and visit[0] in features:
            features[visit[0]] = out
        return out

    @torch.no_grad()
    def logits(self, hidden):
        """hidden [..., H] -> soft-capped logits over the real vocabulary [..., unpadded_vocab_size] (float32 for
        fork mode, `cd` for high)."""
        cfg, p = self.cfg, self.p
        z = linear(hidden.to(self.last), self.w["unembed.weight"][: cfg.unpadded_vocab_size], p, fp32_out=True)
        cap = cfg.final_logit_softcapping
        return cap * torch.tanh(z / cap)


class Decoder:
    """Token-by-token decoding of B sequences in step (all at the same position), with the state the model carries
    from one token to the next: per layer application slot (Config.applications: Mini's loop layers have one slot per
    pass) the K/V of earlier positions (after the short convolution, the L2 clip and
    the value norm), and the last kernel-1 inputs of the four short convolutions (K and V projections, attention
    output, MoE output). The attention residuals and everything else are per token. Same arithmetic as
    Inkling2.hidden() on the whole sequence (the starter engine is this in fork precision)."""

    def __init__(self, model: Inkling2, max_batch: int, max_context: int):
        self.m, self.max_batch, self.max_context = model, max_batch, max_context
        cfg, p = model.cfg, model.p
        lags = cfg.sconv_kernel_size - 1
        kv = cfg.num_key_value_heads * cfg.head_dim
        H = cfg.hidden_size
        self.cache = []
        slot_layer = {slot: i for i, _, slot in cfg.applications()}
        for slot in range(cfg.cache_slots()):
            dev = model.layer_device[slot_layer[slot]]
            z = lambda *shape: torch.zeros(*shape, dtype=p.sd, device=dev)  # noqa: E731
            self.cache.append({"k": z(max_batch, max_context, cfg.num_key_value_heads, cfg.head_dim), "v": z(max_batch, max_context, cfg.num_key_value_heads, cfg.head_dim),
                               "k_in": z(max_batch, lags, kv), "v_in": z(max_batch, lags, kv), "attn": z(max_batch, lags, H), "mlp": z(max_batch, lags, H)})
        self.pos = 0

    def reset(self):
        for c in self.cache:
            for t in c.values():
                t.zero_()
        self.pos = 0

    @staticmethod
    def _push(hist, x):
        """hist [B, lags, D] + x [B, 1, D] -> the sconv input window [B, lags + 1, D]; hist moves on by one."""
        full = torch.cat([hist, x], dim=1)
        hist.copy_(full[:, 1:])
        return full

    @torch.no_grad()
    def step(self, tokens):
        m, cfg, p = self.m, self.m.cfg, self.m.p
        B, t = tokens.shape[0], self.pos
        assert t < self.max_context, "context full"
        H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        h0 = rms_norm(m.w["embed.weight"][tokens.to(m.first)][:, None], m.w["embed_norm.weight"], cfg.rms_norm_eps, p)
        state = AttnResState(m, h0)

        def layer(i, state, msgs, slot):
            dev, w, c = m.layer_device[i], m.layers[i], self.cache[slot]
            state.to(dev)
            local = cfg.is_local(i)
            xq, xk, xv = m.layer_inputs(i, state, msgs)
            q = linear(xq, w["attn.wq_du.weight"], p)
            r = linear(xq, w["attn.wr_du.weight"], p)
            gate = linear(xq, w["attn.gate_attn.weight"], p)
            sink = linear(xq, w["attn.attn_sink_proj.weight"], p)
            kfull = self._push(c["k_in"][:B], linear(xk, w["attn.wk_dv.weight"], p).to(p.sd))
            vfull = self._push(c["v_in"][:B], linear(xv, w["attn.wv_dv.weight"], p).to(p.sd))
            if local:
                k = static_sconv(kfull, w["attn.k_sconv.weight"], p)[:, -1:]
                v = static_sconv(vfull, w["attn.v_sconv.weight"], p)[:, -1:]
                tau = None
            else:  # the dynamic taps come from the current token's input only: earlier rows of `source` are unused
                pad = lambda x: torch.cat([x.new_zeros(B, kfull.shape[1] - 1, x.shape[-1]), x], dim=1)  # noqa: E731
                k = dynamic_sconv(kfull, w["attn.k_sconv.static_w"], w["attn.k_sconv.weight_proj"], pad(xk), p)[:, -1:]
                v = dynamic_sconv(vfull, w["attn.v_sconv.static_w"], w["attn.v_sconv.weight_proj"], pad(xv), p)[:, -1:]
                pos = torch.tensor([float(t)], device=dev, dtype=p.cd)
                tau = (1.0 + cfg.log_scaling_alpha * torch.log(torch.clamp((pos + 1) / cfg.log_scaling_n_floor, min=1.0))).expand(B, 1)
            q = l2_clip(q.view(B, 1, H, D), cfg.qk_l2_clip_threshold, p, w["attn.q_l2_clip.scale"], tau)
            k = l2_clip(k.view(B, 1, KV, D), cfg.qk_l2_clip_threshold, p)
            v = v.view(B, 1, KV, D)
            if cfg.use_value_norm_no_gamma:
                v = rms_norm(v, None, cfg.rms_norm_eps, p)
            c["k"][:B, t] = k[:, 0]
            c["v"][:B, t] = v[:, 0]
            r = l2_clip(r.view(B, 1, H, cfg.d_rel), cfg.rel_l2_clip_threshold, p, w["attn.rel_l2_clip.scale"], tau)
            rel = torch.einsum("rthd,de->rthe", r, w["attn.rel_logits_proj.proj"].to(p.sd))  # [B, 1, H, E]
            window = cfg.sliding_window_size if local else None
            lo = max(0, t + 1 - window) if window else 0
            keys = c["k"][:B, lo:t + 1].to(p.cd).repeat_interleave(H // KV, dim=2)  # [B, L, H, D]
            vals = c["v"][:B, lo:t + 1].to(p.cd).repeat_interleave(H // KV, dim=2)
            scores = torch.einsum("bhd,blhd->bhl", q[:, 0].to(p.cd), keys) * D ** -0.5
            dist = t - torch.arange(lo, t + 1, device=dev)
            extent = rel.shape[-1]
            bias = rel[:, 0].to(p.cd)[..., dist.clamp(max=extent - 1)].masked_fill(~(dist < extent), 0.0)  # [B, H, L]
            scores = scores + bias
            lse = torch.logsumexp(scores, dim=-1)  # [B, H]
            out = torch.einsum("bhl,blhd->bhd", torch.exp(scores - lse[..., None]), vals)
            out = out * torch.sigmoid(gate[:, 0].to(p.cd).view(B, H, D))
            out = out * torch.sigmoid(lse - sink[:, 0].to(p.cd)).unsqueeze(-1)
            attn_raw = linear(out.reshape(B, 1, H * D).to(p.sd), w["attn.wo_ud.weight"], p).to(p.sd)
            attn = static_sconv(self._push(c["attn"][:B], attn_raw), w["attn_sconv.weight"], p)[:, -1:]
            mlp_in = m.mlp_input(i, state, attn, msgs)
            mlp_raw = moe(m, i, mlp_in).to(p.sd)
            mlp = static_sconv(self._push(c["mlp"][:B], mlp_raw), w["mlp_sconv.weight"], p)[:, -1:]
            state.finalize(i, attn, mlp)

        m.run(state, layer)
        state.to(m.last)
        hidden = rms_norm(state.output(), m.w["norm.weight"], cfg.rms_norm_eps, p)
        self.pos += 1
        return m.logits(hidden[:, 0])


def logits_from_hidden(hidden, unembed, cap: float, unpadded: int, dtype=torch.float64):
    """The unembedding of hidden states taken from an engine (BF16 [..., H]): soft-capped logits over the real
    vocabulary, computed in `dtype` (the engines compute the product in FP32)."""
    z = hidden.to(dtype) @ unembed[:unpadded].to(dtype).t()
    return cap * torch.tanh(z / cap)


def eos_tokens(model_dir) -> list[int]:
    """EOS ids of config.json (top level and text_config) and generation_config.json, as make_reference.py bans them."""
    found: set[int] = set()
    for name in ("config.json", "generation_config.json"):
        f = Path(model_dir) / name
        if not f.exists():
            continue
        conf = json.loads(f.read_text())
        for c in (conf, conf.get("text_config") or {}):
            e = c.get("eos_token_id")
            found |= set() if e is None else ({e} if isinstance(e, int) else set(e))
    return sorted(found)


@torch.no_grad()
def greedy(model: Inkling2, prompts: torch.Tensor, total: int, banned: list[int], deadline: float = 0.0) -> torch.Tensor:
    """prompts [R, P] -> [R, <= total]: the prompt fed one token per step, then greedy decoding, in fork precision."""
    model.p = Prec("fork")
    R, P = prompts.shape
    dec = Decoder(model, R, total)
    ids = prompts.clone().cpu()
    t0 = time.time()
    for t in range(total - 1):
        logits = dec.step(ids[:, t].to(model.first))
        if t + 1 < P:
            continue
        logits[:, banned] = -float("inf")
        ids = torch.cat([ids, logits.argmax(-1).cpu()[:, None]], dim=1)
        done = t + 1
        if done in (P + 15, P + 99) or done % 250 == 0:
            el = time.time() - t0
            print(f"MK_INK greedy {ids.shape[1]} of {total} tokens at {el:.0f}s ({el / done * 1e3:.0f} ms per step)", flush=True)
        if deadline and time.time() - t0 > deadline:
            print(f"MK_INK greedy stopped by the deadline ({deadline:.0f}s) at {ids.shape[1]} of {total} tokens", flush=True)
            break
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ids", required=True, help="int64 [R, T] .npy (written by --greedy-prompts)")
    ap.add_argument("--positions", help="int64 [P] .npy: positions whose logits are kept")
    ap.add_argument("--greedy-prompts", default="", help="JSON list of prompt token-id lists: make --ids instead (see above)")
    ap.add_argument("--sequences", type=int, default=8, help="--greedy-prompts: the first this many prompts")
    ap.add_argument("--total", type=int, default=2000, help="--greedy-prompts: sequence length")
    ap.add_argument("--deadline", type=float, default=0.0, help="--greedy-prompts: stop decoding after this many seconds (0: never)")
    ap.add_argument("--modes", default="high", help="comma list of fork, high: one weight load, one pass per mode")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64", help="precision of the high mode")
    ap.add_argument("--batch", type=int, default=8, help="sequences per forward")
    ap.add_argument("--out-dir", required=True, help="writes logits_<mode>.npy float32 [R, P, V] and routing_<mode>.npz")
    ap.add_argument("--engine-hidden", default="", help="optional float32 [R, T, H] .npy of an engine's final hidden "
                    "states: written as logits_engine.npy through the same unembedding (float64)")
    args = ap.parse_args()
    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = Inkling2(args.model_dir, mode="fork")
    print(f"MK_INK loaded in {time.time() - t0:.0f}s; memory per card GB "
          f"{[round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in range(torch.cuda.device_count())]}", flush=True)
    cfg = model.cfg
    if args.greedy_prompts:
        pool = json.loads(Path(args.greedy_prompts).read_text())[: args.sequences]
        prompts = torch.tensor(pool, dtype=torch.long)
        banned = eos_tokens(args.model_dir)
        print(f"MK_INK greedy: {prompts.shape[0]} prompts of {prompts.shape[1]} tokens to {args.total}; banned EOS {banned}", flush=True)
        ids = greedy(model, prompts, args.total, banned, args.deadline)
        np.save(args.ids, ids.numpy().astype(np.int64))
        print(f"MK_INK greedy done {tuple(ids.shape)} in {time.time() - t0:.0f}s", flush=True)
        return 0
    ids = torch.from_numpy(np.load(args.ids)).long()
    pos = torch.from_numpy(np.load(args.positions)).long()
    if args.engine_hidden:
        hid = torch.from_numpy(np.load(args.engine_hidden))
        un = model.w["unembed.weight"]
        z = torch.stack([logits_from_hidden(hid[r][pos].to(un.device), un, cfg.final_logit_softcapping,
                                            cfg.unpadded_vocab_size).float().cpu() for r in range(hid.shape[0])])
        np.save(out_dir / "logits_engine.npy", z.numpy())
        print(f"MK_INK engine logits {tuple(z.shape)}", flush=True)
    for mode in args.modes.split(","):
        model.p = Prec(mode, getattr(torch, args.dtype))
        out = np.zeros((ids.shape[0], len(pos), cfg.unpadded_vocab_size), dtype=np.float32)
        routes = {}
        t1 = time.time()
        for s in range(0, ids.shape[0], args.batch):
            record = {}
            h = model.hidden(ids[s:s + args.batch], record)
            out[s:s + args.batch] = model.logits(h[:, pos.to(h.device)]).float().cpu().numpy()
            for i, r in record.items():
                routes.setdefault(i, []).append(r.numpy().astype(np.int16))
            del h
            print(f"MK_INK mode={mode} sequences {s}..{min(s + args.batch, ids.shape[0]) - 1} at {time.time() - t1:.0f}s; peak GB "
                  f"{[round(torch.cuda.max_memory_allocated(d) / 1e9, 1) for d in range(torch.cuda.device_count())]}", flush=True)
        np.save(out_dir / f"logits_{mode}.npy", out)
        np.savez_compressed(out_dir / f"routing_{mode}.npz", **{f"layer{i}": np.concatenate(v) for i, v in routes.items()})
        print(f"MK_INK mode={mode} done {out.shape} in {time.time() - t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
