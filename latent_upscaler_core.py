from __future__ import annotations

import gc
import os
import re
from enum import Enum

import folder_paths
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


MODEL_FOLDER = "latent_upscale_models"
if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    )

VAE_DOWNSAMPLE = 16
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


class UpscaleMode(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXELS = "megapixels"


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _is_rocm_build():
    return getattr(torch.version, "hip", None) is not None


def _resolve_device(backend):
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not _is_rocm_build() or not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but this PyTorch build cannot access an AMD GPU.")
        return torch.device("cuda")
    if backend == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device backend: {backend}")


def _backend_label(device):
    if device.type == "cuda" and _is_rocm_build():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    return "CPU"


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        normed = self.norm(x)
        q = self.q(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        k = self.k(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        v = self.v(normed).reshape(b, c, t * h * w).transpose(1, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.squeeze(1).transpose(1, 2).reshape(b, c, t, h, w)
        return x + self.proj_out(attended)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        hidden = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(hidden.dtype)
        while emb_out.ndim < hidden.ndim:
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(hidden)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.in_blocks = self._make_blocks(
            in_blocks, channels, embed_dim, dropout, attn,
            temporal_every, temporal_kernel,
        )
        self.out_blocks = self._make_blocks(
            out_blocks, channels, embed_dim, dropout, attn,
            temporal_every, temporal_kernel,
        )
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _make_blocks(count, channels, embed_dim, dropout, attn,
                     temporal_every, temporal_kernel):
        blocks = nn.ModuleList()
        for index in range(count):
            if (index == 1 or index == count - 1) and attn:
                blocks.append(AttnBlock3D(channels))
            blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(TemporalConv(channels, temporal_kernel))
        return blocks

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(value * scale)) for value in x.shape[-3:])
        else:
            return x
        if size == x.shape[-3:]:
            return x
        b, c, t, _, _ = x.shape
        temporal_kernel = 0
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                temporal_kernel = block.dwconv.weight.shape[2]
                break
        overlap = temporal_kernel
        chunk = 32
        if not enable_chunking or t <= chunk:
            return self._forward_seg(x, scale, size)
        print(f"[MinimaxH3-3D] temporal chunking: T={t} chunks={(t + chunk - 1) // chunk} overlap={overlap}")
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        output = torch.zeros(b, c, t, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weights = torch.zeros(1, 1, t, 1, 1, device=x.device, dtype=x.dtype)
        start = 0
        while start < t:
            seg_start = start
            seg_end = min(t, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(t, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(t + 2 * overlap, out_end + overlap)
            segment = padded[:, :, lo:hi].contiguous()
            segment_out = self._forward_seg(segment, scale, (hi - lo, size[-2], size[-1]))
            crop_start = out_start + overlap - lo
            valid = segment_out[:, :, crop_start:crop_start + out_end - out_start]
            weight = torch.ones(out_end - out_start, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                length = seg_start - out_start
                weight[:length] = torch.arange(1, length + 1, device=x.device, dtype=x.dtype) / (length + 1)
            if out_end > seg_end:
                length = out_end - seg_end
                weight[-length:] = torch.arange(length, 0, -1, device=x.device, dtype=x.dtype) / (length + 1)
            weight_view = weight.view(1, 1, -1, 1, 1)
            output[:, :, out_start:out_end] += valid * weight_view
            weights[:, :, out_start:out_end] += weight_view
            start += chunk
            del segment, segment_out, valid
            if start % (chunk * 4) == 0:
                gc.collect()
        return output / weights.clamp_min(1e-8)

    def _forward_seg(self, x, scale, size):
        scale_embedding = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device,
        ).unsqueeze(0)
        embedding = self.embed(scale_embedding)
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, embedding.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, embedding.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


MODEL_CACHE = {}


def get_models_dir():
    return folder_paths.get_folder_paths(MODEL_FOLDER)[0]


def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(place models in: {get_models_dir()})"]


def _load_raw_sd(path):
    if path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as file:
                state = {key: file.get_tensor(key) for key in file.keys()}
        except ImportError:
            from safetensors.torch import load_file
            state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return {
        key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value
        for key, value in state.items()
    }


def _extract_upscaler_sd(state):
    if any(key.startswith("upscaler.") for key in state):
        return {
            key[len("upscaler."):]: value
            for key, value in state.items() if key.startswith("upscaler.")
        }
    return state


def _detect_arch(state):
    config = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12,
        "channels": 512, "dropout": 0.1, "attn": False,
        "temporal_every": 2, "temporal_kernel": 5,
    }
    if "conv_in.weight" in state:
        config["in_channels"] = state["conv_in.weight"].shape[1]
        config["channels"] = state["conv_in.weight"].shape[0]
    input_ids, output_ids = set(), set()
    temporal = False
    for key, value in state.items():
        match = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if match:
            input_ids.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if match:
            output_ids.add(int(match.group(1)))
        if key.endswith("dwconv.weight"):
            temporal = True
            config["temporal_kernel"] = value.shape[2]
    if input_ids:
        config["in_blocks"] = len(input_ids)
    if output_ids:
        config["out_blocks"] = len(output_ids)
    if not temporal:
        config["temporal_every"] = 0
    config["attn"] = False
    return config


def load_model(name, device, precision):
    if str(name).startswith("("):
        raise ValueError("Please place the upscale model in models/latent_upscale_models")
    backend = _backend_label(device)
    cache_key = f"{name}::{backend}::{precision}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key].to(device, non_blocking=True)
    try:
        path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, name)
    except Exception as exc:
        raise FileNotFoundError(f"Model file not found: {name}") from exc
    state = _extract_upscaler_sd(_load_raw_sd(path))
    config = _detect_arch(state)
    model = LatentResizer3D(**config)
    model.load_state_dict(state, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)
    MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D GH] Loaded upscale model: {name}")
    print(
        f"  Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Temporal: {'on' if config['temporal_every'] > 0 else 'off'} "
        f"(every={config['temporal_every']}, kernel={config['temporal_kernel']}) | "
        f"Backend: {backend} | Precision: {precision}"
    )
    return model


def clear_model_cache():
    for model in MODEL_CACHE.values():
        model.to("cpu")
    if mm is not None:
        mm.soft_empty_cache()
