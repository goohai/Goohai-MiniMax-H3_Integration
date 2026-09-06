from __future__ import annotations

import gc
import math

import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
    VAE_DOWNSAMPLE,
    UpscaleMode,
    _make_norm_tensors,
    _resolve_device,
    scan_models,
    load_model,
)


def _parts(samples):
    if not getattr(samples, "is_nested", False):
        raise ValueError("Minimax H3 GH Upscaler requires AV latent (video + audio)")
    p = tuple(samples.unbind())
    if len(p) != 2 or p[0].ndim != 5 or p[1].ndim != 4:
        raise ValueError("Expected NestedTensor(video[5D], audio[4D])")
    return p[0], p[1]


def _starts(size, tile, overlap):
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    out = list(range(0, max(1, size - tile + 1), step))
    last = size - tile
    if out[-1] != last:
        out.append(last)
    return sorted(set(out))


def _window(n, left, right, device):
    w = torch.ones(n, device=device, dtype=torch.float32)
    if left:
        w[:left] = torch.linspace(0.0, 1.0, left + 1, device=device)[1:]
    if right:
        w[-right:] = torch.linspace(1.0, 0.0, right + 1, device=device)[1:]
    return w


class GoohaiMinimaxH3LatentUpscaler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GoohaiMinimaxH3LatentUpscalerGH",
            display_name="Minimax H3 Latent Upscaler (GH)",
            category="Goohai/MiniMax H3 Integration/Upscaling",
            inputs=[
                io.Latent.Input("av_latent"),
                io.Combo.Input("model_name", options=scan_models()),
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option(UpscaleMode.SCALE_BY, [
                        io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                    ]),
                    io.DynamicCombo.Option(UpscaleMode.TARGET_DIMENSIONS, [
                        io.Int.Input("width", default=1280, min=64, max=8192, step=8),
                        io.Int.Input("height", default=704, min=64, max=8192, step=8),
                    ]),
                    io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                        io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1),
                    ]),
                ]),
                io.Int.Input("align", default=32, min=32, max=512, step=32),
                io.Boolean.Input("force_unload", default=True),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16"),
            ],
            outputs=[io.Latent.Output("av_latent")],
        )

    @classmethod
    def execute(cls, av_latent, model_name, mode, align, force_unload, device, precision):
        video, audio = _parts(av_latent["samples"])
        orig_dtype = video.dtype
        dev = _resolve_device(device)
        compute_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        s = video.to(dev, dtype=compute_dtype, copy=True)
        b, c, t, h, w = s.shape
        selected = mode["mode"] if isinstance(mode, dict) else str(mode)
        if selected == UpscaleMode.SCALE_BY:
            scale = float(mode["scale"])
            target_w, target_h = w * scale, h * scale
        elif selected == UpscaleMode.TARGET_DIMENSIONS:
            target_w, target_h = float(mode["width"]) / 16, float(mode["height"]) / 16
            scale = ((target_w / w) + (target_h / h)) / 2
        elif selected == UpscaleMode.MEGAPIXELS:
            px = float(mode["megapixels"]) * 1024 * 1024 / (VAE_DOWNSAMPLE ** 2)
            target_h = math.sqrt(px * h / w); target_w = px / target_h; scale = ((target_w / w) + (target_h / h)) / 2
        else:
            raise ValueError(f"Unsupported upscale mode: {selected}")
        target_h = max(1, int(round(target_h / (align / 16))) * int(align / 16))
        target_w = max(1, int(round(target_w / (align / 16))) * int(align / 16))
        oh, ow = int(target_h), int(target_w)
        if oh < h or ow < w:
            raise ValueError("Upscaler only supports scale >= 1")
        if (oh, ow) == (h, w):
            return io.NodeOutput(av_latent)
        model = load_model(model_name, dev, precision)
        mean, std = _make_norm_tensors(dev, compute_dtype)
        def run_spatial_tiled(tile, overlap):
            rows, cols = _starts(h, tile, overlap), _starts(w, tile, overlap)
            out = torch.zeros((b, c, t, oh, ow), device=dev, dtype=compute_dtype)
            weights = torch.zeros((1, 1, 1, oh, ow), device=dev, dtype=torch.float32)
            with torch.inference_mode():
                for r0 in rows:
                    r1 = min(h, r0 + tile)
                    for c0 in cols:
                        c1 = min(w, c0 + tile)
                        piece = (s[:, :, :, r0:r1, c0:c1] - mean) / std
                        pred = model(piece, scale=scale, target_size=(t, max(1, round((r1-r0) * oh / h)), max(1, round((c1-c0) * ow / w))), enable_chunking=True)
                        pred = pred * std + mean
                        pr0, pc0 = round(r0 * oh / h), round(c0 * ow / w)
                        pr1, pc1 = min(oh, pr0 + pred.shape[-2]), min(ow, pc0 + pred.shape[-1])
                        wl = round(overlap * ow / w) if c0 > 0 else 0; wr = round(overlap * ow / w) if c1 < w else 0
                        wt = round(overlap * oh / h) if r0 > 0 else 0; wb = round(overlap * oh / h) if r1 < h else 0
                        win_h = _window(pred.shape[-2], wt, wb, dev); win_w = _window(pred.shape[-1], wl, wr, dev)
                        win = (win_h[:, None] * win_w[None, :]).view(1, 1, 1, pred.shape[-2], pred.shape[-1])
                        out[:, :, :, pr0:pr1, pc0:pc1] += pred[:, :, :, :pr1-pr0, :pc1-pc0] * win[:, :, :, :pr1-pr0, :pc1-pc0]
                        weights[:, :, :, pr0:pr1, pc0:pc1] += win[:, :, :, :pr1-pr0, :pc1-pc0]
                        del piece, pred, win
                        mm.soft_empty_cache()
            return out / weights.clamp_min(1e-8)

        # Fast path: try a single full-frame inference first. If the GPU cannot
        # hold the peak activations, transparently fall back to smaller tiles.
        try:
            with torch.inference_mode():
                full = (s - mean) / std
                out = model(full, scale=scale, target_size=(t, oh, ow), enable_chunking=True)
                out = out * std + mean
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            del full
            mm.soft_empty_cache()
            out = None
            tiled_error = None
            for tile_limit in (32, 24, 16, 8, 4):
                tile = max(4, min(h, w, tile_limit))
                overlap = min(tile - 1, max(1, tile // 8))
                try:
                    out = run_spatial_tiled(tile, overlap)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as tile_exc:
                    if not isinstance(tile_exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(tile_exc).lower():
                        raise
                    tiled_error = tile_exc
                    mm.soft_empty_cache()
                    gc.collect()
            if out is None:
                raise RuntimeError("Minimax H3 GH upscaler could not fit even the smallest automatic tile") from tiled_error
        finally:
            if 's' in locals():
                del s
        out = out.to("cpu", dtype=orig_dtype)
        result = av_latent.copy()
        result["samples"] = NestedTensor((out, audio.to("cpu")))
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            mask_parts = tuple(masks.unbind())
            if len(mask_parts) == 2:
                video_mask, audio_mask = mask_parts
                if isinstance(video_mask, torch.Tensor) and video_mask.ndim == 5:
                    vm = video_mask.to(dev, dtype=torch.float32)
                    vm = F.interpolate(vm, size=(vm.shape[2], oh, ow), mode="trilinear", align_corners=False)
                    video_mask = vm.to("cpu", dtype=mask_parts[0].dtype)
                result["noise_mask"] = NestedTensor((video_mask, audio_mask.to("cpu")))
        if dev.type == "cuda" and force_unload:
            model.to("cpu")
            mm.soft_empty_cache()
        gc.collect()
        return io.NodeOutput(result)


NODE_CLASS_MAPPINGS = {"GoohaiMinimaxH3LatentUpscalerGH": GoohaiMinimaxH3LatentUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"GoohaiMinimaxH3LatentUpscalerGH": "Minimax H3 Latent Upscaler (GH)"}
