
import inspect

import torch
import torch.nn.functional as F
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_management
from comfy.nested_tensor import NestedTensor
from comfy.k_diffusion.sampling import to_d
from comfy.utils import model_trange
import latent_preview
from comfy_api.latest import io
from .sampling import time_shift_sigma, time_shift_slope
from comfy.ldm.minimax.model import PackedLayout

H3_VIDEO_FRAMES = 17
H3_LATENT_CHANNELS = 24
H3_LATENT_TIME = 5

def _make_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                        keyframes=None, refs=None, frame_count=None):
    kwargs = {"keyframes": keyframes, "refs": refs}
    try:
        parameters = inspect.signature(PackedLayout.__init__).parameters
        supports_extra = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "frame_count" in parameters or supports_extra:
            kwargs["frame_count"] = frame_count
    except (TypeError, ValueError):
        pass
    return PackedLayout(
        text_len, latent_t, latent_h, latent_w, audio_t, **kwargs
    )

def _resize_keyframe_image(image, target_h, target_w):

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"keyframe source_image must be Tensor, got {type(image).__name__}")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"keyframe source_image must be BHWC, got shape={tuple(image.shape)}")

    x = image[:1]
    if x.shape[1] != target_h or x.shape[2] != target_w:
        x = F.interpolate(
            x.permute(0, 3, 1, 2), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
    return x

def _reencode_keyframe(item, full_h, full_w, debug=False):

    source_image = item.get("source_image")
    video_vae = item.get("video_vae")
    if source_image is None or video_vae is None or not callable(getattr(video_vae, "encode", None)):
        return None
    target_h, target_w = int(full_h) * 16, int(full_w) * 16
    image = _resize_keyframe_image(source_image, target_h, target_w)
    latent = video_vae.encode(image)
    if not isinstance(latent, torch.Tensor) or latent.dim() != 5:
        raise TypeError(
            "re-encoded H3 keyframe must be a 5D tensor, "
            f"got {type(latent).__name__} shape={getattr(latent, 'shape', None)}"
        )
    if tuple(latent.shape[-2:]) != (int(full_h), int(full_w)):
        raise ValueError(
            "re-encoded H3 keyframe spatial size mismatch: "
            f"expected {(int(full_h), int(full_w))}, got {tuple(latent.shape[-2:])}"
        )
    if debug:
        print(
            f"    [H3 关键帧] source_image 重新编码: "
            f"pixels={target_w}x{target_h} latent={tuple(latent.shape)}"
        )
    return latent

def _h3_extract(samples, debug=False):

    type_name = type(samples).__name__

    if hasattr(samples, "is_nested") and samples.is_nested:
        try:
            parts = list(samples.unbind())
            video = None
            audio = None
            for p in parts:
                if isinstance(p, torch.Tensor):
                    if video is None:

                        video = p
                    else:
                        audio = p
            if video is not None:
                if debug:
                    print(f"  · [H3 extract] NestedTensor video={tuple(video.shape)} "
                          f"audio={tuple(audio.shape) if audio is not None else None}")
                return video, audio, {"type": "nested_tensor"}
        except Exception as e:
            if debug:
                print(f"  · [H3 extract] NestedTensor unbind failed: {e}")

    if isinstance(samples, torch.Tensor):
        if debug:
            print(f"  · [H3 extract] plain tensor {tuple(samples.shape)}")
        return samples, None, {"type": "tensor"}

    if isinstance(samples, (tuple, list)):
        video = None
        audio = None
        for item in samples:
            if isinstance(item, torch.Tensor):
                if video is None:
                    video = item
                else:
                    audio = item
        if video is not None:
            fmt = "tuple" if isinstance(samples, tuple) else "list"
            if debug:
                print(f"  · [H3 extract] {fmt} video={tuple(video.shape)} "
                      f"audio={tuple(audio.shape) if audio is not None else None}")
            return video, audio, {"type": fmt}
        raise TypeError(
            f"H3 extract: {type_name} 中未找到 video tensor. "
            f"items: {[type(it).__name__ for it in samples]}"
        )

    pub_attrs = [a for a in dir(samples) if not a.startswith("_")][:25]
    raise TypeError(
        f"H3 extract: 不支持的格式 '{type_name}'. "
        f"期望 5D tensor / NestedTensor / (tensor, tensor) tuple. "
        f"可用属性: {pub_attrs}"
    )

def _h3_reconstruct(video, audio, format_info, debug=False):

    fmt = format_info.get("type", "tensor")
    if fmt == "nested_tensor":
        parts = [video] + ([audio] if audio is not None else [])
        return NestedTensor(parts)
    if fmt == "tensor":
        return video
    if fmt == "tuple":
        return (video, audio) if audio is not None else (video,)
    if fmt == "list":
        return [video, audio] if audio is not None else [video]

    return (video, audio) if audio is not None else video

def _h3_make_nested(video, audio):

    if audio is not None:
        return NestedTensor([video, audio])
    return video

def _h3_noise_masks(latent_dict):

    masks = latent_dict.get("noise_mask")
    if masks is None:
        return None, None
    if getattr(masks, "is_nested", False):
        parts = tuple(masks.unbind())
        if len(parts) == 2:
            return parts[0], parts[1]
    if isinstance(masks, torch.Tensor):
        return masks, None
    raise ValueError("H3: unsupported AV noise_mask layout")

def _h3_make_mask(video_mask, audio_mask):

    if video_mask is None and audio_mask is None:
        return None
    if video_mask is None:
        raise ValueError("H3: audio noise mask requires a video noise mask")
    if audio_mask is not None:
        return NestedTensor((video_mask, audio_mask))
    return video_mask

def _crop_spatial_tensor(tensor, tile_axis, start, end):

    if tensor is None:
        return None
    if tile_axis == "H":
        return tensor[:, :, :, start:end, :].contiguous()
    return tensor[:, :, :, :, start:end].contiguous()

def _adjust_frame_count(latent_5d, target_frames, mode, debug=False):

    B, C, T, H, W = latent_5d.shape
    target_T = round((target_frames - 3) / 4) + 1

    if T >= target_T:
        return latent_5d

    if mode == "error":
        raise ValueError(
            f"H3: latent 时间维 T={T}, 期望至少 T={target_T} "
            f"(对应 {target_frames} 帧). 当前 mode=error, 请调整输入或换模式."
        )

    if mode == "replicate_last":
        pad_n = target_T - T
        last = latent_5d[:, :, -1:, :, :].expand(-1, -1, pad_n, -1, -1)
        out = torch.cat([latent_5d, last], dim=2)
        if debug:
            print(f"  · [frame] replicate_last: T {T} -> {target_T} (+{pad_n})")
    elif mode == "zero":
        pad_n = target_T - T
        zeros = torch.zeros(
            B, C, pad_n, H, W,
            dtype=latent_5d.dtype, device=latent_5d.device
        )
        out = torch.cat([latent_5d, zeros], dim=2)
        if debug:
            print(f"  · [frame] zero: T {T} -> {target_T} (+{pad_n})")
    else:
        raise ValueError(f"H3: 未知 pad 模式 '{mode}'")

    return out.contiguous()

def _compute_tile_starts(total, n_tiles, overlap):

    if n_tiles <= 1:
        return [0], total

    stride = int(round(total / n_tiles))

    tile_size = stride + 2 * overlap

    starts = []
    for i in range(n_tiles):
        start = i * stride - overlap
        start = max(0, start)
        starts.append(start)

    dedup = []
    for s in starts:
        if not dedup or s > dedup[-1]:
            dedup.append(s)
    starts = dedup

    return starts, tile_size

def _make_window_1d(length, ov_left, ov_right, dtype, device):

    w = torch.ones(length, dtype=dtype, device=device)
    if ov_left > 0:
        n = min(ov_left, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos(t * 3.14159265)
            w[:n] = torch.minimum(w[:n], fade)
    if ov_right > 0:
        n = min(ov_right, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos((1 - t) * 3.14159265)
            w[-n:] = torch.minimum(w[-n:], fade)
    return w

class _GoohaiMinimaxH3TiledSamplerLegacy:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE", {
                    "tooltip": "随机噪声。通常与普通 SamplerCustomAdvanced 接法相同，二采可与一采共用同一个 RandomNoise。"
                }),
                "guider": ("GUIDER", {
                    "tooltip": "引导器：决定使用哪个 H3 模型和哪组提示词/首尾帧条件。高清二采复用低清 guider 时，本节点会跳过低清关键帧的二次硬注入，避免首帧发糊；若关键帧已按高清尺寸重新编码，则会正常保留。"
                }),
                "sampler": ("SAMPLER", {
                    "tooltip": "采样算法，例如 Euler。它决定每个 tile 如何逐步去噪。"
                }),
                "sigmas": ("SIGMAS", {
                    "tooltip": "噪声调度/采样步。二采时它同时决定采样步数和重绘强度；denoise 越高，对一采画面改动越大。"
                }),
                "latent_image": ("LATENT", {
                    "tooltip": "音视频联合潜空间（AV_Latent）。"
                }),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否分块 / Enable tiling：开启=分块采样，关闭=整幅一次采样。"
                }),
                "n_tiles": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "分块数。数值越大，理论上单块显存越低，但采样次数更多、速度更慢。1 等同不分块。"
                }),
                "tile_overlap": ("INT", {
                    "default": 128, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "重叠宽度 / Tile overlap，单位为输出像素。"
                }),
                "refine_seams": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "旧兼容路径的接缝二次精修。非 Euler 等模式会在融合后对接缝再做低噪采样；Euler 已自动逐步同步融合，不会重复执行。"
                }),
                "refine_steps": ("INT", {
                    "default": 4, "min": 1, "max": 25, "step": 1,
                    "tooltip": "仅在旧兼容路径且开启接缝精修时生效。步数越多速度越慢，建议 4~8；Euler 逐步同步模式会忽略此项。"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("输出", "降噪输出")
    FUNCTION = "sample_tiled"
    CATEGORY = "Goohai/MiniMax H3 Integration"
    DESCRIPTION = (
        "用于 H3 高清二采：把视频 latent 沿长边分块采样，再用渐变权重融合，"
        "以降低单次显存压力。音频保持完整形状参与联合采样；高清 FL2VA 首/尾帧按 tile 精确裁切，"
        "低清首/尾帧不会插值后重复硬注入，避免高清二采首帧拖影和发糊。"
        "Euler 会自动使用逐步同步融合，提高跨块人物一致性；其他采样器自动兼容旧逻辑。"
        "块数越多速度越慢；重叠越大接缝越轻，但节省的显存越少。"
    )

    def sample_tiled(self, noise, guider, sampler, sigmas, latent_image,
                     enable_tiling=True, n_tiles=2, tile_overlap=128,
                     refine_seams=False, refine_steps=4, **legacy_options):
        tile_axis = "auto"
        max_size_for_no_tile = 384
        target_frames = 17
        frame_padding_mode = "replicate_last"
        debug = True

        latent = latent_image.copy()
        raw_samples = latent["samples"]

        if debug:
            print(f"→ [H3] TiledSampler(Fixed): input type="
                  f"{type(raw_samples).__name__} enable_tiling={enable_tiling}")

        try:
            video_tensor, audio_tensor, fmt_info = _h3_extract(raw_samples, debug)
        except TypeError as e:
            print(f"→ [H3] TiledSampler: extract failed: {e}")
            raise

        if video_tensor.dim() != 5:
            raise ValueError(
                f"H3: video latent 必须是 5D [B,C,T,H,W], "
                f"实际 {video_tensor.dim()}D shape={tuple(video_tensor.shape)}"
            )

        if audio_tensor is not None:
            if debug:
                print(f"→ [H3] audio 已保留: shape={tuple(audio_tensor.shape)}, "
                      f"保持完整形状并参与联合采样 (不参与空间分块)")
        else:
            print(f"→ [H3] ⚠ 未检测到 audio latent. "
                  f"若使用 H3 视频+音频联合模型，将导致 x[1] 越界. "
                  f"请确认 latent_image 是 NestedTensor(video, audio).")

        video_tensor = _adjust_frame_count(
            video_tensor, target_frames, frame_padding_mode, debug
        )

        B, C, F, H, W = video_tensor.shape
        video_mask, audio_mask = _h3_noise_masks(latent)
        if debug and audio_mask is not None and torch.any(audio_mask < 1.0).item():
            print(
                "  · [H3 原声同步] 检测到锁定音频 mask：二采将保持完整音频条件，"
                "并按 mask 锁定音频 latent"
            )

        if not enable_tiling:
            if debug:
                print(f"  · bypass: 单次采样 (video shape={tuple(video_tensor.shape)})")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info,
                video_mask, audio_mask, debug
            )

        if tile_axis == "auto":
            tile_axis = "H" if H >= W else "W"
        axis_size = H if tile_axis == "H" else W

        max_size_latent = max(1, int(round(int(max_size_for_no_tile) / 16.0)))
        if axis_size <= max_size_latent or n_tiles <= 1:
            if debug:
                reason = ("axis_size ≤ max" if axis_size <= max_size_latent
                          else f"n_tiles={n_tiles}")
                print(f"  · auto-bypass ({reason})")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info,
                video_mask, audio_mask, debug
            )

        overlap_pixels = max(0, int(tile_overlap))
        overlap_latent = int(round(overlap_pixels / 16.0))
        if overlap_pixels > 0 and overlap_latent == 0:
            overlap_latent = 1
        starts, tile_size = _compute_tile_starts(axis_size, n_tiles, overlap_latent)
        if tile_size >= axis_size * 0.9 and len(starts) > 1:
            print(
                f"→ [H3 分块提示] 当前轴长度={axis_size} latent、重叠={overlap_pixels}px/{overlap_latent}latent，"
                f"导致单块长度={tile_size}，已经接近整幅画面。这样几乎不能节省显存；"
                f"建议把重叠像素降到 64~128，或减少分块数。"
            )
        if debug:
            print(f"  · axis={tile_axis} size={axis_size} "
                  f"n_tiles={n_tiles} overlap={overlap_pixels}px/{overlap_latent}latent "
                  f"starts={starts} tile_size={tile_size}")

        device = comfy.model_management.get_torch_device()
        dtype = video_tensor.dtype

        video_tensor = video_tensor.to(device=device)

        if audio_tensor is not None:
            audio_tensor = audio_tensor.to(device=device)

        output = torch.zeros_like(video_tensor, dtype=torch.float32, device=device)
        weights_shape = (1, 1, 1,
                         H if tile_axis == "H" else 1,
                         W if tile_axis == "W" else 1)
        weights = torch.zeros(weights_shape, dtype=torch.float32, device=device)

        denoised_output = torch.zeros_like(output)
        denoised_present = False

        sampled_audio_accum = None
        sampled_audio_count = 0

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        _GoohaiMinimaxH3TiledSamplerLegacy._clean_minimax_layout(guider, debug)

        keyframe_contexts = _GoohaiMinimaxH3TiledSamplerLegacy._prepare_minimax_keyframes(
            guider, H, W, debug
        )

        full_nested = _h3_make_nested(video_tensor, audio_tensor)
        full_noise = noise.generate_noise({"samples": full_nested})
        if hasattr(full_noise, "is_nested") and full_noise.is_nested:
            full_video_noise = full_noise.unbind()[0]
            full_audio_noise = full_noise.unbind()[1] if len(full_noise.unbind()) > 1 else None
        else:
            full_video_noise = full_noise
            full_audio_noise = None

        if self._can_use_synchronized_euler(sampler, keyframe_contexts):
            if debug:
                print("  · [一致性优化] 启用 Euler 逐步同步分块：每一步融合全部 tile 后再推进")
                if refine_seams:
                    print("  · [一致性优化] 同步融合已逐步消除接缝，本次不再重复执行接缝二采")
            return self._sample_synchronized_euler(
                latent, noise, guider, sampler, sigmas,
                video_tensor, audio_tensor, fmt_info,
                full_noise, starts, tile_size, tile_axis,
                keyframe_contexts, video_mask, audio_mask, debug,
            )

        if debug:
            sampler_name = getattr(getattr(sampler, "sampler_function", None), "__name__", "未知")
            reason = "存在高清硬关键帧" if any(
                not ctx.get("disable_hard_injection") for ctx in keyframe_contexts
            ) else f"采样器={sampler_name}"
            print(f"  · [兼容路径] {reason}，沿用逐块完整采样")

        for tile_idx, ax_start in enumerate(starts):
            if tile_axis == "H":
                ax_end = min(ax_start + tile_size, H)
                tile_latent = video_tensor[:, :, :, ax_start:ax_end, :].contiguous()
                tile_video_noise = full_video_noise[:, :, :, ax_start:ax_end, :].contiguous()
            else:
                ax_end = min(ax_start + tile_size, W)
                tile_latent = video_tensor[:, :, :, :, ax_start:ax_end].contiguous()
                tile_video_noise = full_video_noise[:, :, :, :, ax_start:ax_end].contiguous()

            actual_size = tile_latent.shape[3 if tile_axis == "H" else 4]

            if debug:
                print(f"  · tile {tile_idx+1}/{len(starts)}: "
                      f"range=[{ax_start},{ax_end}) "
                      f"video shape={tuple(tile_latent.shape)}")

            tile_nested = _h3_make_nested(tile_latent, audio_tensor)

            tile_noise = _h3_make_nested(tile_video_noise, full_audio_noise)
            tile_mask = _h3_make_mask(
                _crop_spatial_tensor(video_mask, tile_axis, ax_start, ax_end),
                audio_mask,
            )

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )

            _GoohaiMinimaxH3TiledSamplerLegacy._apply_minimax_keyframe_region(
                keyframe_contexts, tile_axis, ax_start, ax_end, debug
            )
            _GoohaiMinimaxH3TiledSamplerLegacy._clean_minimax_layout(guider, debug)
            try:
                tile_samples = guider.sample(
                    tile_noise, tile_nested, sampler, sigmas,
                    denoise_mask=tile_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=noise.seed,
                )
            finally:
                _GoohaiMinimaxH3TiledSamplerLegacy._restore_minimax_keyframes(keyframe_contexts)

            if hasattr(tile_samples, "is_nested") and tile_samples.is_nested:
                tile_parts = tile_samples.unbind()
                tile_samples_video = tile_parts[0]
                tile_samples_audio = tile_parts[1] if len(tile_parts) > 1 else None
            else:
                tile_samples_video = tile_samples
                tile_samples_audio = None

            tile_samples_video = tile_samples_video.to(device=device)

            if debug and isinstance(tile_samples_video, torch.Tensor):
                v_min = tile_samples_video.min().item()
                v_max = tile_samples_video.max().item()
                v_mean = tile_samples_video.mean().item()
                print(f"    sampled video: shape={tuple(tile_samples_video.shape)} "
                      f"range=[{v_min:.3f},{v_max:.3f}] mean={v_mean:.3f}")

            has_prev = tile_idx > 0
            has_next = tile_idx < len(starts) - 1
            ov_left = 0
            ov_right = 0
            if has_prev:
                prev_end = starts[tile_idx - 1] + tile_size
                ov_left = max(0, min(prev_end, ax_end) - ax_start)
            if has_next:
                next_start = starts[tile_idx + 1]
                ov_right = max(0, ax_end - max(ax_start, next_start))

            ov_left = min(ov_left, actual_size)
            ov_right = min(ov_right, actual_size)

            window_1d = _make_window_1d(
                actual_size, ov_left, ov_right, torch.float32, device
            )
            if tile_axis == "H":
                window = window_1d.view(1, 1, 1, -1, 1)
                output[:, :, :, ax_start:ax_end, :] += tile_samples_video.float() * window
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                window = window_1d.view(1, 1, 1, 1, -1)
                output[:, :, :, :, ax_start:ax_end] += tile_samples_video.float() * window
                weights[:, :, :, :, ax_start:ax_end] += window

            try:
                model = guider.model_patcher.model
                if hasattr(model, "process_latent_out") and "x0" in x0_output and x0_output["x0"] is not None:
                    x0 = x0_output["x0"]

                    if hasattr(x0, "is_nested") and x0.is_nested:
                        x0_video = x0.unbind()[0]
                    else:
                        x0_video = x0
                    x0_proc = model.process_latent_out(x0_video)
                    if isinstance(x0_proc, torch.Tensor) and x0_proc.shape == tile_samples_video.shape:
                        denoised_present = True
                        x0_proc = x0_proc.to(device=device)
                        if tile_axis == "H":
                            denoised_output[:, :, :, ax_start:ax_end, :] += \
                                x0_proc.float() * window
                        else:
                            denoised_output[:, :, :, :, ax_start:ax_end] += \
                                x0_proc.float() * window
            except Exception as e:
                if debug:
                    print(f"    ⚠ x0 处理失败: {type(e).__name__}: {e}")

            if debug:
                print(f"    fades: left={ov_left} right={ov_right} "
                      f"weight_acc min={weights.min().item():.3f}")

            del tile_samples, tile_samples_video, tile_samples_audio, tile_latent, tile_nested, tile_noise, window, window_1d
            if device.type == "cuda":
                torch.cuda.empty_cache()

        wmin = weights.min().item()
        wmax = weights.max().item()
        if debug:
            print(f"  · final weights: min={wmin:.4f} max={wmax:.4f}")
        if wmin < 1e-3:
            print(f"→ [H3] ⚠ weight min={wmin:.4f} 太小, 建议增大 tile_overlap.")
        if wmax > 1.05:
            print(f"→ [H3] ⚠ weight max={wmax:.4f} > 1.05, cosine 渐变异常.")

        output = output / weights.clamp(min=1e-8)
        if denoised_present:
            denoised_output = denoised_output / weights.clamp(min=1e-8)
        del weights

        if refine_seams and len(starts) > 1:
                output = self._refine_seams(
                output, full_video_noise, audio_tensor, full_audio_noise,
                starts, tile_size, overlap_latent, tile_axis, noise, guider, sampler, sigmas,
                refine_steps, device, dtype, keyframe_contexts,
                video_mask, audio_mask, debug
            )

        intermediate_device = comfy.model_management.intermediate_device()
        output = output.to(dtype=dtype, device=intermediate_device)
        if denoised_present:
            denoised_output_final = denoised_output.to(
                dtype=dtype, device=intermediate_device
            )
            del denoised_output
        else:
            denoised_output_final = output
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if debug:
            print(f"→ [H3] final output video: shape={tuple(output.shape)} "
                  f"dtype={output.dtype}")

        final_audio = audio_tensor.to(intermediate_device) if audio_tensor is not None else None
        reconstructed = _h3_reconstruct(output, final_audio, fmt_info, debug)
        denoised_reconstructed = _h3_reconstruct(
            denoised_output_final, final_audio, fmt_info, debug
        )

        out_dict = latent.copy()
        out_dict["samples"] = reconstructed
        out_denoised_dict = latent.copy()
        out_denoised_dict["samples"] = denoised_reconstructed

        return (out_dict, out_denoised_dict)

    @staticmethod
    def _can_use_synchronized_euler(sampler, keyframe_contexts):

        sampler_function = getattr(sampler, "sampler_function", None)
        is_standard_euler = getattr(sampler_function, "__name__", "") == "sample_euler"
        is_goohai_dual = bool(getattr(sampler, "_goohai_dual_clock", False))
        if not (is_standard_euler or is_goohai_dual):
            return False

        if is_goohai_dual:
            return True
        return not any(not ctx.get("disable_hard_injection") for ctx in keyframe_contexts)

    @staticmethod
    def _sample_synchronized_euler(
        latent_dict, noise, guider, sampler, sigmas,
        video_tensor, audio_tensor, fmt_info,
        full_noise, starts, tile_size, tile_axis,
        keyframe_contexts, video_mask=None, audio_mask=None, debug=False,
    ):

        full_nested = _h3_make_nested(video_tensor, audio_tensor)
        full_shapes = [tuple(video_tensor.shape)]
        if audio_tensor is not None:
            full_shapes.append(tuple(audio_tensor.shape))

        axis_dim = 3 if tile_axis == "H" else 4
        axis_total = video_tensor.shape[axis_dim]

        regions = []
        for tile_idx, ax_start in enumerate(starts):
            ax_end = min(ax_start + tile_size, axis_total)
            actual_size = ax_end - ax_start
            prev_end = starts[tile_idx - 1] + tile_size if tile_idx > 0 else ax_start
            next_start = starts[tile_idx + 1] if tile_idx < len(starts) - 1 else ax_end
            ov_left = min(actual_size, max(0, min(prev_end, ax_end) - ax_start))
            ov_right = min(actual_size, max(0, ax_end - max(ax_start, next_start)))
            window_1d = _make_window_1d(
                actual_size, ov_left, ov_right, torch.float32, video_tensor.device
            )
            window = (
                window_1d.view(1, 1, 1, -1, 1)
                if tile_axis == "H"
                else window_1d.view(1, 1, 1, 1, -1)
            )
            regions.append((ax_start, ax_end, window))

        weight_shape = (
            (1, 1, 1, axis_total, 1)
            if tile_axis == "H"
            else (1, 1, 1, 1, axis_total)
        )
        weights = torch.zeros(weight_shape, dtype=torch.float32, device=video_tensor.device)
        for ax_start, ax_end, window in regions:
            if tile_axis == "H":
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                weights[:, :, :, :, ax_start:ax_end] += window
        weights = weights.clamp(min=1e-8)

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        original_extra = dict(getattr(sampler, "extra_options", {}) or {})
        original_inpaint = dict(getattr(sampler, "inpaint_options", {}) or {})
        dual_clock = bool(getattr(sampler, "_goohai_dual_clock", False))
        shift_video = float(getattr(sampler, "_goohai_shift_video", 12.0))
        shift_audio = float(getattr(sampler, "_goohai_shift_audio", 3.0))
        audio_velocity_is_raw = bool(getattr(sampler, "_goohai_audio_velocity_is_raw", False))
        full_denoise_mask = _h3_make_mask(video_mask, audio_mask)

        @torch.no_grad()
        def synchronized_euler(
            model, x, step_sigmas, extra_args=None, callback=None, disable=None,
            s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
            **_unused,
        ):
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])

            prepared_model = model.inner_model.inner_model
            saved_model_shapes = getattr(prepared_model, "latent_shapes", None)
            saved_conds = {}
            payload_conds = []
            packed_full_mask = extra_args.get("denoise_mask")
            full_mask_streams = (
                comfy.utils.unpack_latents(packed_full_mask, full_shapes)
                if packed_full_mask is not None else None
            )
            full_latent_streams = (
                comfy.utils.unpack_latents(model.latent_image, full_shapes)
                if getattr(model, "latent_image", None) is not None else None
            )
            full_noise_streams = (
                comfy.utils.unpack_latents(model.noise, full_shapes)
                if getattr(model, "noise", None) is not None else None
            )

            for cond_group in getattr(model.inner_model, "conds", {}).values():
                if cond_group is None:
                    continue
                for cond in cond_group:
                    model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                    payload_cond = model_conds.get("minimax_payload")
                    payload = getattr(payload_cond, "cond", None)
                    if isinstance(payload, dict):
                        payload_conds.append((payload_cond, payload))

            def set_tile_shapes(tile_shapes):
                prepared_model.latent_shapes = tile_shapes
                for cond_group in getattr(model.inner_model, "conds", {}).values():
                    if cond_group is None:
                        continue
                    for cond in cond_group:
                        model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                        shape_cond = model_conds.get("latent_shapes")
                        if shape_cond is not None and hasattr(shape_cond, "cond"):
                            saved_conds.setdefault(id(shape_cond), (shape_cond, shape_cond.cond))
                            shape_cond.cond = tile_shapes

            def crop_keyframe_latent(latent, start, end):
                if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
                    return latent
                region = (
                    latent[:, :, :, start:end, :]
                    if tile_axis == "H"
                    else latent[:, :, :, :, start:end]
                ).contiguous()
                pad_h = (-region.shape[-2]) % 2
                pad_w = (-region.shape[-1]) % 2
                if pad_h or pad_w:
                    region = F.pad(region, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                return region

            def install_tile_payloads(tile_shapes, start, end):
                vs = tile_shapes[0]
                tile_h = (int(vs[3]) + 1) // 2 * 2
                tile_w = (int(vs[4]) + 1) // 2 * 2
                audio_t = int(tile_shapes[1][-1]) if len(tile_shapes) > 1 else 0
                restorations = []
                for payload_cond, original_payload in payload_conds:
                    payload = dict(original_payload)
                    keyframes = []
                    for item in list(payload.get("keyframes") or []):
                        copied = dict(item)
                        copied["latent"] = crop_keyframe_latent(item.get("latent"), start, end)
                        keyframes.append(copied)
                    payload["keyframes"] = keyframes or payload.get("keyframes")

                    refs = list(payload.get("refs") or [])
                    text_tags = payload.get("text_token_tags")
                    old_layout = payload.get("layout")
                    text_len = 0
                    if old_layout is not None and getattr(old_layout, "segments", None):
                        text_len = int(old_layout.segments[0][1])
                    elif text_tags is not None:

                        text_len = int(text_tags.shape[-1]) if hasattr(text_tags, "shape") else int(text_tags.numel())
                    if text_len <= 0:
                        raise RuntimeError("H3 synchronized tiling could not determine text token length")

                    payload["layout"] = _make_packed_layout(
                        text_len, int(vs[2]), tile_h, tile_w, audio_t,
                        keyframes=keyframes or None,
                        refs=refs or None,
                        frame_count=payload.get("frame_count"),
                    )

                    ref_latents = [
                        r.get("latent") for r in refs
                        if r.get("latent") is not None and r.get("kind") != "t8_keyframe_latent"
                    ]
                    ordered = []
                    kfi = refi = 0
                    for _, _, kind in payload["layout"].segments:
                        if kind == "cond" and kfi < len(keyframes):
                            ordered.append(keyframes[kfi]["latent"])
                            kfi += 1
                        elif kind == "ref_img" and refi < len(ref_latents):
                            ordered.append(ref_latents[refi])
                            refi += 1
                    payload["cond_video_latents"] = ordered
                    restorations.append((payload_cond, payload_cond.cond))
                    payload_cond.cond = payload
                return restorations

            def restore_tile_payloads(restorations):
                for payload_cond, old_payload in restorations:
                    payload_cond.cond = old_payload

            try:
                for step_idx in model_trange(len(step_sigmas) - 1, disable=disable):
                    if s_churn > 0:
                        gamma = (
                            min(s_churn / (len(step_sigmas) - 1), 2 ** 0.5 - 1)
                            if s_tmin <= step_sigmas[step_idx] <= s_tmax else 0.0
                        )
                    else:
                        gamma = 0.0
                    sigma_hat = step_sigmas[step_idx] * (gamma + 1)
                    if gamma > 0:
                        eps = torch.randn_like(x) * s_noise
                        x = x + eps * (
                            sigma_hat ** 2 - step_sigmas[step_idx] ** 2
                        ) ** 0.5

                    streams = comfy.utils.unpack_latents(x, full_shapes)
                    video_x = streams[0]
                    audio_x = streams[1] if len(streams) > 1 else None
                    video_denoised = torch.zeros_like(video_x, dtype=torch.float32)
                    audio_denoised = (
                        torch.zeros_like(audio_x, dtype=torch.float32)
                        if audio_x is not None else None
                    )

                    for ax_start, ax_end, window in regions:
                        if tile_axis == "H":
                            video_tile = video_x[:, :, :, ax_start:ax_end, :].contiguous()
                        else:
                            video_tile = video_x[:, :, :, :, ax_start:ax_end].contiguous()

                        tile_streams = [video_tile]
                        if audio_x is not None:
                            tile_streams.append(audio_x)
                        tile_x, tile_shapes = comfy.utils.pack_latents(tile_streams)
                        tile_mask = None
                        if full_mask_streams is not None:
                            tile_mask_streams = [
                                _crop_spatial_tensor(
                                    full_mask_streams[0], tile_axis, ax_start, ax_end
                                )
                            ]
                            if len(full_mask_streams) > 1:
                                tile_mask_streams.append(full_mask_streams[1])
                            tile_mask, _ = comfy.utils.pack_latents(tile_mask_streams)
                        tile_latent_image = None
                        if full_latent_streams is not None:
                            tile_latent_streams = [
                                _crop_spatial_tensor(
                                    full_latent_streams[0], tile_axis, ax_start, ax_end
                                )
                            ]
                            if len(full_latent_streams) > 1:
                                tile_latent_streams.append(full_latent_streams[1])
                            tile_latent_image, _ = comfy.utils.pack_latents(tile_latent_streams)
                        tile_noise_value = None
                        if full_noise_streams is not None:
                            tile_noise_streams = [
                                _crop_spatial_tensor(
                                    full_noise_streams[0], tile_axis, ax_start, ax_end
                                )
                            ]
                            if len(full_noise_streams) > 1:
                                tile_noise_streams.append(full_noise_streams[1])
                            tile_noise_value, _ = comfy.utils.pack_latents(tile_noise_streams)
                        set_tile_shapes(tile_shapes)
                        tile_payload_restore = install_tile_payloads(
                            tile_shapes, ax_start, ax_end
                        )
                        saved_latent_image = getattr(model, "latent_image", None)
                        saved_noise = getattr(model, "noise", None)
                        tile_extra_args = dict(extra_args)
                        tile_extra_args["denoise_mask"] = tile_mask
                        if tile_latent_image is not None:
                            model.latent_image = tile_latent_image
                        if tile_noise_value is not None:
                            model.noise = tile_noise_value
                        try:
                            tile_pred = model(
                                tile_x, sigma_hat * s_in, **tile_extra_args
                            )
                        finally:
                            model.latent_image = saved_latent_image
                            model.noise = saved_noise
                            restore_tile_payloads(tile_payload_restore)
                        pred_streams = comfy.utils.unpack_latents(tile_pred, tile_shapes)
                        pred_video = pred_streams[0].float()

                        if tile_axis == "H":
                            video_denoised[:, :, :, ax_start:ax_end, :] += pred_video * window
                        else:
                            video_denoised[:, :, :, :, ax_start:ax_end] += pred_video * window
                        if audio_denoised is not None:

                            audio_denoised += pred_streams[1].float()

                    video_denoised /= weights
                    merged_streams = [video_denoised.to(dtype=video_x.dtype)]
                    if audio_denoised is not None:
                        audio_denoised /= float(len(regions))
                        merged_streams.append(audio_denoised.to(dtype=audio_x.dtype))
                    denoised, _ = comfy.utils.pack_latents(merged_streams)

                    prepared_model.latent_shapes = full_shapes
                    if callback is not None:
                        callback({
                            "x": x, "i": step_idx,
                            "sigma": step_sigmas[step_idx],
                            "sigma_hat": sigma_hat,
                            "denoised": denoised,
                        })
                    if dual_clock and audio_x is not None:

                        sigma_v = sigma_hat
                        sigma_v_next = step_sigmas[step_idx + 1]
                        sigma_a = time_shift_sigma(sigma_v, shift_video, shift_audio)
                        sigma_a_next = time_shift_sigma(sigma_v_next, shift_video, shift_audio)
                        slope = time_shift_slope(sigma_v, shift_video, shift_audio)
                        dv = to_d(video_x, sigma_v, merged_streams[0])

                        da = to_d(audio_x, sigma_v, merged_streams[1])
                        video_delta = sigma_v_next - sigma_v
                        audio_delta = sigma_a_next - sigma_a
                        if not audio_velocity_is_raw:
                            audio_delta = audio_delta / slope
                        if full_mask_streams is not None and len(full_mask_streams) > 1:
                            current_audio_mask = full_mask_streams[1].to(
                                device=audio_x.device, dtype=audio_x.dtype
                            )
                            audio_delta = video_delta + current_audio_mask * (
                                audio_delta - video_delta
                            )
                        x_streams = [video_x + dv * video_delta, audio_x + da * audio_delta]
                        x, _ = comfy.utils.pack_latents(x_streams)
                    else:
                        d = to_d(x, sigma_hat, denoised)
                        x = x + d * (step_sigmas[step_idx + 1] - sigma_hat)
                return x
            finally:
                prepared_model.latent_shapes = saved_model_shapes
                for cond_obj, old_value in saved_conds.values():
                    cond_obj.cond = old_value

        sync_sampler = comfy.samplers.KSAMPLER(
            synchronized_euler,
            extra_options=original_extra,
            inpaint_options=original_inpaint,
        )

        try:
            samples = guider.sample(
                full_noise, full_nested, sync_sampler, sigmas,
                denoise_mask=full_denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
        finally:
            _GoohaiMinimaxH3TiledSamplerLegacy._restore_minimax_keyframes(keyframe_contexts)

        sampled_video, sampled_audio, _ = _h3_extract(samples, debug)
        intermediate_device = comfy.model_management.intermediate_device()
        sampled_video = sampled_video.to(
            device=intermediate_device, dtype=video_tensor.dtype
        )

        final_audio = (
            audio_tensor.to(intermediate_device) if audio_tensor is not None else None
        )
        reconstructed = _h3_reconstruct(sampled_video, final_audio, fmt_info, debug)

        denoised_reconstructed = reconstructed
        try:
            x0 = x0_output.get("x0")
            if x0 is not None:
                x0_video, _x0_audio, _ = _h3_extract(x0, debug=False)
                x0_video = x0_video.to(
                    device=intermediate_device, dtype=video_tensor.dtype
                )
                denoised_reconstructed = _h3_reconstruct(
                    x0_video, final_audio, fmt_info, debug
                )
        except Exception as exc:
            if debug:
                print(f"  · [一致性优化] 去噪预测提取失败，回退采样结果: {exc}")

        out = latent_dict.copy()
        out["samples"] = reconstructed
        out_denoised = latent_dict.copy()
        out_denoised["samples"] = denoised_reconstructed

        del weights, regions
        if video_tensor.device.type == "cuda":
            torch.cuda.empty_cache()
        return (out, out_denoised)

    @staticmethod
    def _prepare_minimax_keyframes(guider, full_h, full_w, debug=False):

        contexts = []
        if not hasattr(guider, "original_conds"):
            return contexts

        for cond_key, cond_list in guider.original_conds.items():
            for cond in cond_list:
                if not isinstance(cond, dict):
                    continue
                original = cond.get("minimax_keyframes")
                if not original:
                    continue

                prepared = []
                mismatched = []
                reencoded = []
                for kf in original:
                    if not isinstance(kf, dict):
                        raise TypeError(
                            "H3 分块采样：minimax_keyframes 中存在非字典项目，"
                            f"无法适配（实际类型 {type(kf).__name__}）。"
                        )
                    item = kf.copy()
                    latent = item.get("latent")
                    if not isinstance(latent, torch.Tensor) or latent.dim() != 5:
                        shape = tuple(latent.shape) if isinstance(latent, torch.Tensor) else None
                        raise TypeError(
                            "H3 分块采样：首/尾帧条件必须是 5D latent "
                            f"[B,C,T,H,W]，实际类型={type(latent).__name__} shape={shape}。"
                        )

                    old_hw = tuple(latent.shape[-2:])
                    if old_hw != (full_h, full_w):

                        fresh = _reencode_keyframe(item, full_h, full_w, debug)
                        if fresh is not None:
                            item["latent"] = fresh
                            reencoded.append(old_hw)
                        else:
                            mismatched.append(old_hw)
                    prepared.append(item)

                    if debug:
                        print(
                            f"  · [H3 关键帧] {cond_key}: "
                            f"条件={old_hw[1]}x{old_hw[0]} "
                            f"二采={full_w}x{full_h}"
                        )

                disable_hard_injection = bool(mismatched)
                if reencoded and debug:
                    sizes = "、".join(
                        f"{w}x{h}" for h, w in dict.fromkeys(reencoded)
                    )
                    print(
                        f"  · [H3 关键帧] 已使用 Goohai source_image + video_vae "
                        f"重新编码高清条件（原 latent={sizes}，目标={full_w}x{full_h}）"
                    )
                if disable_hard_injection:
                    sizes = "、".join(
                        f"{w}x{h}" for h, w in dict.fromkeys(mismatched)
                    )
                    print(
                        f"→ [H3 关键帧保护] 检测到低清首/尾帧 latent（{sizes}），"
                        f"当前高清二采 latent 为 {full_w}x{full_h}。"
                        "已自动跳过低清关键帧的二次硬注入，避免首帧拖影/发糊；"
                        "一采 latent 中的首尾画面和提示词图像语义仍然保留。"
                        "如需高清硬锁定，请用 MiniMax H3 Image to Video (Tail) "
                        "按二采尺寸重新编码原始首/尾图。"
                    )

                contexts.append({
                    "cond": cond,
                    "original": original,
                    "prepared": prepared,
                    "disable_hard_injection": disable_hard_injection,
                })
        return contexts

    @staticmethod
    def _apply_minimax_keyframe_region(contexts, tile_axis, start, end, debug=False):

        for ctx in contexts:
            if ctx.get("disable_hard_injection"):

                ctx["cond"].pop("minimax_keyframes", None)
                continue

            tiled = []
            for kf in ctx["prepared"]:
                item = kf.copy()
                latent = item["latent"]
                if tile_axis == "H":
                    region = latent[:, :, :, start:end, :].contiguous()
                else:
                    region = latent[:, :, :, :, start:end].contiguous()

                pad_h = (-region.shape[-2]) % 2
                pad_w = (-region.shape[-1]) % 2
                if pad_h or pad_w:
                    region = F.pad(
                        region,
                        (0, pad_w, 0, pad_h, 0, 0),
                        mode="replicate",
                    )
                item["latent"] = region
                tiled.append(item)

                if debug:
                    print(
                        f"    [H3 关键帧] 当前区域 {tile_axis}[{start}:{end}] "
                        f"→ latent={tuple(region.shape)}"
                    )
            ctx["cond"]["minimax_keyframes"] = tiled

    @staticmethod
    def _restore_minimax_keyframes(contexts):

        for ctx in contexts:
            ctx["cond"]["minimax_keyframes"] = ctx["original"]

    @staticmethod
    def _refine_seams(output, full_video_noise, audio_tensor, full_audio_noise,
                      starts, tile_size, tile_overlap, tile_axis, noise, guider, sampler, sigmas,
                      refine_steps, device, dtype, keyframe_contexts=None,
                      video_mask=None, audio_mask=None, debug=False):

        if refine_steps <= 0 or sigmas.shape[-1] <= 1:
            return output

        refine_sigmas = sigmas[-(refine_steps + 1):].clone()
        if debug:
            print(f"  · [refine] 低噪 sigma 子序列: {refine_sigmas.tolist()}")

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        axis_dim = 3 if tile_axis == "H" else 4
        axis_total = output.shape[axis_dim]

        for seam_idx in range(len(starts) - 1):
            right_start = starts[seam_idx + 1]

            seam_center = right_start

            half = max(8, tile_overlap)
            band_start = max(0, seam_center - half)
            band_end = min(axis_total, seam_center + half)

            if band_end - band_start < 2:
                continue

            if debug:
                print(f"  · [refine] 接缝 {seam_idx+1}/{len(starts)-1}: "
                      f"band=[{band_start},{band_end})")

            if tile_axis == "H":
                band_latent = output[:, :, :, band_start:band_end, :].contiguous()
                band_noise = full_video_noise[:, :, :, band_start:band_end, :].contiguous()
            else:
                band_latent = output[:, :, :, :, band_start:band_end].contiguous()
                band_noise = full_video_noise[:, :, :, :, band_start:band_end].contiguous()

            band_nested = _h3_make_nested(band_latent, audio_tensor)
            band_noise_nested = _h3_make_nested(band_noise, full_audio_noise)
            band_mask = _h3_make_mask(
                _crop_spatial_tensor(video_mask, tile_axis, band_start, band_end),
                audio_mask,
            )

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, refine_sigmas.shape[-1] - 1, x0_output
            )
            _GoohaiMinimaxH3TiledSamplerLegacy._apply_minimax_keyframe_region(
                keyframe_contexts or [], tile_axis, band_start, band_end, debug
            )
            _GoohaiMinimaxH3TiledSamplerLegacy._clean_minimax_layout(guider, debug)
            try:
                band_samples = guider.sample(
                    band_noise_nested, band_nested, sampler, refine_sigmas,
                    denoise_mask=band_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=noise.seed,
                )
            finally:
                _GoohaiMinimaxH3TiledSamplerLegacy._restore_minimax_keyframes(keyframe_contexts or [])

            if hasattr(band_samples, "is_nested") and band_samples.is_nested:
                band_samples_video = band_samples.unbind()[0]
            else:
                band_samples_video = band_samples
            band_samples_video = band_samples_video.to(device=device)

            if tile_axis == "H":
                output[:, :, :, band_start:band_end, :] = band_samples_video.float()
            else:
                output[:, :, :, :, band_start:band_end] = band_samples_video.float()

            del band_latent, band_noise, band_nested, band_noise_nested, band_samples, band_samples_video
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return output

    @staticmethod
    def _clean_minimax_layout(guider, debug=False):

        if hasattr(guider, 'model_patcher') and hasattr(guider.model_patcher, 'model'):
            model = guider.model_patcher.model
            if hasattr(model, 'diffusion_model'):
                dm = model.diffusion_model
                if hasattr(model, '_cached_extra_conds'):
                    cached = model._cached_extra_conds
                    if isinstance(cached, dict):
                        for k, v in cached.items():
                            if hasattr(v, 'cond') and isinstance(v.cond, dict):
                                if 'layout' in v.cond:
                                    if debug:
                                        print(f"  · [H3] 清理已缓存的 layout")
                                    del v.cond['layout']
                                if 'cond_video_latents' in v.cond:
                                    if debug:
                                        print(f"  · [H3] 清理已缓存的 cond_video_latents")
                                    del v.cond['cond_video_latents']

    @staticmethod
    def _single_pass(noise, guider, sampler, sigmas, latent_dict,
                     video_tensor, audio_tensor, fmt_info,
                     video_mask=None, audio_mask=None, debug=False):

        _GoohaiMinimaxH3TiledSamplerLegacy._clean_minimax_layout(guider, debug)
        keyframe_contexts = _GoohaiMinimaxH3TiledSamplerLegacy._prepare_minimax_keyframes(
            guider, video_tensor.shape[-2], video_tensor.shape[-1], debug
        )

        latent_for_sample = _h3_reconstruct(video_tensor, audio_tensor, fmt_info, debug)
        latent_dict["samples"] = latent_for_sample

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        _GoohaiMinimaxH3TiledSamplerLegacy._apply_minimax_keyframe_region(
            keyframe_contexts, "H", 0, video_tensor.shape[-2], debug
        )
        _GoohaiMinimaxH3TiledSamplerLegacy._clean_minimax_layout(guider, debug)
        try:
            samples = guider.sample(
                noise.generate_noise(latent_dict),
                latent_for_sample,
                sampler,
                sigmas,
                denoise_mask=_h3_make_mask(video_mask, audio_mask),
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
        finally:
            _GoohaiMinimaxH3TiledSamplerLegacy._restore_minimax_keyframes(keyframe_contexts)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent_dict.copy()
        out["samples"] = samples

        out_denoised = out.copy()
        try:
            model = guider.model_patcher.model
            if hasattr(model, "process_latent_out") and "x0" in x0_output and x0_output["x0"] is not None:
                x0_proc = model.process_latent_out(x0_output["x0"])
                if isinstance(x0_proc, torch.Tensor) and x0_proc.shape == samples.shape:
                    out_denoised["samples"] = x0_proc
        except Exception:
            pass

        return (out, out_denoised)

class GoohaiMinimaxH3TiledSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GoohaiMinimaxH3TiledSampler",
            display_name="Minimax-H3 Second-Pass Tiled Sampler",
            category="Goohai/MiniMax H3 Integration/Sampling",
            inputs=[
                io.Noise.Input("noise"), io.Guider.Input("guider"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Boolean.Input("enable_tiling", default=True),
                io.Int.Input("n_tiles", default=2, min=1, max=8),
                io.Int.Input("tile_overlap", default=128, min=0, max=2048, step=64),
                io.Boolean.Input("refine_seams", default=False),
                io.Int.Input("refine_steps", default=4, min=1, max=25),
            ],
            outputs=[io.Latent.Output(display_name="输出"), io.Latent.Output(display_name="降噪输出")],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image,
                enable_tiling=True, n_tiles=2, tile_overlap=128,
                refine_seams=False, refine_steps=4):
        return io.NodeOutput(*_GoohaiMinimaxH3TiledSamplerLegacy().sample_tiled(
            noise, guider, sampler, sigmas, latent_image, enable_tiling,
            n_tiles, tile_overlap, refine_seams, refine_steps
        ))

NODE_CLASS_MAPPINGS = {
    "GoohaiMinimaxH3TiledSampler": GoohaiMinimaxH3TiledSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoohaiMinimaxH3TiledSampler": "Minimax-H3二采分块采样器 / Minimax-H3 Second-Pass Tiled Sampler",
}
