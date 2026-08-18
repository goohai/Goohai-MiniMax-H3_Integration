from __future__ import annotations

import math
import types

import torch

import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax import model as minimax_model
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts


DEFAULT_SAMPLER_NAME = "dual_clock_euler"
DEFAULT_SCHEDULER_NAME = "native_flow"
SAMPLER_OPTIONS = [DEFAULT_SAMPLER_NAME]
if hasattr(comfy.model_sampling, "ModelSamplingAV"):
    SAMPLER_OPTIONS.extend(
        name for name in comfy.samplers.SAMPLER_NAMES if name != DEFAULT_SAMPLER_NAME
    )
SCHEDULER_OPTIONS = [
    DEFAULT_SCHEDULER_NAME,
    *(name for name in comfy.samplers.SCHEDULER_NAMES if name != DEFAULT_SCHEDULER_NAME),
]

HYBRID_KEYFRAME_SENTINEL = "t8_keyframe_latent"
HYBRID_LAYOUT_PATCH_VERSION = 1


class MiniMaxH3FlowSamplingGH(
    comfy.model_sampling.ModelSamplingDiscreteFlow,
    comfy.model_sampling.CONST,
):
    @property
    def audio_scale(self):
        # This sampler advances audio on its own sigma clock. Keep ComfyUI's
        # FLOW_AV carry neutral so the audio transform is not applied twice.
        return 1.0


def model_uses_raw_audio_velocity(model) -> bool:
    base_model = getattr(model, "model", None)
    return callable(getattr(base_model, "audio_scale", None))


def shift_sigma(base_sigma, shift: float):
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base_sigma, to_shift)


def time_shift_slope(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base_sigma) ** 2
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base_sigma) ** 2
    return numerator / denominator


def native_flow_sigmas(steps: int, shift_video: float) -> torch.Tensor:
    base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift_sigma(base_sigmas, shift_video)


def _scheduler_sigmas(model_sampling, scheduler: str, steps: int, shift_video: float):
    if scheduler == DEFAULT_SCHEDULER_NAME:
        return native_flow_sigmas(steps, shift_video)
    if scheduler not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()


def _make_sampling(model, original_sampling, shift_video, shift_audio, use_native_av):
    if use_native_av:
        native_av_cls = getattr(comfy.model_sampling, "ModelSamplingAV", None)
        if native_av_cls is None or not model_uses_raw_audio_velocity(model):
            raise RuntimeError(
                "The selected sampler requires current MiniMax H3 FLOW_AV support. "
                "Use dual_clock_euler for cross-version compatibility."
            )

        class MiniMaxH3NativeAVSamplingGH(native_av_cls, comfy.model_sampling.CONST):
            pass

        model_sampling = MiniMaxH3NativeAVSamplingGH(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    else:
        model_sampling = MiniMaxH3FlowSamplingGH(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video)

    if hasattr(original_sampling, "noise_scale"):
        model_sampling.set_noise_scale(original_sampling.noise_scale)
    return model_sampling


def _clean_locked_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
    """Restore the pre-FLOW_AV H3 inpaint input for locked AV regions.

    ComfyUI's generic FLOW inpaint path mixes noise into latent_image before
    KSampler applies the denoise mask. For locked source audio this means the
    H3 model does not consistently see the clean drive-audio latent. Returning
    latent_image directly restores the original MiniMax H3 behavior locally on
    the cloned model used by this sampler.
    """
    return latent_image


def _has_locked_audio_region(av_latent: dict) -> bool:
    masks = av_latent.get("noise_mask")
    if not getattr(masks, "is_nested", False):
        return False
    parts = tuple(masks.unbind())
    if len(parts) != 2:
        return False
    audio_mask = parts[1]
    return bool(torch.any(audio_mask < 1.0).item())


def _pixel_frames_from_latent_t(latent_t: int) -> int:
    if latent_t < 1:
        raise RuntimeError("MiniMax H3 target video latent length must be positive")
    spans = getattr(minimax_model, "FRAME_PER_TOKEN", (1, 4, 4, 4, 4))
    return sum(spans[index % len(spans)] for index in range(latent_t))


def _hybrid_ref_advance(ref: dict) -> float:
    kind = ref.get("kind")
    if kind == HYBRID_KEYFRAME_SENTINEL:
        return 0.0
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(ref.get("ref_audio_t", 0))
    if kind in {"video", "video_audio"}:
        frame_rescale = float(getattr(minimax_model, "FRAME_RESCALE", 5.0 / 3.0))
        return max(
            float(ref.get("ref_audio_t", 0)),
            frame_rescale * _pixel_frames_from_latent_t(int(ref.get("latent_t", 0))),
        )
    raise RuntimeError(f"Unsupported MiniMax H3 Hybrid reference kind: {kind!r}")


def repair_hybrid_keyframe_layout(out: dict, kwargs: dict) -> dict:
    """Align exact keyframes with the target timeline after packed references.

    Current ComfyUI places reference media before the target AV sequence, but
    leaves first/last keyframe RoPE positions on the pre-reference timeline.
    Only Hybrid payloads need this correction; ordinary FL2VA is untouched.
    """
    keyframes = list(kwargs.get("minimax_keyframes") or [])
    refs = list(kwargs.get("minimax_refs") or [])
    if not keyframes or not refs:
        return out

    cond = out.get("minimax_payload")
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError("MiniMax H3 Hybrid patch could not access the packed payload")
    layout = payload.get("layout")
    if layout is None or not hasattr(layout, "signature") or not hasattr(layout, "segments"):
        raise RuntimeError("MiniMax H3 Hybrid packed layout is missing or incompatible")

    frame_count_value = kwargs.get("minimax_frame_count")
    if frame_count_value is None:
        raise RuntimeError("MiniMax H3 Hybrid conditioning is missing minimax_frame_count")
    frame_count = int(frame_count_value)
    text_len, latent_t = int(layout.signature[0]), int(layout.signature[1])
    frame_rescale = float(getattr(minimax_model, "FRAME_RESCALE", 5.0 / 3.0))
    target_span = frame_rescale * _pixel_frames_from_latent_t(latent_t)
    ref_offset = sum(_hybrid_ref_advance(ref) for ref in refs)

    cond_segments = [
        (int(start), int(stop))
        for start, stop, kind in layout.segments
        if kind == "cond"
    ]
    if len(cond_segments) != len(keyframes):
        raise RuntimeError(
            "MiniMax H3 Hybrid keyframe/layout count changed; GH compatibility patch refused"
        )

    for (start, stop), keyframe in zip(cond_segments, keyframes):
        pixel_index = int(keyframe["resolved_frame_index"])
        if pixel_index == 0:
            cond_t = float(text_len)
        elif pixel_index == frame_count - 1:
            cond_t = float(text_len) + target_span - frame_rescale
        else:
            raise RuntimeError(
                "MiniMax H3 Hybrid compatibility currently supports exact first/last keyframes only"
            )
        layout.position_ids[start:stop, 0] = cond_t + ref_offset
    return out


def _patch_hybrid_keyframe_layout(model):
    original_extra_conds = model.get_model_object("extra_conds")
    if getattr(original_extra_conds, "_gh_hybrid_layout_patch_version", None) is not None:
        return

    def _patched_extra_conds(_self, **kwargs):
        out = original_extra_conds(**kwargs)
        return repair_hybrid_keyframe_layout(out, kwargs)

    _patched_extra_conds._gh_hybrid_layout_patch_version = HYBRID_LAYOUT_PATCH_VERSION
    model.add_object_patch(
        "extra_conds", types.MethodType(_patched_extra_conds, model.model)
    )


def _audio_step_scale(
    sigma_video,
    sigma_audio,
    slope_audio,
    denoise_mask,
    audio_velocity_is_raw: bool,
):
    flat_scale = -sigma_video
    dual_scale = -sigma_audio if audio_velocity_is_raw else -sigma_audio / slope_audio
    if denoise_mask is None:
        return dual_scale
    return flat_scale + denoise_mask * (dual_scale - flat_scale)


def sample_minimax_h3_dual_clock_euler_gh(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
    audio_velocity_is_raw: bool = False,
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            f"MiniMax H3 packed latent changed: expected {packed_values}, got {x.shape[-1]}"
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError("MiniMax H3 denoise mask does not match the packed AV latent")
        audio_mask = denoise_mask[..., video_values:]

    s_in = x.new_ones([x.shape[0]])
    for step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
        sigma_video = sigmas[step]
        sigma_video_next = sigmas[step + 1]
        denoised = model(x, sigma_video * s_in, **extra_args)
        derivative = to_d(x, sigma_video, denoised)

        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        sigma_audio_next = time_shift_sigma(sigma_video_next, shift_video, shift_audio)
        slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)
        video_delta = sigma_video_next - sigma_video
        audio_delta = sigma_audio_next - sigma_audio
        if not audio_velocity_is_raw:
            audio_delta = audio_delta / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(
                sigma_video,
                sigma_audio,
                slope_audio,
                audio_mask,
                audio_velocity_is_raw,
            )
            denoised[..., video_values:] = (
                x[..., video_values:] + derivative[..., video_values:] * endpoint_scale
            )
            callback({
                "x": x,
                "i": step,
                "sigma": sigma_video,
                "sigma_hat": sigma_video,
                "denoised": denoised,
            })

        x = torch.cat((
            x[..., :video_values] + derivative[..., :video_values] * video_delta,
            x[..., video_values:] + derivative[..., video_values:] * audio_delta,
        ), dim=-1)
    return x


def setup_dual_clock_sampling_gh(
    model,
    av_latent: dict,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str = DEFAULT_SAMPLER_NAME,
    scheduler: str = DEFAULT_SCHEDULER_NAME,
):
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            f"Unexpected MiniMax H3 AV channels: video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if sampler_name not in SAMPLER_OPTIONS:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    use_native_av = sampler_name != DEFAULT_SAMPLER_NAME
    patched_model = model.clone()
    original_sampling = model.get_model_object("model_sampling")
    model_sampling = _make_sampling(
        model, original_sampling, shift_video, shift_audio, use_native_av
    )
    patched_model.add_object_patch("model_sampling", model_sampling)

    # Keep first/last keyframes on the target AV timeline when Hybrid refs are
    # packed before it. This is local to the cloned MODEL returned by GH.
    _patch_hybrid_keyframe_layout(patched_model)

    # Scope the compatibility behavior to cloned MODEL outputs that actually
    # contain locked/remixed audio. Native generated-audio runs keep the exact
    # current ComfyUI path, and the global MiniMax H3 class stays untouched.
    if _has_locked_audio_region(av_latent):
        clean_inpaint = types.MethodType(_clean_locked_latent_inpaint, patched_model.model)
        patched_model.add_object_patch("scale_latent_inpaint", clean_inpaint)

    transformer_options = patched_model.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = shift_video
    transformer_options["minimax_h3_sigma_shift_audio"] = shift_audio
    patched_model.model_options["transformer_options"] = transformer_options

    if use_native_av:
        sampler = comfy.samplers.sampler_object(sampler_name)
    else:
        audio_velocity_is_raw = model_uses_raw_audio_velocity(model)
        video_values = math.prod(video.shape[1:])
        packed_values = video_values + math.prod(audio.shape[1:])

        def sampler_function(model_wrap, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_minimax_h3_dual_clock_euler_gh(
                model_wrap,
                x,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                video_values=video_values,
                packed_values=packed_values,
                shift_video=shift_video,
                shift_audio=shift_audio,
                audio_velocity_is_raw=audio_velocity_is_raw,
            )

        sampler_function.__name__ = "sample_minimax_h3_dual_clock_euler_gh"
        sampler = comfy.samplers.KSAMPLER(sampler_function)

    sigmas = _scheduler_sigmas(model_sampling, scheduler, steps, shift_video)
    return patched_model, sampler, sigmas
