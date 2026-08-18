from __future__ import annotations

import gc
import json

import torch
import torch.nn.functional as torch_functional
import torchaudio

from comfy import model_management
from comfy_extras.nodes_audio import vae_decode_audio

from .core import encode_audio_once, nested_av_parts, replace_audio_latent, validate_audio


def clean_generated_audio_start(audio: dict) -> dict:
    """Suppress a short H3 start burst only when a silence gap precedes sustained audio."""
    waveform, sample_rate = validate_audio(audio, "decoded_audio")
    analysis_samples = min(waveform.shape[-1], round(sample_rate * 0.75))
    window_samples = max(1, round(sample_rate * 0.01))
    window_count = analysis_samples // window_samples
    if window_count < 20:
        return audio

    analysis = waveform[..., : window_count * window_samples]
    mono = analysis.float().pow(2).mean(dim=(0, 1)).reshape(window_count, window_samples)
    rms = mono.mean(dim=1).sqrt()
    peak = float(rms.max().item())
    if peak < 0.01:
        return audio

    active_threshold = max(0.008, peak * 0.12)
    silence_threshold = max(0.0025, peak * 0.045)
    active = rms >= active_threshold
    quiet = rms <= silence_threshold

    initial_limit = min(window_count, 5)  # the burst must begin within 50 ms
    initial_indices = torch.nonzero(active[:initial_limit], as_tuple=False).flatten()
    if initial_indices.numel() == 0:
        return audio
    burst_start = int(initial_indices[0].item())

    # Require at least 40 ms of near-silence after a short burst, which rules
    # out ordinary speech or music that simply begins at time zero.
    quiet_run = 4
    burst_end = None
    burst_end_limit = min(window_count - quiet_run, 25)
    for index in range(burst_start + 2, burst_end_limit + 1):
        if bool(quiet[index : index + quiet_run].all().item()):
            burst_end = index
            break
    if burst_end is None or int(active[burst_start:burst_end].sum().item()) < 2:
        return audio

    # Find a later, sustained main signal: at least 6 active windows in an
    # 80-ms interval, beginning after a meaningful silence gap.
    sustained_windows = 8
    main_onset = None
    search_start = burst_end + 6
    search_end = min(window_count - sustained_windows, 65)
    for index in range(search_start, search_end + 1):
        if int(active[index : index + sustained_windows].sum().item()) >= 6:
            main_onset = index
            break
    if main_onset is None:
        return audio

    gap = quiet[burst_end:main_onset]
    if gap.numel() < 6 or float(gap.float().mean().item()) < 0.6:
        return audio

    silence_end_sample = min(waveform.shape[-1], burst_end * window_samples)
    fade_end_sample = min(waveform.shape[-1], main_onset * window_samples)
    if fade_end_sample <= silence_end_sample:
        return audio

    cleaned = waveform.clone()
    cleaned[..., :silence_end_sample] = 0
    fade_length = fade_end_sample - silence_end_sample
    phase = torch.linspace(
        0.0, torch.pi / 2, fade_length,
        device=cleaned.device, dtype=cleaned.dtype,
    )
    fade = torch.sin(phase).square()
    cleaned[..., silence_end_sample:fade_end_sample] *= fade
    return {**audio, "waveform": cleaned}


def inject_audio_latent(av_latent: dict, source_audio: dict, audio_vae, mode: str, strength: float):
    mode = mode.lower()
    if mode not in {"lock", "remix"}:
        raise ValueError("Audio latent control mode must be lock or remix")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    encoded = encode_audio_once(audio_vae, source_audio)
    denoise = 0.0 if mode == "lock" else strength
    return replace_audio_latent(av_latent, encoded, denoise), source_audio


def decode_av_latent(av_latent: dict, video_vae, audio_vae):
    try:
        video, audio = nested_av_parts(av_latent)
        images = video_vae.decode(video)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
        if av_latent.get("gh_h3_audio_mode") in {"native", "reference_only"}:
            decoded_audio = clean_generated_audio_start(decoded_audio)
        video_latent = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
        audio_latent = video_latent.copy()
        video_latent["samples"] = video
        audio_latent["samples"] = audio
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            video_mask, audio_mask = masks.unbind()
            video_latent["noise_mask"] = video_mask
            audio_latent["noise_mask"] = audio_mask
        return images, decoded_audio, video_latent, audio_latent
    finally:
        # Release temporary decode allocations without unloading VAE models or
        # invalidating tensors that are returned to downstream nodes.
        gc.collect()
        model_management.soft_empty_cache()


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _match_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    channels = waveform.shape[1]
    if channels == target_channels:
        return waveform
    if channels == 1:
        return waveform.expand(-1, target_channels, -1)
    if target_channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    # For uncommon multichannel inputs, make a deterministic stereo-compatible downmix.
    mono = waveform.mean(dim=1, keepdim=True)
    return mono.expand(-1, target_channels, -1)


def mix_audio(
    source_audio: dict,
    generated_audio: dict,
    source_gain_db: float = 0.0,
    generated_gain_db: float = -6.0,
    duck_generated: float = 0.5,
    output_sample_rate: str = "source",
    peak_limit_dbfs: float = -1.0,
):
    source, source_rate = validate_audio(source_audio, "source_audio")
    generated, generated_rate = validate_audio(generated_audio, "generated_audio")
    if output_sample_rate == "source":
        target_rate = source_rate
    elif output_sample_rate == "generated":
        target_rate = generated_rate
    else:
        target_rate = int(output_sample_rate)

    source = _resample(source, source_rate, target_rate)
    generated = _resample(generated, generated_rate, target_rate)
    target_channels = max(source.shape[1], generated.shape[1])
    source = _match_channels(source, target_channels)
    generated = _match_channels(generated, target_channels)
    target_samples = max(source.shape[-1], generated.shape[-1])
    source = torch_functional.pad(source, (0, target_samples - source.shape[-1]))
    generated = torch_functional.pad(generated, (0, target_samples - generated.shape[-1]))

    source = source * (10.0 ** (source_gain_db / 20.0))
    generated = generated * (10.0 ** (generated_gain_db / 20.0))
    if duck_generated > 0:
        envelope = source.abs().mean(dim=1, keepdim=True)
        window = max(1, round(target_rate * 0.02))
        envelope = torch_functional.avg_pool1d(envelope, window, stride=1, padding=window // 2)
        envelope = envelope[..., :target_samples]
        activity = (envelope / 0.05).clamp(0.0, 1.0)
        generated = generated * (1.0 - float(duck_generated) * activity)

    mixed = source + generated
    limit = 10.0 ** (peak_limit_dbfs / 20.0)
    peak = mixed.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(peak), torch.full_like(peak, limit) / peak)
    mixed = mixed * scale
    return {"waveform": mixed, "sample_rate": target_rate}


def trim_av_output(frames: torch.Tensor, start_seconds: float, duration_seconds: float, audio=None, fps: float = 24.0):
    if frames.ndim != 4:
        raise ValueError(f"frames must be IMAGE [N,H,W,C], got {tuple(frames.shape)}")
    if start_seconds < 0 or duration_seconds <= 0 or fps <= 0:
        raise ValueError("trim start must be nonnegative; duration and fps must be positive")
    start_frame = round(start_seconds * fps)
    frame_count = max(1, round(duration_seconds * fps))
    end_frame = start_frame + frame_count
    if end_frame > frames.shape[0]:
        raise ValueError(
            f"Requested output frames [{start_frame}:{end_frame}] exceed decoded frame count {frames.shape[0]}"
        )
    trimmed_frames = frames[start_frame:end_frame]

    trimmed_audio = None
    if audio is not None:
        waveform, sample_rate = validate_audio(audio)
        start_sample = round(start_seconds * sample_rate)
        sample_count = round(duration_seconds * sample_rate)
        sliced = waveform[..., start_sample : start_sample + sample_count]
        if sliced.shape[-1] < sample_count:
            sliced = torch_functional.pad(sliced, (0, sample_count - sliced.shape[-1]))
        trimmed_audio = {"waveform": sliced, "sample_rate": sample_rate}

    report = json.dumps(
        {
            "start_seconds": start_seconds,
            "requested_duration_seconds": duration_seconds,
            "fps": fps,
            "start_frame": start_frame,
            "frame_count": frame_count,
            "actual_video_duration_seconds": frame_count / fps,
        },
        ensure_ascii=False,
        indent=2,
    )
    return trimmed_frames, trimmed_audio, report
