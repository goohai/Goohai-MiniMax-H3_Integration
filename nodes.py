from __future__ import annotations

import math
import os
import json

import folder_paths
import nodes
import torch
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest._input_impl import VideoFromFile
from comfy_extras import nodes_audio

from .conditioning import build_conditioning


NODE_CATEGORY = "Goohai/MiniMax H3 Integration"
MAX_RESOLUTION = 16384
ASPECTS = {
    "adaptive": None,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1.0,
    "21:9": 21 / 9,
}
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _files(content_type: str):
    return sorted(folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), [content_type]))


def _image_files():
    return _files("image")


def _video_files():
    return _files("video")


def _audio_files():
    return _files("audio")


def _uploaded_media(name, tooltip=None, hidden=False):
    """Store a custom-uploaded filename without Combo enum validation.

    The DOM panel owns the upload control.  A fixed upload Combo cannot
    represent an unused slot: ComfyUI validates every submitted Combo value,
    so the old ``(none)`` sentinel made the last empty media slots invalid.
    Empty strings are valid optional STRING values and are converted to None
    by the existing file loaders.
    """
    return io.String.Input(
        name,
        default="",
        optional=True,
        tooltip=tooltip,
        extra_dict={"hidden": True} if hidden else None,
    )


def _hidden_combo(name, options, default=None):
    return io.Combo.Input(
        name,
        options=options,
        default=default,
        extra_dict={"hidden": True},
    )


def _hidden_boolean(name, default):
    return io.Boolean.Input(name, default=default, extra_dict={"hidden": True})


def _hidden_float(name, default, minimum, maximum, step, round_value=None):
    return io.Float.Input(
        name,
        default=default,
        min=minimum,
        max=maximum,
        step=step,
        round=round_value,
        extra_dict={"hidden": True},
    )


def _hidden_int(name, default, minimum, maximum, step=1):
    return io.Int.Input(
        name,
        default=default,
        min=minimum,
        max=maximum,
        step=step,
        extra_dict={"hidden": True},
    )


def _coerce_int(value, default=0, minimum=None, maximum=None):
    """Accept legacy serialized widget values without leaking them downstream."""
    if value is None or value == "" or value == "(none)":
        result = int(default)
    else:
        try:
            result = int(float(value))
        except (TypeError, ValueError):
            result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _mode_model_number(mode):
    return 1 if mode == "all_reference" else 0


def _restore_ui_state(gh_state_json, main_mode, prompt, media_values):
    """Recover DOM-owned values from the workflow's serialized state."""
    if not gh_state_json or gh_state_json == "(none)":
        return main_mode, prompt, media_values
    try:
        state = json.loads(gh_state_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return main_mode, prompt, media_values
    if not isinstance(state, dict):
        return main_mode, prompt, media_values

    restored_mode = state.get("mode")
    if restored_mode in {"text_keyframes", "all_reference"}:
        main_mode = restored_mode

    prompts = state.get("prompts")
    if isinstance(prompts, dict) and isinstance(prompts.get(main_mode), str):
        prompt = prompts[main_mode]
    elif isinstance(state.get("prompt"), str):
        prompt = state["prompt"]

    # When the media list exists it is the source of truth, including removals.
    # This prevents stale hidden widget values from reviving deleted uploads.
    serialized_media = state.get("media")
    if isinstance(serialized_media, list):
        restored_media = {}
        for item in serialized_media:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            slot, entry = item
            if slot not in media_values or not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name and name != "(none)":
                restored_media[slot] = name
        media_values = {name: restored_media.get(name, "") for name in media_values}

    return main_mode, prompt, media_values


def _serialized_first_visual_name(gh_state_json, mode):
    if not gh_state_json or gh_state_json == "(none)":
        return None
    try:
        state = json.loads(gh_state_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    media = dict(state.get("media", [])) if isinstance(state, dict) else {}
    if mode == "text_keyframes":
        ordered_slots = ["first_frame", "last_frame"]
    else:
        # Ref2VA presents videos before pictures, regardless of upload time.
        ordered_slots = [
            *(f"ref_video_{i}" for i in range(1, 4)),
            *(f"ref_image_{i}" for i in range(1, 10)),
        ]
    for slot in ordered_slots:
        entry = media.get(slot)
        if not isinstance(entry, dict) or entry.get("kind") not in {"image", "video"}:
            continue
        name = entry.get("name")
        if isinstance(name, str) and name and name != "(none)":
            return name
    return None


def _serialized_muted_video_slots(gh_state_json):
    if not gh_state_json or gh_state_json == "(none)":
        return set()
    try:
        state = json.loads(gh_state_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    media = state.get("media", []) if isinstance(state, dict) else []
    return {
        slot for slot, entry in media
        if isinstance(slot, str) and slot.startswith("ref_video_")
        and isinstance(entry, dict) and bool(entry.get("muted"))
    }


def _default_model(options, preferred):
    return preferred if preferred in options else (options[0] if options else None)


def _round32(value: float) -> int:
    return max(32, int(value / 32 + 0.5) * 32)


def calculate_canvas(aspect: str, megapixels: float, source_width=None, source_height=None):
    ratio = ASPECTS.get(aspect, ASPECTS["16:9"])
    if ratio is None:
        try:
            ratio = float(source_width) / float(source_height)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = ASPECTS["16:9"]
    area = max(0.05, float(megapixels)) * 1024 * 1024
    # Round both axes independently from the requested aspect ratio. Using
    # area / rounded_width makes even 1:1 drift after the second rounding.
    width = _round32((area * ratio) ** 0.5)
    height = _round32((area / ratio) ** 0.5)
    if ratio == 1.0:
        height = width
    return min(MAX_RESOLUTION, width), min(MAX_RESOLUTION, height)


def calculate_length(duration_seconds: float) -> int:
    requested_frames = max(5, round(float(duration_seconds) * 24))
    return requested_frames + (5 - (requested_frames % 17)) % 17


def _load_image_file(value):
    if not value or value == "(none)":
        return None
    image, _mask = nodes.LoadImage().load_image(value)
    return image


def _load_audio_file(value):
    if not value or value == "(none)":
        return None
    return nodes_audio.LoadAudio.load(value)[0]


def _trim_audio_to_duration(audio, duration_seconds):
    """Return a same-rate AUDIO value exactly matching the requested duration."""
    if audio is None:
        return None
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    sample_count = max(1, round(float(duration_seconds) * sample_rate))
    trimmed = waveform[..., :sample_count]
    if trimmed.shape[-1] < sample_count:
        trimmed = torch.nn.functional.pad(trimmed, (0, sample_count - trimmed.shape[-1]))
    return {"waveform": trimmed, "sample_rate": sample_rate}


def _sample_video_frames(frames, source_fps, target_frames):
    """Take the first target window at 24fps, padding short sources at the end."""
    if frames is None or frames.shape[0] <= 1:
        return frames
    source_fps = float(source_fps or 0)
    requested = max(1, int(target_frames))
    # Keep the old helper contract for direct callers that pass a legacy FPS
    # value. The node itself passes the calculated target frame count.
    if requested <= 30:
        target_fps = requested
        duration = min(frames.shape[0] / source_fps, 360 / 24) if source_fps > 0 else 0
        target_frames = min(360, max(48, round(duration * target_fps)))
        if source_fps <= 0:
            sampled = frames[:target_frames]
        else:
            source_end = min(frames.shape[0] - 1, max(0, math.ceil(duration * source_fps) - 1))
            indices = torch.linspace(0, source_end, target_frames).round().long()
            return frames[indices]
    else:
        target_frames = requested
    if source_fps <= 0:
        sampled = frames[:target_frames]
    else:
        source_indices = torch.floor(torch.arange(target_frames, dtype=torch.float32) * source_fps / 24).long()
        source_indices = source_indices.clamp(max=frames.shape[0] - 1)
        sampled = frames[source_indices]
    if sampled.shape[0] < target_frames:
        sampled = torch.cat([sampled, sampled[-1:].repeat(target_frames - sampled.shape[0], 1, 1, 1)], dim=0)
    return sampled[:target_frames]


def _load_video_frames(value, target_frames):
    if not value or value == "(none)":
        return None, None
    path = folder_paths.get_annotated_filepath(value)
    components = VideoFromFile(path).get_components()
    frames = components.images
    frames = _sample_video_frames(frames, components.frame_rate, target_frames)
    return frames, components.audio


class MiniMaxH3IntegrationGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        image_files = _image_files()
        video_files = _video_files()
        audio_files = _audio_files()
        return io.Schema(
            node_id="MiniMaxH3IntegrationGH",
            category=NODE_CATEGORY,
            description="Integrated MiniMax H3 conditioning input. The diffusion model remains external.",
            inputs=[
                _hidden_combo("main_mode", ["text_keyframes", "all_reference"], "text_keyframes"),
                io.Combo.Input(
                    "clip_name",
                    options=folder_paths.get_filename_list("text_encoders"),
                    default=_default_model(folder_paths.get_filename_list("text_encoders"), DEFAULT_CLIP),
                ),
                io.Combo.Input(
                    "video_vae_name",
                    options=folder_paths.get_filename_list("vae"),
                    default=_default_model(folder_paths.get_filename_list("vae"), DEFAULT_VIDEO_VAE),
                ),
                io.Combo.Input(
                    "audio_vae_name",
                    options=folder_paths.get_filename_list("vae"),
                    default=_default_model(folder_paths.get_filename_list("vae"), DEFAULT_AUDIO_VAE),
                ),
                io.Combo.Input("aspect", options=list(ASPECTS), default="16:9"),
                io.Float.Input("megapixels", default=0.5, min=0.2, max=2.0, step=0.1, round=0.1),
                io.Int.Input(
                    "duration_seconds",
                    default=5,
                    min=2,
                    max=15,
                    step=1,
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="", extra_dict={"hidden": True}),
                _uploaded_media("first_frame", tooltip="Optional first frame", hidden=True),
                _uploaded_media("last_frame", tooltip="Optional last frame", hidden=True),
                _uploaded_media("hybrid_audio", tooltip="Optional audio for keyframe Hybrid", hidden=True),
                _uploaded_media("ref_image_1", hidden=True),
                _uploaded_media("ref_image_2", hidden=True),
                _uploaded_media("ref_image_3", hidden=True),
                _uploaded_media("ref_image_4", hidden=True),
                _uploaded_media("ref_image_5", hidden=True),
                _uploaded_media("ref_image_6", hidden=True),
                _uploaded_media("ref_image_7", hidden=True),
                _uploaded_media("ref_image_8", hidden=True),
                _uploaded_media("ref_image_9", hidden=True),
                _uploaded_media("ref_video_1", hidden=True),
                _uploaded_media("ref_video_2", hidden=True),
                _uploaded_media("ref_video_3", hidden=True),
                _uploaded_media("ref_audio_1", hidden=True),
                _uploaded_media("ref_audio_2", hidden=True),
                _uploaded_media("ref_audio_3", hidden=True),
                _hidden_combo("task_type", ["auto", "t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid"], "auto"),
                _hidden_combo("audio_mode", ["native", "lock_source", "reference_only", "remix_source"], "lock_source"),
                _hidden_float("audio_denoise_strength", 0.0, 0.0, 1.0, 0.01, 0.01),
                # Match T8: 1 selects the first audio; 0 disables remapping.
                _hidden_int("drive_audio_ordinal", 1, 0, 6),
                _hidden_boolean("strict_prompt_tags", True),
                _hidden_combo("ref_image_size", ["match", "max"], "match"),
                # Keep this last so adding persistence does not shift legacy
                # widgets_values positions for any existing input.
                io.String.Input("gh_state_json", default="", optional=True, extra_dict={"hidden": True}),
            ],
            # A private socket type keeps the adapter as the only compatible
            # expansion node when dragging from this output.
            outputs=[io.Custom("MiniMax").Output(display_name=">")],
        )

    @classmethod
    def execute(cls, main_mode, aspect, megapixels, duration_seconds, clip_name, video_vae_name, audio_vae_name,
                prompt, first_frame, last_frame, hybrid_audio, ref_image_1, ref_image_2, ref_image_3,
                ref_image_4, ref_image_5, ref_image_6, ref_image_7, ref_image_8, ref_image_9,
                ref_video_1, ref_video_2, ref_video_3, ref_audio_1, ref_audio_2, ref_audio_3,
                task_type, audio_mode, audio_denoise_strength,
                drive_audio_ordinal, strict_prompt_tags, ref_image_size,
                gh_state_json=""):
        # ComfyUI can replay an older hidden-widget snapshot as the literal
        # string "(none)". T8 treats this control as an int, with 0 disabling
        # remapping, so normalize it before calling the shared conditioning.
        drive_audio_ordinal = _coerce_int(drive_audio_ordinal, default=1, minimum=0, maximum=6)
        media_values = {
            "first_frame": first_frame, "last_frame": last_frame, "hybrid_audio": hybrid_audio,
            "ref_image_1": ref_image_1, "ref_image_2": ref_image_2, "ref_image_3": ref_image_3,
            "ref_image_4": ref_image_4, "ref_image_5": ref_image_5, "ref_image_6": ref_image_6,
            "ref_image_7": ref_image_7, "ref_image_8": ref_image_8, "ref_image_9": ref_image_9,
            "ref_video_1": ref_video_1, "ref_video_2": ref_video_2, "ref_video_3": ref_video_3,
            "ref_audio_1": ref_audio_1, "ref_audio_2": ref_audio_2, "ref_audio_3": ref_audio_3,
        }
        main_mode, prompt, media_values = _restore_ui_state(
            gh_state_json, main_mode, prompt, media_values
        )
        first_frame = media_values["first_frame"]
        last_frame = media_values["last_frame"]
        hybrid_audio = media_values["hybrid_audio"]
        ref_image_1, ref_image_2, ref_image_3 = (media_values[f"ref_image_{i}"] for i in range(1, 4))
        ref_image_4, ref_image_5, ref_image_6 = (media_values[f"ref_image_{i}"] for i in range(4, 7))
        ref_image_7, ref_image_8, ref_image_9 = (media_values[f"ref_image_{i}"] for i in range(7, 10))
        ref_video_1, ref_video_2, ref_video_3 = (media_values[f"ref_video_{i}"] for i in range(1, 4))
        ref_audio_1, ref_audio_2, ref_audio_3 = (media_values[f"ref_audio_{i}"] for i in range(1, 4))
        first = _load_image_file(first_frame)
        last = _load_image_file(last_frame)
        hybrid = _load_audio_file(hybrid_audio)
        # The dedicated Hybrid upload is the GH equivalent of T8's first
        # autogrow ref_audio input. Keep it as a reference and also provide it
        # as the internal drive track so Hybrid works without an <Audio N> tag.
        refs = [_load_image_file(v) for v in [ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6, ref_image_7, ref_image_8, ref_image_9]]
        refs = {f"ref_image_{i}": v for i, v in enumerate(refs, 1) if v is not None}
        ref_videos, video_audio = {}, {}
        muted_video_slots = _serialized_muted_video_slots(gh_state_json)
        for i, value in enumerate([ref_video_1, ref_video_2, ref_video_3], 1):
            frames, soundtrack = _load_video_frames(value, calculate_length(duration_seconds))
            if frames is not None:
                ref_videos[f"ref_video_{i}"] = frames
                if soundtrack is not None and f"ref_video_{i}" not in muted_video_slots:
                    video_audio[f"ref_video_audio_{i}"] = soundtrack
        audios = [_load_audio_file(v) for v in [ref_audio_1, ref_audio_2, ref_audio_3]]
        audios = {f"ref_audio_{i}": v for i, v in enumerate(audios, 1) if v is not None}
        ordered_drive_audios = [
            video_audio[f"ref_video_audio_{i}"]
            for i in range(1, 4)
            if f"ref_video_audio_{i}" in video_audio
        ] + list(audios.values())
        selected_drive_audio = (
            ordered_drive_audios[drive_audio_ordinal - 1]
            if main_mode == "all_reference" and 0 < drive_audio_ordinal <= len(ordered_drive_audios)
            else None
        )
        internal_drive_audio = hybrid if main_mode == "text_keyframes" and hybrid is not None else selected_drive_audio

        visual_source = None
        serialized_first_visual = _serialized_first_visual_name(gh_state_json, main_mode)
        if main_mode == "text_keyframes":
            if serialized_first_visual and first_frame == serialized_first_visual:
                visual_source = first
            elif serialized_first_visual and last_frame == serialized_first_visual:
                visual_source = last
            else:
                visual_source = first if first is not None else last
        else:
            if serialized_first_visual:
                for slot, filename in ((f"ref_video_{i}", value) for i, value in enumerate(
                    [ref_video_1, ref_video_2, ref_video_3], 1
                )):
                    if filename == serialized_first_visual:
                        visual_source = ref_videos.get(slot)
                        break
                if visual_source is None:
                    for slot, filename in ((f"ref_image_{i}", value) for i, value in enumerate(
                        [ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6, ref_image_7, ref_image_8, ref_image_9], 1
                    )):
                        if filename == serialized_first_visual:
                            visual_source = refs.get(slot)
                            break
            ordered_visual_values = [
                ref_videos.get(f"ref_video_{i}") for i in range(1, 4)
            ] + [
                refs.get(f"ref_image_{i}") for i in range(1, 10)
            ]
            for value in (() if visual_source is not None else ordered_visual_values):
                if value is not None:
                    visual_source = value
                    break
        if visual_source is not None:
            source_height, source_width = int(visual_source.shape[1]), int(visual_source.shape[2])
        else:
            source_width = source_height = None
        width, height = calculate_canvas(aspect, megapixels, source_width, source_height)
        length = calculate_length(duration_seconds)
        clip = nodes.CLIPLoader().load_clip(clip_name, "minimax")[0]
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0]
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0]

        if main_mode == "text_keyframes":
            refs, ref_videos, video_audio = {}, {}, {}
            audios = {"ref_audio_1": hybrid} if hybrid is not None else {}
        elif main_mode == "all_reference":
            # Keep keyframe uploads in the workflow state so switching back
            # does not lose them, but do not feed them into Ref2VA execution.
            first, last, hybrid = None, None, None
        elif hybrid is not None:
            audios["ref_audio_1"] = hybrid

        # Only keyframe Hybrid has an internal drive track. Reference-only
        # mode must always let the model generate audio.
        if internal_drive_audio is None:
            audio_mode = "native"
        is_original_audio = audio_mode == "lock_source"
        final_audio = _trim_audio_to_duration(internal_drive_audio, duration_seconds)

        result = build_conditioning(
            clip, video_vae, audio_vae, prompt, width, height, length,
            "auto", audio_mode, audio_denoise_strength, True,
            strict_prompt_tags, ref_image_size, internal_drive_audio, final_audio,
            first, last, refs, ref_videos, video_audio, audios,
        )
        bundle = {
            "positive": result[0], "av_latent": result[1], "mux_audio": result[2],
            "conditioned_prompt": result[3], "media_map_json": result[4], "report": result[5],
            "video_vae": video_vae, "audio_vae": audio_vae,
            "width": width, "height": height, "length": length,
            "duration_seconds": duration_seconds, "resolved_mode": main_mode,
            "is_original_audio": is_original_audio,
        }
        return io.NodeOutput(bundle)


class MiniMaxH3IntegrationAdapterGH(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3IntegrationAdapterGH",
            category=NODE_CATEGORY,
            inputs=[io.Custom("MiniMax").Input("integration", display_name="<")],
            outputs=[
                io.Conditioning.Output("positive"), io.Latent.Output("av_latent"),
                io.Vae.Output("video_vae"), io.Vae.Output("audio_vae"), io.Int.Output("models number"),
                io.Audio.Output("mux_audio"), io.Boolean.Output("is_original_audio"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("media_map_json"), io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, integration):
        return io.NodeOutput(
            integration["positive"], integration["av_latent"],
            integration["video_vae"], integration["audio_vae"],
            _mode_model_number(integration.get("resolved_mode")),
            integration["mux_audio"], bool(integration.get("is_original_audio", False)),
            integration["conditioned_prompt"],
            integration["media_map_json"], integration["report"],
        )


class MiniMaxH3IntegrationExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3IntegrationGH, MiniMaxH3IntegrationAdapterGH]


def comfy_entrypoint():
    return MiniMaxH3IntegrationExtension()
