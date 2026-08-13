import sys
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[2]
sys.path.insert(0, str(COMFY))

from gh_h3_test.nodes import (  # noqa: E402
    _coerce_int,
    _mode_model_number,
    _restore_ui_state,
    _serialized_first_visual_name,
    _sample_video_frames,
    _trim_audio_to_duration,
    calculate_canvas,
    calculate_length,
)
from gh_h3_test.conditioning import (  # noqa: E402
    HYBRID_KEYFRAME_SENTINEL,
    KEYFRAME_REF_CONCAT,
    REFS_OVERWRITE,
    classify_cond_video_latents_policy,
    extra_conds_source_from_file,
    resolve_task_type,
)
from gh_h3_test.prompt_tags import canonicalize_media_tags, prepare_prompt  # noqa: E402


def test_canvas_is_32_aligned_and_matches_requested_area():
    width, height = calculate_canvas("16:9 (Widescreen)", 2.0)
    assert (width, height) == (1920, 1088)
    assert width % 32 == 0
    assert height % 32 == 0


def test_square_canvas_preserves_square_ratio_after_alignment():
    assert calculate_canvas("1:1", 2.0) == (1440, 1440)


def test_adaptive_canvas_uses_source_ratio_and_32_alignment():
    width, height = calculate_canvas("adaptive", 2.0, 21, 9)
    assert (width, height) == (2208, 960)
    assert width % 32 == 0 and height % 32 == 0
    assert calculate_canvas("adaptive", 2.0) == calculate_canvas("16:9 (Widescreen)", 2.0)


def test_adaptive_ratio_source_follows_uploaded_visual_order():
    import json

    state = {"mode": "all_reference", "media": [
        ["ref_image_1", {"name": "style.png", "kind": "image"}],
        ["ref_video_1", {"name": "source.mp4", "kind": "video"}],
    ]}
    assert _serialized_first_visual_name(json.dumps(state), "all_reference") == "source.mp4"


def test_adaptive_ratio_source_uses_video_slots_before_image_slots():
    import json

    state = {"mode": "all_reference", "media": [
        ["ref_image_1", {"name": "style.png", "kind": "image"}],
        ["ref_video_2", {"name": "later.mp4", "kind": "video"}],
    ]}
    assert _serialized_first_visual_name(json.dumps(state), "all_reference") == "later.mp4"


def test_forced_video_fps_samples_source_timeline_and_keeps_h3_minimum():
    frames = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    sampled = _sample_video_frames(frames, 2, 1)
    assert sampled.shape[0] == 48
    assert sampled[0, 0, 0, 0].item() == 0
    assert sampled[-1, 0, 0, 0].item() == 9
    assert sampled[:, 0, 0, 0].tolist().count(9) < 4


def test_low_forced_fps_raises_effective_rate_and_covers_short_video():
    frames = torch.arange(26, dtype=torch.float32).reshape(26, 1, 1, 1)
    sampled = _sample_video_frames(frames, 24, 1)
    assert sampled.shape[0] == 48
    assert sampled[0, 0, 0, 0].item() == 0
    assert sampled[-1, 0, 0, 0].item() == 25
    assert sampled[:, 0, 0, 0].tolist().count(25) < 4


def test_reference_video_is_capped_at_fifteen_seconds():
    frames = torch.arange(400, dtype=torch.float32).reshape(400, 1, 1, 1)
    capped = _sample_video_frames(frames, 24, 24)
    assert capped.shape[0] == 360
    assert capped[-1, 0, 0, 0].item() == 359


def test_sixty_fps_reference_keeps_original_speed_at_twenty_four_fps():
    frames = torch.arange(300, dtype=torch.float32).reshape(300, 1, 1, 1)
    sampled = _sample_video_frames(frames, 60, 120)
    assert sampled.shape[0] == 120
    assert sampled[0, 0, 0, 0].item() == 0
    assert sampled[-1, 0, 0, 0].item() == 297


def test_aligned_reference_continues_at_source_speed_when_source_is_long_enough():
    frames = torch.arange(320, dtype=torch.float32).reshape(320, 1, 1, 1)
    sampled = _sample_video_frames(frames, 60, 124)
    assert sampled.shape[0] == 124
    assert sampled[119, 0, 0, 0].item() == 297
    assert sampled[-1, 0, 0, 0].item() == 307


def test_twelve_fps_reference_duplicates_frames_across_timeline():
    frames = torch.arange(60, dtype=torch.float32).reshape(60, 1, 1, 1)
    sampled = _sample_video_frames(frames, 12, 120)
    assert sampled.shape[0] == 120
    assert sampled[:6, 0, 0, 0].tolist() == [0, 0, 1, 1, 2, 2]
    assert sampled[-1, 0, 0, 0].item() == 59


def test_cond_video_policy_detects_refs_overwrite_and_concat():
    overwrite = '''
def extra_conds(self, **kwargs):
    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
'''
    concat = '''
def extra_conds(self, **kwargs):
    payload["cond_video_latents"] = (
        [kf["latent"] for kf in keyframes] + [r["latent"] for r in refs if "latent" in r]
    )
'''
    assert classify_cond_video_latents_policy(overwrite) == REFS_OVERWRITE
    assert classify_cond_video_latents_policy(concat) == KEYFRAME_REF_CONCAT


def test_rh_comfyui_minimax_extra_conds_still_overwrites_with_refs():
    model_base = Path("/root/custom_nodes/.RH/ComfyUI/comfy/model_base.py")
    if not model_base.exists():
        pytest.skip("RH ComfyUI is not present")
    source = extra_conds_source_from_file(model_base)
    assert classify_cond_video_latents_policy(source) == REFS_OVERWRITE


def test_auto_task_resolution():
    assert resolve_task_type("auto", None, None, False) == "t2va"
    assert resolve_task_type("auto", object(), None, False) == "i2va"
    assert resolve_task_type("auto", None, object(), False) == "l2va"
    assert resolve_task_type("auto", object(), object(), False) == "fl2va"
    assert resolve_task_type("auto", object(), object(), True) == "hybrid"
    assert resolve_task_type("auto", None, None, True) == "ref2va"


def test_fl2va_picture_labels_do_not_use_angle_brackets():
    assert canonicalize_media_tags("<Picture 1> to picture 2", "fl2va") == "picture 1 to picture 2"
    assert canonicalize_media_tags("picture 1", "i2va") == "<Picture 1>"
    assert canonicalize_media_tags("Picture 1", "l2va") == "<Picture 1>"
    assert canonicalize_media_tags("Picture 1", "hybrid") == "<Picture 1>"


def test_drive_audio_is_trimmed_or_padded_to_requested_duration():
    audio = {"waveform": torch.ones((1, 1, 3)), "sample_rate": 2}
    result = _trim_audio_to_duration(audio, 2.0)
    assert result["waveform"].shape == (1, 1, 4)
    assert result["waveform"][..., -1].item() == 0

    audio = {"waveform": torch.ones((1, 1, 6)), "sample_rate": 2}
    result = _trim_audio_to_duration(audio, 2.0)
    assert result["waveform"].shape == (1, 1, 4)


def test_invalid_media_tag_uses_user_facing_error_message():
    with pytest.raises(ValueError, match="当前提示词引用了不存在的素材标签，请引用正确的标签后重试"):
        prepare_prompt("<Picture 2>", {"pictures": 1}, strict=True, task_type="i2va")


def test_serialized_ui_state_restores_mode_prompt_and_keyframes():
    state = {
        "mode": "text_keyframes",
        "media": [
            ["first_frame", {"name": "first.png", "kind": "image"}],
            ["last_frame", {"name": "last.png", "kind": "image"}],
        ],
        "prompts": {
            "text_keyframes": "keyframe prompt",
            "all_reference": "reference prompt",
        },
    }
    media = {"first_frame": "", "last_frame": "", "ref_image_1": "stale.png"}
    mode, prompt, restored = _restore_ui_state(
        __import__("json").dumps(state), "all_reference", "", media
    )
    assert mode == "text_keyframes"
    assert prompt == "keyframe prompt"
    assert restored == {
        "first_frame": "first.png",
        "last_frame": "last.png",
        "ref_image_1": "",
    }


def test_hybrid_keyframe_reference_audio_is_packed_as_first_audio_reference():
    from gh_h3_test.conditioning import build_conditioning

    class FakeClip:
        def __init__(self):
            self.tokenize_calls = []
        def tokenize(self, prompt, **kwargs):
            self.tokenize_calls.append((prompt, kwargs))
            return {"prompt": prompt, "kwargs": kwargs}
        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros((1, 4, 8)), {"tokens": tokens}]]

    class FakeVideoVAE:
        def encode(self, images):
            frames, height, width = images.shape[:3]
            latent_t = 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2
            return torch.zeros((1, 24, latent_t, max(1, height // 16), max(1, width // 16)))

    class FakeAudioVAE:
        audio_sample_rate = 32000
        audio_sample_rate_output = 32000
        def encode(self, waveform_last):
            latent_t = max(1, math.ceil(waveform_last.shape[1] / 800))
            return torch.full((1, 32, 2, latent_t), 0.25)

    def make_audio():
        return {"waveform": torch.full((1, 1, 160000), 0.1), "sample_rate": 32000}

    clip = FakeClip()
    args = {
        "clip": clip,
        "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(),
        "prompt": "A person sings",
        "width": 128,
        "height": 128,
        "length": 124,
        "task_type": "auto",
        "audio_mode": "native",
        "first_frame": torch.zeros((1, 128, 128, 3)),
        "ref_audios": {"ref_audio_1": make_audio()},
    }
    conditioning, *_ = build_conditioning(**args)
    metadata = conditioning[0][1]
    assert metadata["minimax_refs"][0]["kind"] == HYBRID_KEYFRAME_SENTINEL
    assert metadata["minimax_refs"][1]["kind"] == "audio"
    assert [item["type"] for item in clip.tokenize_calls[0][1]["minimax_ref_items"]] == ["image", "audio"]


def test_first_last_frames_use_official_keyframe_conditioning_path():
    from gh_h3_test.conditioning import build_conditioning

    class FakeClip:
        def __init__(self):
            self.tokenize_calls = []
        def tokenize(self, prompt, **kwargs):
            self.tokenize_calls.append((prompt, kwargs))
            return {"prompt": prompt, "kwargs": kwargs}
        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros((1, 4, 8)), {"tokens": tokens}]]

    class FakeVideoVAE:
        def encode(self, images):
            return torch.full((1, 24, 1, 8, 8), float(len(images)))

    class FakeAudioVAE:
        audio_sample_rate = 32000
        audio_sample_rate_output = 32000

    clip = FakeClip()
    conditioning, *_ = build_conditioning(
        clip=clip,
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt="keyframed scene",
        width=128,
        height=128,
        length=124,
        task_type="auto",
        audio_mode="native",
        first_frame=torch.zeros((1, 128, 128, 3)),
        last_frame=torch.ones((1, 128, 128, 3)),
    )
    metadata = conditioning[0][1]
    keyframes = metadata["minimax_keyframes"]
    assert [item["resolved_frame_index"] for item in keyframes] == [0, 123]
    assert metadata["minimax_frame_count"] == 124
    assert "minimax_refs" not in metadata
    assert len(clip.tokenize_calls[0][1]["images"]) == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [("(none)", 1), (None, 1), ("2", 2), (2.8, 2), (0, 0), (99, 9)],
)
def test_legacy_primary_audio_ordinal_is_a_valid_t8_integer(value, expected):
    assert _coerce_int(value, default=1, minimum=0, maximum=9) == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("text_keyframes", 0), ("all_reference", 1), ("renamed_button_label", 0)],
)
def test_adapter_model_number_uses_internal_mode(mode, expected):
    assert _mode_model_number(mode) == expected


@pytest.mark.parametrize(
    ("seconds", "frames"),
    [(0.04, 5), (5.0, 124), (6.0, 158), (15.0, 362)],
)
def test_duration_uses_h3_frame_grid(seconds, frames):
    assert calculate_length(seconds) == frames
