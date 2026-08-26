from __future__ import annotations

import json
import re


MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)
OFFICIAL_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)


def canonicalize_media_tags(prompt: str, task_type: str | None = None) -> str:
    def replacement(match: re.Match) -> str:
        media_type = (match.group(1) or match.group(3)).lower()
        ordinal = int(match.group(2) or match.group(4))
        if (task_type or "").lower() == "fl2va" and media_type in {"image", "picture"}:
            return f"picture {ordinal}"
        official_type = "Picture" if media_type in {"image", "picture"} else media_type.title()
        return f"<{official_type} {ordinal}>"

    return MEDIA_TAG_RE.sub(replacement, prompt or "")


def prepare_prompt(
    prompt: str,
    counts: dict[str, int],
    strict: bool = True,
    task_type: str | None = None,
    valid_ordinals: dict[str, set[int]] | None = None,
) -> tuple[str, list[str]]:
    normalized = canonicalize_media_tags(prompt, task_type)

    warnings: list[str] = []
    limits = {
        "picture": int(counts.get("pictures", 0)),
        "video": int(counts.get("videos", 0)),
        "audio": int(counts.get("audios", 0)),
    }
    for match in OFFICIAL_TAG_RE.finditer(normalized):
        media_type, ordinal = match.group(1).lower(), int(match.group(2))
        valid = valid_ordinals.get(media_type) if valid_ordinals else None
        missing = ordinal not in valid if valid is not None else ordinal < 1 or ordinal > limits[media_type]
        if missing:
            warnings.append(
                f"{match.group(0)} is not connected; available {media_type} ordinals are "
                f"{sorted(valid) if valid is not None else list(range(1, limits[media_type] + 1))}"
            )
    if strict and warnings:
        raise ValueError("当前提示词引用了不存在的素材标签，请引用正确的标签后重试")
    return normalized, warnings


def pack_media_tag_ordinals(prompt: str, ordinal_maps: dict[str, dict[int, int]]) -> str:
    """Map stable UI slot ordinals to the dense order used by MiniMax's tokenizer."""
    def replacement(match: re.Match) -> str:
        media_type = match.group(1).lower()
        ordinal = int(match.group(2))
        packed = ordinal_maps.get(media_type, {}).get(ordinal, ordinal)
        return f"<{match.group(1).title()} {packed}>"

    return OFFICIAL_TAG_RE.sub(replacement, prompt or "")


def media_map_json(pictures, videos, audios) -> str:
    def mapped(values):
        if isinstance(values, dict):
            return {str(index): label for index, label in sorted(values.items())}
        return {str(index + 1): label for index, label in enumerate(values)}

    return json.dumps(
        {
            "pictures": mapped(pictures),
            "videos": mapped(videos),
            "audios": mapped(audios),
        },
        ensure_ascii=False,
        indent=2,
    )
