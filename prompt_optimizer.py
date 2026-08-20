from __future__ import annotations

import asyncio
import base64
from difflib import SequenceMatcher
import gc
import json
import os
import platform
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request

import aiohttp
from aiohttp import web
import folder_paths
from server import PromptServer


DEFAULT_CONFIG = {
    "mode": "api",
    "provider": "runninghub",
    "api_url": "https://www.runninghub.cn/openapi/v2",
    "api_key": "",
    "model": "openai/gpt-5.6-sol",
    "protocol": "runninghub",
    "read_media": True,
    "output_language": "中文",
    "local_model": "",
    "local_mmproj": "",
    "local_device": "cuda",
    "auto_optimize": False,
}
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "gpt-4.1-mini", "openai"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash", "gemini"),
    "openrouter": ("https://openrouter.ai/api/v1", "google/gemini-2.5-flash", "openai"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max", "openai"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-VL-72B-Instruct", "openai"),
    "runninghub": ("https://www.runninghub.cn/openapi/v2", "openai/gpt-5.6-sol", "runninghub"),
}
RUNNINGHUB_APP_ID = "2089252473927196673"
RUNNINGHUB_MODELS = (
    "openai/gpt-5.6-sol", "openai/gpt-5.6-sol-saver", "openai/gpt-5.6-terra",
    "openai/gpt-5.6-terra-saver", "openai/gpt-5.5", "openai/gpt-5.5-saver",
    "openai/gpt-5.6-luna", "openai/gpt-5.6-luna-saver", "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash", "openai/gpt-5.5-pro", "anthropic/claude-fable-5",
    "openai/gpt-5.4-pro", "anthropic/claude-opus-5", "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-4.7", "glm-5.2", "anthropic/claude-opus-4.6", "openai/gpt-5.4",
    "openai/gpt-5.3-codex", "glm-5.1", "glm-5-turbo", "qwen/qwen3.8-max",
    "anthropic/claude-sonnet-4.6", "glm-5", "anthropic/claude-sonnet-5",
    "qwen/qwen3.7-max", "glm-5v-turbo", "qwen/qwen3.7-plus", "deepseek/deepseek-v4-pro",
    "xai/grok-4.6", "xai/grok-4.5", "xai/grok-4.3", "qwen/qwen3.6-plus",
    "google/gemini-3.1-pro-preview", "bytedance/doubao-seed-evolving",
    "bytedance/doubao-seed-2.1-pro", "anthropic/claude-sonnet-4.5",
    "bytedance/doubao-seed-2.1-turbo", "anthropic/claude-opus-4.5",
    "bytedance/doubao-seed-2.0-pro", "bytedance/doubao-seed-2.0-code",
    "deepseek/deepseek-v4-flash", "qwen/qwen3.6-flash", "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano", "google/gemini-3-flash-preview", "google/gemini-2.5-flash",
    "bytedance/doubao-seed-2.0-lite", "bytedance/doubao-seed-2.0-mini",
    "minimax/minimax-m2.7", "anthropic/claude-haiku-4.5", "qwen/qwen3.6-max-preview",
    "anthropic/claude-haiku-4.5-saver", "anthropic/claude-opus-4.6-saver",
    "anthropic/claude-opus-4.7-saver", "anthropic/claude-opus-4.8-saver",
    "anthropic/claude-sonnet-4.6-saver", "google/gemini-2.5-pro",
    "google/gemini-3.5-flash-lite", "google/gemini-3.6-flash",
)
RUNNINGHUB_DETAIL_URL = f"https://www.runninghub.cn/call-api/api-detail/{RUNNINGHUB_APP_ID}?apiType=4"
_RUNNINGHUB_MODELS_CACHE: tuple[str, ...] = RUNNINGHUB_MODELS
_RUNNINGHUB_MODELS_LOCK = asyncio.Lock()
_ROUTES_REGISTERED = False
_ACTIVE_REQUESTS: dict[str, asyncio.Task] = {}
_ACTIVE_CANCEL_EVENTS: dict[str, threading.Event] = {}
_ACTIVE_RH_TASKS: dict[str, tuple[str, str]] = {}
_ASYNC_OPTIMIZER_JOBS: dict[str, asyncio.Task] = {}
_GGUF_LOCK = threading.RLock()
_GGUF_MODEL = None
_GGUF_HANDLER = None
_GGUF_CONFIG: tuple[str, str, str] | None = None


def _llm_root() -> Path:
    # This package lives in ComfyUI/custom_nodes/<package>; keep the model
    # location deterministic for both desktop and hosted ComfyUI installs.
    return Path(__file__).resolve().parents[2] / "models" / "llm"


def _is_visual_model(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    text = json.dumps(config, ensure_ascii=False).lower()
    markers = (
        "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen-vl", "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
        "vision", "visual", "gemma3", "gemma-3", "minicpm-v", "internvl",
    )
    if not any(marker in text for marker in markers):
        return False
    return any((path / name).exists() for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"))


def _is_mmproj(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and path.suffix.lower() == ".gguf" and any(
        marker in name for marker in ("mmproj", "mm-projector", "projector")
    )


_GGUF_PRECISION_SUFFIX = re.compile(
    r"(?:[-_.](?:q\d+(?:[_-][a-z0-9]+)*|iq\d+(?:[_-][a-z0-9]+)*|f16|f32|fp16|fp32|bf16))+$",
    re.IGNORECASE,
)


def _gguf_identity(name: str, *, projector: bool = False) -> str:
    stem = Path(name).stem
    if projector:
        stem = re.sub(r"^(?:mmproj|mm-projector|projector)[-_. ]*", "", stem, flags=re.IGNORECASE)
    stem = _GGUF_PRECISION_SUFFIX.sub("", stem)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _gguf_model_signature(name: str) -> tuple[str, str] | None:
    """Return a stable model family/version and parameter size from loose filenames."""
    normalized = Path(name).stem.lower().replace("_", "-")
    family = re.search(r"(qwen\s*-?\s*3(?:[.\-]?\d+)?(?:\s*-?\s*vl)?)", normalized)
    size = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z])", normalized)
    if not family or not size:
        return None
    family_key = re.sub(r"[^a-z0-9]+", "", family.group(1))
    return family_key, f"{size.group(1)}b"


def _matching_mmproj(model_path: Path) -> list[Path]:
    root = _llm_root()
    identity = _gguf_identity(model_path.name)
    projectors = [path for path in root.rglob("*.gguf") if _is_mmproj(path)] if root.is_dir() else []
    exact = [path for path in projectors if _gguf_identity(path.name, projector=True) == identity]
    if exact:
        return sorted(exact, key=lambda path: path.name.lower())
    # Projectors are commonly named only by model generation and parameter
    # size, while the main model may add UD, publisher or quantization tokens.
    # First keep only the same model/version/size, then choose the closest name.
    signature = _gguf_model_signature(model_path.name)
    compatible = [path for path in projectors if signature and _gguf_model_signature(path.name) == signature]
    pool = compatible or projectors
    scored = []
    for path in pool:
        candidate = _gguf_identity(path.name, projector=True)
        containment = 1 if identity in candidate or candidate in identity else 0
        ratio = SequenceMatcher(None, identity, candidate).ratio()
        scored.append(((containment, ratio), path))
    if not scored:
        return []
    best = max(score for score, _path in scored)
    # If no family/size match exists, stay conservative and reject weak names.
    if not compatible and best[0] == 0 and best[1] < 0.72:
        return []
    return sorted(
        [path for score, path in scored if score == best],
        key=lambda path: path.name.lower(),
    )


def _gguf_handler_name(path: Path) -> str | None:
    name = path.name.lower()
    if re.search(r"qwen[-_. ]?3[._-]?(?:5|6|8)", name):
        return "qwen35"
    if "qwen3-vl" in name or "qwen3vl" in name:
        return "qwen3vl"
    return None


def _scan_visual_models() -> list[dict]:
    root = _llm_root()
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if _is_visual_model(path):
            result.append({"name": path.name, "path": str(path), "relative_path": path.name, "format": "transformers"})
    for path in sorted(root.rglob("*.gguf"), key=lambda item: str(item).lower()):
        if _is_mmproj(path) or _gguf_handler_name(path) is None:
            continue
        relative = path.relative_to(root).as_posix()
        candidates = _matching_mmproj(path)
        result.append({
            "name": path.name,
            "path": str(path),
            "relative_path": relative,
            "format": "gguf",
            "mmproj_candidates": [candidate.relative_to(root).as_posix() for candidate in candidates],
        })
    return result


def _scan_mmproj_models() -> list[dict]:
    root = _llm_root()
    if not root.is_dir():
        return []
    return [
        {
            "name": path.name,
            "path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
        }
        for path in sorted(root.rglob("*.gguf"), key=lambda item: str(item).lower())
        if _is_mmproj(path)
    ]


def _find_visual_model(selected: str) -> dict | None:
    return next((item for item in _scan_visual_models() if item["relative_path"] == selected), None)


def _normalize_config(data: dict | None) -> dict:
    current = DEFAULT_CONFIG
    data = data if isinstance(data, dict) else {}
    provider = str(data.get("provider") or current["provider"]).lower()
    preset = PROVIDERS.get(provider)
    config = {
        "mode": "local" if str(data.get("mode") or current.get("mode") or "api").lower() == "local" else "api",
        "provider": provider,
        "api_url": str(data.get("api_url") or (preset[0] if preset else current["api_url"])).strip(),
        "api_key": str(data.get("api_key") if "api_key" in data else current["api_key"]),
        "model": str(data.get("model") or (preset[1] if preset else current["model"])).strip(),
        "protocol": str(data.get("protocol") or (preset[2] if preset else current["protocol"])).lower(),
        "read_media": bool(data.get("read_media", current["read_media"])),
        "output_language": "中文" if str(data.get("output_language") or current.get("output_language") or "中文").lower() in {"中文", "chinese", "zh"} else "English",
        "local_model": str(data.get("local_model") or current.get("local_model") or "").strip(),
        "local_mmproj": str(data.get("local_mmproj") or current.get("local_mmproj") or "").strip(),
        "local_device": str(data.get("local_device") or current.get("local_device") or "cuda").lower(),
        "auto_optimize": bool(data.get("auto_optimize", current.get("auto_optimize", False))),
    }
    return config


def _public_config(config: dict) -> dict:
    return {
        **config,
        "api_key": "",
        "has_api_key": bool(config.get("api_key")),
        "runninghub_models": list(_RUNNINGHUB_MODELS_CACHE),
    }


def _runninghub_models_from_page(html: str) -> tuple[str, ...]:
    """Extract node 1 model choices from the public RunningHub Nuxt payload."""
    match = re.search(
        r'<script[^>]+id=["\']__NUXT_DATA__["\'][^>]*>([\s\S]*?)</script>',
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError):
        return ()
    candidates = []
    for value in payload if isinstance(payload, list) else ():
        if not isinstance(value, str) or not value.startswith("[["):
            continue
        try:
            field_data = json.loads(value)
        except (TypeError, ValueError):
            continue
        if not isinstance(field_data, list) or not field_data or not isinstance(field_data[0], list):
            continue
        models = [str(item).strip() for item in field_data[0] if isinstance(item, str) and str(item).strip()]
        if len(models) >= 5 and sum("/" in item for item in models) >= 3:
            candidates.append(models)
    if not candidates:
        return ()
    return tuple(dict.fromkeys(max(candidates, key=len)))


async def _refresh_runninghub_models() -> tuple[str, ...]:
    global _RUNNINGHUB_MODELS_CACHE
    async with _RUNNINGHUB_MODELS_LOCK:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(RUNNINGHUB_DETAIL_URL) as response:
                if response.status != 200:
                    raise RuntimeError(f"RunningHub model list returned HTTP {response.status}")
                models = _runninghub_models_from_page(await response.text())
                if not models:
                    raise RuntimeError("RunningHub model list was not found in the application detail")
                _RUNNINGHUB_MODELS_CACHE = models
                return models


def _resolve_input_file(name: str) -> Path:
    value = str(name or "").strip()
    if not value:
        raise ValueError("参考素材文件名为空")
    annotated = folder_paths.get_annotated_filepath(value)
    if annotated and os.path.isfile(annotated):
        return Path(annotated)
    path = Path(folder_paths.get_input_directory()) / value
    if path.is_file():
        return path
    raise FileNotFoundError(f"找不到参考素材: {value}")


def _ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError("RunningHub 视频预处理需要 FFmpeg，但当前 ComfyUI 环境未找到 FFmpeg") from error


async def _runninghub_video(source_name: str, duration: float, output_path: Path) -> None:
    source = _resolve_input_file(source_name)
    limit = max(0.1, float(duration or 5))
    # fps=1 chooses one frame for every source second without changing source
    # playback speed. Audio is trimmed at the same point and copied at normal speed.
    command = [
        _ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-t", f"{limit:.3f}",
        "-vf", "fps=1:start_time=0:round=down,scale=512:512:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-r", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(output_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError("RunningHub 参考视频预处理失败: " + stderr.decode("utf-8", "replace")[-1000:])


async def _runninghub_upload(session: aiohttp.ClientSession, api_key: str, *, path: Path | None = None,
                             data_url: str | None = None, filename: str = "reference.jpg") -> str:
    form = aiohttp.FormData()
    if path is not None:
        form.add_field("file", path.read_bytes(), filename=path.name, content_type="video/mp4")
    else:
        try:
            header, encoded = str(data_url or "").split(",", 1)
            mime = header.split(";", 1)[0].split(":", 1)[1]
            form.add_field("file", base64.b64decode(encoded), filename=filename, content_type=mime)
        except Exception as error:
            raise ValueError("RunningHub 参考图片数据无效") from error
    async with session.post(
        "https://www.runninghub.cn/openapi/v2/media/upload/binary",
        headers={"Authorization": f"Bearer {api_key}"}, data=form,
    ) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"RunningHub 素材上传失败 ({response.status}): {body[:1000]}")
        data = json.loads(body)
    if data.get("code") not in (0, "0", None):
        raise RuntimeError(f"RunningHub 素材上传失败: {data.get('message') or data.get('msg') or data}")
    result = data.get("data") or {}
    value = result.get("fileName") or result.get("download_url")
    if not value:
        raise RuntimeError("RunningHub 素材上传响应缺少 fileName")
    return str(value)


async def _runninghub_cancel(api_key: str, task_id: str) -> None:
    if not api_key or not task_id:
        return
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://www.runninghub.cn/task/openapi/cancel",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"apiKey": api_key, "taskId": task_id},
        ) as response:
            await response.read()


async def _runninghub_result_text(session: aiohttp.ClientSession, results: list) -> str:
    for item in results:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item["text"]).strip()
    for item in results:
        if not isinstance(item, dict):
            continue
        output_type = str(item.get("outputType") or item.get("fileType") or "").lower().lstrip(".")
        url = item.get("url") or item.get("fileUrl")
        if output_type not in {"txt", "text"} or not url:
            continue
        async with session.get(str(url)) as response:
            body = await response.read()
            if response.status >= 400:
                raise RuntimeError(f"RunningHub TXT 结果下载失败 ({response.status})")
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return body.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
    raise RuntimeError("RunningHub 返回结果中没有文本或 TXT 文件")


def _local_model_path(config: dict) -> Path:
    selected = str(config.get("local_model") or "").strip()
    if not selected:
        raise ValueError("请先选择本地视觉模型")
    found = _find_visual_model(selected)
    if found is None:
        raise ValueError("所选本地模型不存在，或不是可识别的视觉模型")
    return Path(found["path"])


def _selected_local_model(config: dict) -> dict:
    selected = str(config.get("local_model") or "").strip()
    found = _find_visual_model(selected)
    if found is None:
        raise ValueError("所选本地模型不存在，或不是可识别的视觉模型")
    return found


def _selected_mmproj(config: dict, model: dict) -> Path:
    candidates = list(model.get("mmproj_candidates") or [])
    selected = str(config.get("local_mmproj") or "").strip()
    if selected:
        available = {item["relative_path"] for item in _scan_mmproj_models()}
        if selected not in available:
            raise ValueError("已选择的mmproj视觉模型不存在，请重新选择")
        path = (_llm_root() / selected).resolve()
        if not path.is_file() or _llm_root().resolve() not in path.parents:
            raise ValueError("已选择的mmproj文件不存在")
        return path
    if not candidates:
        raise ValueError("未找到与当前GGUF模型匹配的mmproj视觉模型")
    if len(candidates) > 1:
        raise ValueError("检测到多个匹配的mmproj视觉模型，请在配置页面选择一个版本")
    return (_llm_root() / candidates[0]).resolve()


def _cuda_tag() -> str | None:
    try:
        import torch
        value = str(torch.version.cuda or "")
    except Exception:
        value = ""
    match = re.match(r"(\d+)\.(\d+)", value)
    return f"cu{match.group(1)}{match.group(2)}" if match else None


def _gguf_dependency_status() -> dict:
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_tag = _cuda_tag()
    platform_tag = "win_amd64" if system == "windows" and machine in {"amd64", "x86_64"} else (
        "linux_x86_64" if system == "linux" and machine in {"amd64", "x86_64"} else None
    )
    status = {
        "available": False,
        "reason": "",
        "system": platform.system(),
        "machine": platform.machine(),
        "python_tag": python_tag,
        "cuda_tag": cuda_tag or "",
        "release_url": "https://github.com/JamePeng/llama-cpp-python/releases",
        "wheel_url": "",
        "wheel_name": "",
    }
    try:
        import llama_cpp
        from llama_cpp.llama_chat_format import Qwen35ChatHandler  # noqa: F401
        status["available"] = True
        status["version"] = str(getattr(llama_cpp, "__version__", "unknown"))
        return status
    except Exception as error:
        status["reason"] = str(error)
    if system == "darwin":
        status["release_url"] = "https://github.com/JamePeng/llama-cpp-python/releases"
        return status
    if not cuda_tag or not platform_tag:
        return status
    # JamePeng release assets follow a stable version/cuda/python/platform
    # naming scheme. Resolve the current latest matching asset when online.
    try:
        request = urllib.request.Request(
            "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=30",
            headers={"User-Agent": "Goohai-MiniMax-H3-Integration"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            releases = json.load(response)
        marker = f"-{cuda_tag}-{'win' if system == 'windows' else 'linux'}-"
        for release in releases:
            if marker not in str(release.get("tag_name") or ""):
                continue
            for asset in release.get("assets") or []:
                name = str(asset.get("name") or "")
                if python_tag in name and platform_tag in name:
                    status["wheel_name"] = name
                    status["wheel_url"] = str(asset.get("browser_download_url") or "")
                    status["release_url"] = str(release.get("html_url") or status["release_url"])
                    return status
    except Exception:
        pass
    return status


def _unload_gguf_model() -> bool:
    global _GGUF_MODEL, _GGUF_HANDLER, _GGUF_CONFIG
    with _GGUF_LOCK:
        loaded = _GGUF_MODEL is not None or _GGUF_HANDLER is not None
        try:
            if _GGUF_MODEL is not None:
                _GGUF_MODEL.close()
        except Exception:
            pass
        try:
            stack = getattr(_GGUF_HANDLER, "_exit_stack", None)
            if stack is not None:
                stack.close()
        except Exception:
            pass
        _GGUF_MODEL = _GGUF_HANDLER = _GGUF_CONFIG = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return loaded


def _data_url_image(data_url: str):
    from io import BytesIO
    from PIL import Image
    _header, encoded = str(data_url).split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _unload_comfy_models() -> None:
    """Release ComfyUI-managed models and cached allocations before/after local VLM use."""
    from comfy import model_management

    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()


def _local_generate(config: dict, payload: dict) -> str:
    """Run one local VLM request with ComfyUI models unloaded on both sides."""
    _unload_comfy_models()
    try:
        return _local_generate_impl(config, payload)
    finally:
        _unload_comfy_models()


def _local_generate_impl(config: dict, payload: dict) -> str:
    """Load one local VLM only for this request, then release its objects."""
    model_path = _local_model_path(config)
    if model_path.suffix.lower() == ".gguf":
        return _gguf_generate(config, payload)
    model = processor = inputs = output = generated = None
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList
        cancel_event = payload.get("_cancel_event")

        class CancelledStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(cancel_event and cancel_event.is_set())
        device = config.get("local_device")
        if device not in {"cpu", "cuda"}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model.to("cpu")
        user_prompt = str(payload.get("prompt") or "")
        media = payload.get("media") if isinstance(payload.get("media"), list) else []
        content = [{"type": "text", "text": "User prompt:\n" + user_prompt}]
        for item in media:
            label = str(item.get("label") or "")
            kind = str(item.get("kind") or "")
            if kind == "audio":
                content.append({"type": "text", "text": f"{label} is an uploaded audio reference; audio data is not transmitted."})
            else:
                for data_url in item.get("images") or []:
                    content.append({"type": "image", "image": _data_url_image(data_url)})
                    content.append({"type": "text", "text": f"This visual belongs to {label}; use the label, never a filename."})
        system = _system_prompt(
            str(payload.get("task") or "T2VA"), float(payload.get("duration") or 5),
            [str(item.get("label")) for item in media if item.get("label")],
            str(config.get("output_language") or "English"),
            payload.get("context") if isinstance(payload.get("context"), dict) else {}, user_prompt,
        )
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [part["image"] for part in content if part.get("type") == "image"]
        inputs = processor(text=[text], images=images or None, padding=True, return_tensors="pt")
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=4096, do_sample=False,
                stopping_criteria=StoppingCriteriaList([CancelledStoppingCriteria()]),
            )
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("提示词优化已取消")
        generated = output[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    finally:
        output = generated = inputs = model = processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _gguf_generate(config: dict, payload: dict) -> str:
    global _GGUF_MODEL, _GGUF_HANDLER, _GGUF_CONFIG
    dependency = _gguf_dependency_status()
    if not dependency.get("available"):
        suffix = f"\n下载地址: {dependency['wheel_url']}" if dependency.get("wheel_url") else (
            f"\n发布页面: {dependency['release_url']}"
        )
        raise RuntimeError(
            "GGUF视觉模型依赖不可用。"
            f"当前环境: {dependency['system']} {dependency['machine']} / {dependency['python_tag']} / "
            f"{dependency.get('cuda_tag') or '无CUDA'}。{suffix}\n安装后请重启ComfyUI。"
        )
    model = _selected_local_model(config)
    model_path = Path(model["path"])
    mmproj_path = _selected_mmproj(config, model)
    handler_name = _gguf_handler_name(model_path)
    if handler_name not in {"qwen35", "qwen3vl"}:
        raise ValueError("当前GGUF视觉模型类型暂不支持")
    cancel_event = payload.get("_cancel_event")
    target_config = (str(model_path), str(mmproj_path), handler_name)
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen35ChatHandler, Qwen3VLChatHandler
        with _GGUF_LOCK:
            if _GGUF_MODEL is None or _GGUF_CONFIG != target_config:
                _unload_gguf_model()
                handler_cls = Qwen35ChatHandler if handler_name == "qwen35" else Qwen3VLChatHandler
                kwargs = {"clip_model_path": str(mmproj_path), "verbose": False}
                if handler_name == "qwen35":
                    kwargs["enable_thinking"] = False
                else:
                    kwargs["force_reasoning"] = False
                _GGUF_HANDLER = handler_cls(**kwargs)
                _GGUF_MODEL = Llama(
                    model_path=str(model_path), chat_handler=_GGUF_HANDLER,
                    n_gpu_layers=-1, n_ctx=16384, verbose=False,
                )
                _GGUF_CONFIG = target_config
        user_prompt = str(payload.get("prompt") or "")
        media = payload.get("media") if isinstance(payload.get("media"), list) else []
        labels = [str(item.get("label")) for item in media if item.get("label")]
        system = _system_prompt(
            str(payload.get("task") or "T2VA"), float(payload.get("duration") or 5), labels,
            str(config.get("output_language") or "English"),
            payload.get("context") if isinstance(payload.get("context"), dict) else {}, user_prompt,
        )
        content = [{"type": "text", "text": "User prompt:\n" + user_prompt}]
        for item in media:
            label = str(item.get("label") or "")
            if str(item.get("kind") or "") == "audio":
                content.append({"type": "text", "text": f"{label} is an uploaded audio reference; audio data is not transmitted."})
                continue
            for index, data_url in enumerate(item.get("images") or []):
                content.append({"type": "text", "text": f"{label} visual {index + 1}."})
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("提示词优化已取消")
        with _GGUF_LOCK:
            result = _GGUF_MODEL.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                max_tokens=4096, temperature=0.2, top_p=0.9,
            )
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("提示词优化已取消")
        return str(result["choices"][0]["message"]["content"] or "").removeprefix(": ").strip()
    finally:
        _unload_gguf_model()


def _endpoint(config: dict) -> str:
    base = config["api_url"].rstrip("/")
    if config["protocol"] == "gemini":
        if re.search(r":generateContent(?:\?|$)", base):
            return base
        return f"{base}/models/{urllib.parse.quote(config['model'], safe='-_.')}:generateContent"
    if config["protocol"] == "responses":
        return base if base.endswith("/responses") else f"{base}/responses"
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _system_prompt(
    task: str,
    duration: float,
    labels: list[str],
    output_language: str = "English",
    context: dict | None = None,
    user_prompt: str = "",
) -> str:
    task_upper = task.upper()
    fl2va = task_upper == "FL2VA"
    full_reference = task_upper in {"REF2VA", "HYBRID"}
    context = context if isinstance(context, dict) else {}
    label_rule = (
        "FL2VA picture labels must be bare lowercase: picture 1 and picture 2, never angle brackets. "
        "This is a mandatory content requirement, not merely a label-format example: the final prompt must explicitly "
        "contain both exact tokens 'picture 1' and 'picture 2'. The first line must state that picture 1 is the exact "
        f"opening-frame anchor at 0.00s and picture 2 is the exact ending-frame anchor at {duration:.2f}s. Never omit, "
        "rename, translate, capitalize, or replace either token, even when the requested output language is Chinese."
        if fl2va else
        "All media labels must use angle brackets exactly, for example <picture 1>, <video 1>, <audio 1>."
    )
    if full_reference:
        structure = (
            "Return exactly these six sections in this order, with each heading followed by a colon: "
            "subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, "
            "non_diegetic_music. Formatting is strict: every section heading must occupy its own line; put a newline "
            "immediately after the colon, write that section's content on the following line, and put one blank line "
            "between sections. Never place a heading and its content on the same line, and never join multiple sections "
            "into one paragraph. In subject_definitions, define reusable visible content as <Subject N> and cite "
            "its source label using one short sentence and no more than four identifying visual traits; define "
            "standalone <Picture N> only for a concrete frame anchor. In summary, use one sentence beginning "
            "with a square-bracketed combination of only the relationships that apply: keyframe completion, "
            "reference generation, video editing, video continuation, audio reuse, or audio reference. In "
            "retention_analysis, write one compact line per label and use fully_preserved, partially_preserved, "
            "attribute_transfer, or weak_reference for visual content, and fully_copy, partially_copy, reference, "
            "or weak_reference for audio. A relationship marker alone is never a complete entry. Every line must use "
            "the exact form '<label>: marker - explanation' and add a concise, concrete explanation after the hyphen. "
            "For fully_preserved, state which identity, appearance, scene, composition, or structural properties remain "
            "unchanged in the target. For attribute_transfer, state exactly which appearance, motion, scene, camera, "
            "timing, or structural attributes transfer from the reference and which target receives them. For audio, "
            "state whether the complete signal is copied unchanged, only selected portions or layers are copied, or "
            "only timbre, rhythm, delivery, or other specified attributes are referenced. Use enough information to "
            "make each reference relationship operational, normally one full sentence per label, but do not repeat an "
            "exhaustive appearance inventory already stated in subject_definitions. Never output bare entries such as "
            "'<picture 1>: fully_preserved', '<video 1>: attribute_transfer', or '<audio 1>: fully_copy'. "
            "Make detailed_description action-first, chronological, and production-ready. Give the scene a clear "
            "dramatic progression—establishment, development, a meaningful visual or emotional beat, and a resolved "
            "ending—without changing the user's intended event. Describe subject blocking, observable state changes, "
            "composition, camera movement, and the final visual landing. Use [Shot 1] and timed later shots only for "
            "motivated real cuts that reveal new information; otherwise design a fluid continuous shot. Normally allow "
            "about 350-700 Chinese characters or 220-450 English words for the complete six-section result, and use "
            "more when duration, dialogue, multiple subjects, or genuine shot complexity requires it."
        )
    else:
        structure = (
            "Return exactly these three sections in order: integrated_multimodal_description, overall_soundscape, "
            "non_diegetic_music. Formatting is strict: every section heading must occupy its own line; put a newline "
            "immediately after the colon, write that section's content on the following line, and put one blank line "
            "between sections. Never place a heading and its content on the same line. Build a complete audiovisual "
            "progression rather than merely paraphrasing the user's sentence. Use [Shot 1] for the opening and "
            "timestamps only for motivated real later cuts. For I2VA, "
            "anchor the supplied picture at 0.00s and develop forward. For L2VA, infer a plausible opening and "
            "converge exactly to the supplied picture at the end. For FL2VA, describe one continuous, observable "
            "motion path from picture 1 at 0.00s to picture 2 at the target end time. Give the main description enough "
            "detail to establish the opening composition, action onset, intermediate development, camera evolution, "
            "and a visually resolved ending within the supplied duration. In FL2VA, mention picture 1 again when "
            "establishing the opening state and picture 2 again when describing the final visual landing; both labels "
            "are compulsory and must not be replaced by generic phrases such as 'the first image' or 'the last image'."
        )

    keyframe_lines = []
    for item in context.get("keyframes") or []:
        if not isinstance(item, dict) or not item.get("label") or not item.get("role"):
            continue
        role_text = {
            "exact_first_frame": "is the exact first-frame anchor at 0.00s",
            "exact_last_frame": f"is the exact last-frame anchor at {duration:.2f}s",
            "keyframe": "is a concrete visual keyframe anchor",
        }.get(str(item["role"]), "is a visual reference")
        keyframe_lines.append(f"{item['label']} {role_text}")
    keyframe_rule = (
        "Keyframe roles supplied by the node: " + "; ".join(keyframe_lines) + ". Preserve these roles exactly."
        if keyframe_lines else
        "No concrete keyframe role was supplied by the node; do not invent one."
    )

    audio_mode = str(context.get("audio_mode") or "native")
    audio_rule = {
        "lock_source": (
            "Audio mode is original-audio output. State only that the source video's original audio is preserved "
            "unchanged as the complete target soundtrack and mark it fully_copy. Because the optimizer does not "
            "receive or hear the audio binary, do not identify, describe, classify, or invent any dialogue, music, "
            "ambience, sound effect, instrument, language, rhythm, or other audible content. In overall_soundscape, "
            "write only the selected-language equivalent of 'Preserve the source video's original audio unchanged.' "
            "Do not add any new sound, voice, animal vocalization, ambience, or music."
        ),
        "reference_only": (
            "Audio mode is reference-only: treat supplied audio as audio reference; do not claim its waveform "
            "is copied into the final soundtrack."
        ),
        "remix_source": (
            "Audio mode is source-audio remix: describe only partial copying or remixing of the supplied audio, "
            "never a 1:1 complete copy."
        ),
        "native": "Audio mode is automatic generation: do not claim any uploaded audio is copied into the final output.",
    }.get(audio_mode, "Describe the audio relationship conservatively and do not invent how it is used.")

    singing_rule = ""
    if re.search(
        r"唱|歌曲|歌词|歌唱|演唱|口型|嘴型|对口型|lip\s*sync|sing(?:ing|s)?|lyrics?|vocal",
        user_prompt,
        flags=re.IGNORECASE,
    ):
        singing_rule = (
            " The request involves singing or lip synchronization. Require accurate phoneme-level lip and jaw "
            "synchronization throughout the vocals: mouth movement should follow syllable onset and release, clear "
            "consonant closures, natural vowel shapes, pauses and breaths, and sustained notes; the mouth rests "
            "naturally during non-vocal spans. Also require stable facial identity and continuous mouth motion without "
            "generic loops, visible delay, frozen lips, or facial deformation. Do not reduce a singing performance "
            "to lip movement while the rest of the body remains still. Unless the user explicitly requests restrained "
            "or static performance, design evolving full-body or upper-body performance across the timeline: shifting "
            "weight, torso and shoulder rhythm, head movement, gaze changes, expressive but natural hand and arm "
            "phrasing, and transitions between distinct gestures. Make these movements non-repetitive and responsive "
            "to musical phrases, accents, pauses, intensity changes, and vocal emotion without inventing exact unheard "
            "beats. Coordinate purposeful camera motion and changing shot emphasis with the performance so the image "
            "does not feel like a static portrait with moving lips. Keep lip-sync requirements compact, but give the "
            "visible performance and camera progression enough concrete detail to guide a lively result. Because the audio binary is not "
            "transmitted to the optimizer, never invent exact lyrics, notes, beats, or timestamps and never claim "
            "to have analyzed or heard the file."
        )
    transfer_rule = ""
    if re.search(
        r"替换|置换|换成|变成|动作迁移|模仿.*动作|复刻.*动作|跟随.*动作|"
        r"replace|replacement|swap|motion\s*(?:transfer|reference)|imitate.*motion|copy.*motion",
        user_prompt,
        flags=re.IGNORECASE,
    ):
        transfer_rule = (
            " This is a subject-replacement or motion-transfer request. The following is a strict task-specific "
            "exception that overrides the general instructions to describe actions, timing, performance, camera, "
            "sound, and shot-by-shot events. Describe only the source-to-target relationship: which subject from "
            "<picture N> replaces which subject in <video N>, while the source video's complete motion, timing, "
            "scene, props, composition, camera movement, cuts, and temporal structure remain unchanged. Never name, "
            "infer, enumerate, paraphrase, or timestamp any concrete source-video action, gesture, pose change, facial "
            "expression, dialogue, object interaction, prop function, story event, sound, ambience, or music, even if "
            "it appears visible or inferable in the sampled frames. Use the sampled frames only to distinguish the "
            "subjects that must be replaced. Define replacement subjects only with official numbered labels such as "
            "<Subject 1> and <Subject 2>, never semantic labels such as <cat> or a Chinese name. Give each subject only "
            "the minimum clearly visible identity cues needed to avoid mismatch: subject type or gender, approximate "
            "age when reliable, main clothing and color, hairstyle or fur pattern, hat, and glasses. Do not infer an "
            "occupation or describe pose, expression, action, scene details, lighting, or unrelated appearance. Do not "
            "define the original subjects being removed as target <Subject N> entries unless needed only to make an "
            "unambiguous one-to-one replacement mapping. In detailed_description, keep a single [Shot 1] for the source "
            "timeline, but write a complete operational paragraph rather than a bare one-sentence relationship. State "
            "which source subject or scene is replaced by which referenced target, which target identity and major "
            "appearance or environment attributes remain stable, and which source-video properties remain unchanged: "
            "motion performance, facial-expression timing, blocking, screen direction, camera trajectory, framing, "
            "cuts, pacing, and duration when applicable. Explain how replaced subjects remain spatially integrated with "
            "preserved props and scene elements, but never invent or enumerate the source video's concrete actions, "
            "dialogue, props, events, or sounds; do not expand the source timeline or reconstruct it from sampled frames. Treat the three sampled images from one <video N> as observation "
            "frames of that single reference video, never as separate target shots or evidence of cuts."
        )
    language_rule = (
        "输出语言强制为中文：所有自然语言说明、主体描述、摘要、镜头描述、声音说明和音乐说明必须使用简体中文，"
        "不得输出英文句子。只有官方固定字段名 subject_definitions、summary、retention_analysis、"
        "detailed_description、integrated_multimodal_description、overall_soundscape、non_diegetic_music，"
        "媒体与主体标签、[Shot N]，以及 fully_preserved、partially_preserved、attribute_transfer、"
        "weak_reference、fully_copy、partially_copy、reference 等固定关系标记保留英文。返回前必须检查并将"
        "其他英文自然语言全部改写成中文。"
        if output_language == "中文" else
        "Output all natural-language content in English. Keep supplied dialogue, lyrics, and visible text in their "
        "original language."
    )
    dialogue_rule = (
        "Official dialogue formatting is mandatory in every task and is independent of the selected narration "
        "language. Put every user-supplied spoken line or lyric inside exactly one <d>[Language]...</d> block, using "
        "the dialogue's actual language name in English. In particular, Chinese speech must use the exact form "
        "<d>[Chinese]中文原文</d>; never write Chinese dialogue only inside quotation marks, and never omit [Chinese]. "
        "Keep the speaker description, stable speaker ID such as (S1), speaking action, and delivery outside the <d> "
        "block; inside it preserve only the user's exact words and punctuation without translation or rewriting. "
        "The output-language setting controls narration only: an English prompt must still retain supplied Chinese "
        "dialogue as <d>[Chinese]...</d>. Placement is as mandatory as formatting: every <d> block must be embedded "
        "inside integrated_multimodal_description or detailed_description at the exact chronological moment when the "
        "speaker speaks or sings, immediately after that speaker's speaking/singing action and delivery description. "
        "Never collect dialogue into a list, appendix, footer, or separate paragraph after the official sections. "
        "Never place a <d> block in subject_definitions, summary, retention_analysis, overall_soundscape, or "
        "non_diegetic_music. The final section is always non_diegetic_music, and absolutely no text or <d> block may "
        "follow that section's content. For multiple supplied lines, preserve their original order and place each line "
        "at its corresponding point in the shot timeline, with intervening actions and pauses described between lines "
        "when the user's request implies them. Before returning, verify that every supplied line appears exactly once "
        "inside the main shot timeline and nowhere else. Do not wrap paraphrased soundscape descriptions, ambient "
        "sounds, or dialogue that the user did not explicitly supply in <d> tags."
    )
    subject_shorthand_rule = (
        "Subject shorthand formatting is mandatory and globally consistent. If a stable short subject ID is used, "
        "it must always be written with parentheses as (S1), (S2), ... through (S20). Never output a bare S1, S2, "
        "or any other unparenthesized S-number in narration, shot descriptions, actions, dialogue attribution, or "
        "sound descriptions. <Subject N> definitions remain unchanged; (S1) is only the shorthand used to refer back "
        "to a defined subject. Before returning, scan the complete result and replace every standalone bare S1-S20 "
        "with its exact parenthesized form."
    )
    return (
        "You are a professional prompt writer for the open-source MiniMax H3 audiovisual model. Return only the "
        "final production-ready prompt without explanations, markdown fences, filenames, or invented media. "
        "Preserve the user's intent and every supplied dialogue, lyric, and visible-text word verbatim; do not "
        "invent dialogue or lyrics that were not provided. This is a video-generation prompt, not image captioning "
        "or visual reverse-prompting. Reference images are already passed into model conditioning, so identify each "
        "visual subject only with the minimum distinctive cues needed to disambiguate it: usually gender or subject "
        "type, main clothing, main hairstyle, and scene. Do not describe facial features, lighting, pose, background "
        "objects, or composition exhaustively unless they are essential to the requested motion or must change over "
        "time. Spend most words on what happens after the reference frame: causally connected actions, reactions, "
        "story progression, timing, performance, camera behavior, and audio-visual synchronization. Develop the "
        "user's idea into concrete visible beats that fit the duration: establish the situation, let the central "
        "action evolve through meaningful intermediate changes, add a restrained climax or reveal when appropriate, "
        "and finish on a clear visual result rather than an abrupt stop. Do not introduce unrelated characters, props, "
        "locations, conflicts, or plot twists. Avoid repeating the same static trait across sections, avoid decorative "
        "adjectives, and never turn quality requirements into a long negative-prompt checklist. Use professional but "
        "executable cinematography with dynamic video direction by default. Unless the user explicitly requests a static pose, locked-off camera, "
        "minimal movement, or a specific fixed composition, do not let the subject remain nearly motionless after the "
        "opening frame. Build visible motion in successive phases with clear variation and continuity: changes in body "
        "weight, posture, orientation, gesture, interaction, expression, spatial position, or object state. Avoid one "
        "gesture repeated mechanically throughout the clip. For music, singing, dance, performance, fashion, or other "
        "rhythmic scenes, let the body express phrase changes through varied natural gestures and coordinated torso, "
        "shoulder, head, hand, and footwork where framing permits; align movement energy and camera emphasis with broad "
        "musical progression without fabricating unheard exact beats. Treat the reference image as a starting anchor, "
        "not the main subject of the description: spend only enough text to identify it, then prioritize motion design, "
        "performance evolution, kinetic atmosphere, and a visually active ending. For each shot, integrate the useful framing or shot scale, subject blocking, and "
        "camera path into the action. When motion benefits the scene, choose a motivated Push In, Pull Out, Pan, Tilt, "
        "Truck, Pedestal, Arc Shot, Tracking Shot, or controlled Zoom, and specify subtle/large amplitude and slow/fast "
        "speed only when meaningful. Let camera movement reveal information, follow motion, emphasize a transformation, "
        "or land on the final state; do not stack random camera terms or use constant movement without purpose. Vary "
        "composition and visual emphasis over time so the video feels directed rather than static, while preserving "
        "spatial continuity, screen direction, subject identity, and reference constraints. Expand into continuous "
        "actions, camera choreography, synchronized audible events, ambience, and music with enough detail to guide "
        "generation effectively. Prefer a well-designed continuous shot when it can express the event clearly, but "
        "allow a small number of motivated cuts when they materially improve narrative clarity or reveal new visual "
        "information. Do not create labels outside the supplied label list, renumber labels, "
        "or replace labels with filenames. "
        f"Task={task}; duration={duration:.2f}s; available labels={', '.join(labels) or 'none'}. "
        f"{label_rule} {keyframe_rule} {audio_rule} {structure}{singing_rule}{transfer_rule} {dialogue_rule} "
        f"{subject_shorthand_rule} "
        "Audio items are label-only references; never claim you listened to them. Before returning, silently check "
        "that every required section is present in the correct order, every used label exists, no filename appears, "
        "no audio content was fabricated, all timing fits the target duration, and the audio relationship matches "
        f"the supplied mode. {language_rule}"
    )


def _user_parts(prompt: str, media: list[dict], read_media: bool) -> list[dict]:
    parts = [{"type": "text", "text": "User prompt:\n" + prompt}]
    for item in media:
        label = str(item.get("label") or "")
        kind = str(item.get("kind") or "")
        if kind == "audio":
            parts.append({"type": "text", "text": f"{label} is an uploaded audio reference (content not transmitted)."})
        elif read_media:
            for index, data_url in enumerate(item.get("images") or []):
                parts.append({"type": "text", "text": f"{label} visual {index + 1}."})
                parts.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})
        else:
            parts.append({"type": "text", "text": f"{label} is an uploaded {kind} reference."})
    return parts


def _strip_filenames(text: str) -> str:
    return re.sub(
        r"(?<![\w/\\])[\w .()\-\u4e00-\u9fff]+\.(?:png|jpe?g|webp|bmp|gif|mp4|mov|webm|mkv|avi|mp3|wav|flac|m4a|ogg|aac)(?!\w)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _format_prompt_sections(text: str) -> str:
    """Force official section headings onto separate lines without rewriting content."""
    headings = (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    pattern = r"\s*(" + "|".join(map(re.escape, headings)) + r")\s*:\s*"
    formatted = re.sub(pattern, lambda match: f"\n\n{match.group(1).lower()}:\n", str(text), flags=re.IGNORECASE)
    return formatted.strip()


def _normalize_subject_shorthand(text: str) -> str:
    """Guarantee that standalone S1-S20 references use official parentheses."""
    return re.sub(
        r"(?<![A-Za-z0-9_(<（])S([1-9]|1\d|20)(?![A-Za-z0-9_)>）])",
        lambda match: f"(S{match.group(1)})",
        str(text),
        flags=re.IGNORECASE,
    )


def _extract_explicit_dialogues(prompt: str) -> list[tuple[str, str]]:
    """Extract only dialogue that the user explicitly supplied, preserving its exact text."""
    source = str(prompt or "")
    candidates = []
    speech_marker = r"(?:说|说道|说着|喊|喊道|问|问道|回答|答道|台词|对白|says?|speaks?|shouts?|asks?|replies?)"
    patterns = (
        rf"{speech_marker}[^\n“”‘’\"']{{0,20}}[：:]?\s*[“\"]([^”\"\n]+)[”\"]",
        rf"{speech_marker}[^\n“”‘’\"']{{0,20}}[：:]?\s*[‘']([^’'\n]+)[’']",
        rf"{speech_marker}\s*[：:]\s*([^\n；;]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = match.group(1).strip().strip("“”‘’\"'").strip()
            existing = {item[0] for item in candidates}
            if not value or value in existing or any(value in item or item in value for item in existing):
                continue
            language = "Chinese" if re.search(r"[\u3400-\u9fff]", value) else "English"
            candidates.append((value, language))
    return candidates


def _ensure_supplied_dialogues(text: str, user_prompt: str, output_language: str) -> str:
    """Keep supplied dialogue exactly once and inside the main shot timeline."""
    result = str(text)
    dialogues_to_insert = []
    for index, (dialogue, language) in enumerate(_extract_explicit_dialogues(user_prompt), start=1):
        tag = f"<d>[{language}]{dialogue}</d>"
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        if tag in timeline:
            # Remove accidental duplicates outside the timeline while preserving the valid occurrence.
            prefix = result[:main_start].replace(tag, "")
            suffix = result[main_end:].replace(tag, "")
            result = prefix + timeline + suffix
            continue
        # A correctly tagged line outside the shot timeline is structurally invalid; move rather than duplicate it.
        result = result.replace(tag, "")
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        if dialogue in timeline:
            replaced = False
            for quoted in (f"“{dialogue}”", f"‘{dialogue}’", f'"{dialogue}"', f"'{dialogue}'"):
                if quoted in timeline:
                    timeline = timeline.replace(quoted, tag, 1)
                    replaced = True
                    break
            if not replaced:
                timeline = timeline.replace(dialogue, tag, 1)
            result = result[:main_start] + timeline + result[main_end:]
            continue
        dialogues_to_insert.append((index, tag))

    if dialogues_to_insert:
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        cue = re.search(
            r"(?:开口说话|开始说话|随后说|继续说|说话|演唱|唱着|begins? speaking|starts? speaking|speaks?|says?|sings?)",
            timeline,
            flags=re.IGNORECASE,
        )
        if cue:
            boundary = re.search(r"[。！？.!?](?:\s|$)", timeline[cue.end():])
            insert_at = cue.end() + boundary.end() if boundary else cue.end()
        else:
            shot = re.search(r"\[Shot 1\]\s*", timeline, flags=re.IGNORECASE)
            insert_at = shot.end() if shot else 0
        sentences = []
        for order, (index, tag) in enumerate(dialogues_to_insert):
            if str(output_language).lower() in {"中文", "chinese", "zh", "zh-cn"}:
                action = "说" if order == 0 else "随后继续说"
                sentences.append(f"画面中的说话者 (S{index}) {action}：{tag}")
            else:
                action = "says" if order == 0 else "then continues"
                sentences.append(f"The on-screen speaker (S{index}) {action}: {tag}")
        insertion = " " + " ".join(sentences) + " "
        timeline = timeline[:insert_at] + insertion + timeline[insert_at:]
        result = result[:main_start] + timeline + result[main_end:]
    return re.sub(r"[ \t]+\n", "\n", result).strip()


def _ensure_fl2va_picture_labels(text: str, task: str, duration: float, output_language: str) -> str:
    """Guarantee that an FL2VA result retains both official bare picture labels."""
    if str(task).upper() != "FL2VA":
        return text
    lowered = str(text).lower()
    if "picture 1" in lowered and "picture 2" in lowered:
        return text
    if str(output_language).lower() in {"中文", "chinese", "zh", "zh-cn"}:
        alignment = (
            f"参考图像与目标视频对齐关系：picture 1 对齐目标视频的 0.00 秒首帧；"
            f"picture 2 对齐目标视频的 {duration:.2f} 秒尾帧。"
        )
    else:
        alignment = (
            "Reference-picture alignment: picture 1 is the target video's exact opening frame at 0.00s; "
            f"picture 2 is its exact ending frame at {duration:.2f}s."
        )
    return f"{alignment}\n\n{text}".strip()


def _extract_openai(data: dict) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            return "".join(str(item.get("text") or "") for item in content if isinstance(item, dict)).strip()
        return str(content).strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("The API returned no optimized prompt")


def _extract_openai_responses(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    texts = []
    for output in data.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if not isinstance(content, dict) or content.get("type") not in {"output_text", "text"}:
                continue
            value = content.get("text")
            if isinstance(value, dict):
                value = value.get("value")
            if value:
                texts.append(str(value))
    if texts:
        return "".join(texts).strip()
    raise RuntimeError("OpenAI Responses returned no optimized prompt")


def _request_parts(config: dict, payload: dict):
    task = str(payload.get("task") or "T2VA")
    duration = float(payload.get("duration") or 5)
    media = payload.get("media") if isinstance(payload.get("media"), list) else []
    labels = [str(item.get("label")) for item in media if item.get("label")]
    user_prompt = str(payload.get("prompt") or "")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    system = _system_prompt(
        task,
        duration,
        labels,
        str(config.get("output_language") or "English"),
        context,
        user_prompt,
    )
    parts = _user_parts(user_prompt, media, bool(config.get("read_media")))
    headers = {"Content-Type": "application/json"}
    if config["protocol"] == "gemini":
        gemini_parts = []
        for part in parts:
            if part["type"] == "text":
                gemini_parts.append({"text": part["text"]})
            else:
                url = part["image_url"]["url"]
                header, encoded = url.split(",", 1)
                mime = header.split(";")[0].split(":", 1)[1]
                gemini_parts.append({"inline_data": {"mime_type": mime, "data": encoded}})
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": gemini_parts}],
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 2048},
        }
        url = _endpoint(config)
        separator = "&" if "?" in url else "?"
        url += separator + "key=" + urllib.parse.quote(config["api_key"])
    elif config["protocol"] == "responses":
        response_parts = []
        for part in parts:
            if part["type"] == "text":
                response_parts.append({"type": "input_text", "text": part["text"]})
            else:
                response_parts.append({
                    "type": "input_image",
                    "image_url": part["image_url"]["url"],
                    "detail": part["image_url"].get("detail", "low"),
                })
        body = {
            "model": config["model"],
            "instructions": system,
            "input": [{"role": "user", "content": response_parts}],
            "temperature": 0.35,
            "max_output_tokens": 2048,
        }
        headers["Authorization"] = f"Bearer {config['api_key']}"
        url = _endpoint(config)
    else:
        body = {
            "model": config["model"],
            "temperature": 0.35,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": parts},
            ],
        }
        headers["Authorization"] = f"Bearer {config['api_key']}"
        url = _endpoint(config)
    return url, headers, body


async def _request_runninghub(config: dict, payload: dict) -> str:
    api_key = str(config.get("api_key") or "")
    request_id = str(payload.get("request_id") or "")
    media = payload.get("media") if isinstance(payload.get("media"), list) else []
    task = str(payload.get("task") or "T2VA")
    duration = float(payload.get("duration") or 5)
    labels = [str(item.get("label")) for item in media if item.get("label")]
    system_prompt = _system_prompt(
        task, duration, labels, str(config.get("output_language") or "English"),
        payload.get("context") if isinstance(payload.get("context"), dict) else {},
        str(payload.get("prompt") or ""),
    )
    mapping_notes = []
    node_values = {
        "11": "None", "2": "None", "12": "None", "13": "None", "14": "None",
        "15": "None", "16": "None", "17": "None", "18": "None",
    }
    image_nodes = ("2", "12", "13", "14", "15", "16", "17", "18")
    timeout = aiohttp.ClientTimeout(total=200)
    temp_dir = Path(tempfile.mkdtemp(prefix="gh_h3_rh_"))
    task_id = ""
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            image_index = 0
            video_uploaded = False
            for item in media:
                kind = str(item.get("kind") or "")
                label = str(item.get("label") or "")
                if kind == "image":
                    images = item.get("images") or []
                    if image_index < len(image_nodes) and images:
                        node_id = image_nodes[image_index]
                        node_values[node_id] = await _runninghub_upload(
                            session, api_key, data_url=images[0], filename=f"reference_{image_index + 1}.jpg"
                        )
                        mapping_notes.append(f"{label} corresponds to uploaded image {image_index + 1}.")
                        image_index += 1
                    else:
                        mapping_notes.append(f"{label} exists as an image reference label, but its visual file was not transmitted.")
                elif kind == "video" and not item.get("labelOnly"):
                    if not video_uploaded and item.get("source_name"):
                        processed = temp_dir / "reference_1fps.mp4"
                        await _runninghub_video(str(item["source_name"]), duration, processed)
                        node_values["11"] = await _runninghub_upload(session, api_key, path=processed)
                        mapping_notes.append(f"{label} corresponds to the uploaded 1 FPS reference video.")
                        video_uploaded = True
                    else:
                        mapping_notes.append(f"{label} exists as a video reference label, but its visual file was not transmitted.")
                elif kind == "audio":
                    mapping_notes.append(f"{label} is an audio reference label; audio data is not separately transmitted.")

            user_prompt = "User prompt:\n" + str(payload.get("prompt") or "")
            if mapping_notes:
                user_prompt += "\n\nReference mapping:\n" + "\n".join(mapping_notes)
            node_info = [
                {"nodeId": "11", "fieldName": "file", "fieldValue": node_values["11"]},
                *[
                    {"nodeId": node_id, "fieldName": "image", "fieldValue": node_values[node_id]}
                    for node_id in image_nodes
                ],
                {"nodeId": "1", "fieldName": "model", "fieldValue": str(config.get("model") or RUNNINGHUB_MODELS[0])},
                {"nodeId": "9", "fieldName": "Text", "fieldValue": system_prompt},
                {"nodeId": "10", "fieldName": "Text", "fieldValue": user_prompt},
            ]
            async with session.post(
                f"https://www.runninghub.cn/openapi/v2/run/ai-app/{RUNNINGHUB_APP_ID}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"nodeInfoList": node_info, "instanceType": "default", "usePersonalQueue": False},
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"RunningHub 应用启动失败 ({response.status}): {body[:1000]}")
                started = json.loads(body)
            if started.get("code") not in (0, "0", None):
                raise RuntimeError(f"RunningHub 应用启动失败: {started.get('msg') or started}")
            start_data = started.get("data") if isinstance(started.get("data"), dict) else started
            task_id = str(start_data.get("taskId") or "")
            if not task_id:
                raise RuntimeError("RunningHub 应用启动响应缺少 taskId")
            if request_id:
                _ACTIVE_RH_TASKS[request_id] = (api_key, task_id)

            while True:
                cancel_event = payload.get("_cancel_event")
                if cancel_event is not None and cancel_event.is_set():
                    await _runninghub_cancel(api_key, task_id)
                    raise asyncio.CancelledError
                await asyncio.sleep(2)
                async with session.post(
                    "https://www.runninghub.cn/openapi/v2/query",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"taskId": task_id},
                ) as response:
                    body = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(f"RunningHub 任务查询失败 ({response.status}): {body[:1000]}")
                    queried = json.loads(body)
                if queried.get("code") not in (0, "0", None):
                    raise RuntimeError(f"RunningHub 任务查询失败: {queried.get('msg') or queried}")
                query_data = queried.get("data") if isinstance(queried.get("data"), dict) else queried
                status = str(query_data.get("status") or query_data.get("taskStatus") or "").upper()
                if status == "SUCCESS":
                    return await _runninghub_result_text(session, query_data.get("results") or [])
                if status in {"FAILED", "CANCELLED", "CANCELED"}:
                    message = query_data.get("errorMessage") or query_data.get("failedReason") or status
                    raise RuntimeError(f"RunningHub 提示词优化失败: {message}")
    except asyncio.CancelledError:
        active = _ACTIVE_RH_TASKS.get(request_id)
        if active:
            await _runninghub_cancel(*active)
        elif task_id:
            await _runninghub_cancel(api_key, task_id)
        raise
    except asyncio.TimeoutError as error:
        if task_id:
            await _runninghub_cancel(api_key, task_id)
        raise RuntimeError("RunningHub 提示词优化超过200秒，云端任务已取消") from error
    finally:
        if request_id:
            _ACTIVE_RH_TASKS.pop(request_id, None)
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _request_async(config: dict, payload: dict) -> str:
    if config.get("mode") == "local":
        model_path = _local_model_path(config)
        if model_path.suffix.lower() != ".gguf" and _local_missing_dependencies():
            raise RuntimeError("本地视觉模型依赖缺失: " + ", ".join(_local_missing_dependencies()))
        try:
            return await asyncio.wait_for(asyncio.to_thread(_local_generate, config, payload), timeout=200)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            cancel_event = payload.get("_cancel_event")
            if cancel_event:
                cancel_event.set()
            raise
    if config.get("provider") == "runninghub" or config.get("protocol") == "runninghub":
        return await _request_runninghub(config, payload)
    url, headers, body = _request_parts(config, payload)
    try:
        timeout = aiohttp.ClientTimeout(total=200)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"API request failed ({response.status}): {text[:1000]}")
                data = json.loads(text)
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
        raise RuntimeError(f"API request failed: {error}") from error
    if config["protocol"] == "gemini":
        try:
            return "".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"]).strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Gemini returned no optimized prompt")
    if config["protocol"] == "responses":
        return _extract_openai_responses(data)
    return _extract_openai(data)


async def get_prompt_optimizer_config(_request):
    config = _public_config(DEFAULT_CONFIG)
    config["models"] = _scan_visual_models()
    config["mmproj_models"] = _scan_mmproj_models()
    config["missing_dependencies"] = _local_missing_dependencies()
    config["gguf_dependency"] = _gguf_dependency_status()
    return web.json_response(config)


def _local_missing_dependencies() -> list[str]:
    import importlib.util
    required = ("torch", "transformers", "PIL", "accelerate", "safetensors")
    return [name for name in required if importlib.util.find_spec(name) is None]


async def list_prompt_optimizer_models(_request):
    return web.json_response({
        "models": _scan_visual_models(),
        "mmproj_models": _scan_mmproj_models(),
        "missing_dependencies": _local_missing_dependencies(),
        "gguf_dependency": _gguf_dependency_status(),
    })


async def refresh_runninghub_models(_request):
    try:
        models = await _refresh_runninghub_models()
        return web.json_response({"runninghub_models": list(models)})
    except Exception as error:
        return web.json_response({"error": str(error)}, status=502)


async def save_prompt_optimizer_config(request):
    return web.json_response(_public_config(_normalize_config(await request.json())))


def _prepare_prompt_optimization(payload: dict) -> tuple[dict, threading.Event]:
    node_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    config = _normalize_config(node_config)
    if config.get("mode") == "local":
        model_path = _local_model_path(config)
        if model_path.suffix.lower() == ".gguf":
            _selected_mmproj(config, _selected_local_model(config))
        elif _local_missing_dependencies():
            raise RuntimeError("本地视觉模型依赖缺失: " + ", ".join(_local_missing_dependencies()))
    elif not config.get("api_url") or not config.get("model") or not config.get("api_key"):
        raise ValueError("请先配置提示词优化 API")
    cancel_event = threading.Event()
    payload["_cancel_event"] = cancel_event
    return config, cancel_event


async def _run_prompt_optimization(config: dict, payload: dict) -> str:
    result = await _request_async(config, payload)
    if not result:
        raise RuntimeError("API 返回了空提示词")
    formatted = _format_prompt_sections(_strip_filenames(result))
    formatted = _ensure_fl2va_picture_labels(
        formatted,
        str(payload.get("task") or "T2VA"),
        float(payload.get("duration") or 5),
        str(config.get("output_language") or "English"),
    )
    formatted = _ensure_supplied_dialogues(
        formatted,
        str(payload.get("prompt") or ""),
        str(config.get("output_language") or "English"),
    )
    return _normalize_subject_shorthand(formatted)


def _register_optimizer_task(request_id: str, task: asyncio.Task, cancel_event: threading.Event) -> None:
    previous_cancel_event = _ACTIVE_CANCEL_EVENTS.pop(request_id, None)
    if previous_cancel_event is not None:
        previous_cancel_event.set()
    previous = _ACTIVE_REQUESTS.pop(request_id, None)
    if previous is not None:
        previous.cancel()
    _ACTIVE_CANCEL_EVENTS[request_id] = cancel_event
    _ACTIVE_REQUESTS[request_id] = task


async def optimize_prompt(request):
    request_id = ""
    try:
        payload = await request.json()
        request_id = str(payload.get("request_id") or "")
        config, cancel_event = _prepare_prompt_optimization(payload)
        task = asyncio.create_task(_run_prompt_optimization(config, payload))
        if request_id:
            _register_optimizer_task(request_id, task, cancel_event)
        formatted = await task
        return web.json_response({"prompt": formatted})
    except asyncio.CancelledError:
        return web.json_response({"error": "提示词优化已取消"}, status=499)
    except Exception as error:
        return web.json_response({"error": str(error)}, status=400)
    finally:
        if request_id:
            _ACTIVE_REQUESTS.pop(request_id, None)
            _ACTIVE_CANCEL_EVENTS.pop(request_id, None)


async def start_prompt_optimization(request):
    """Start a long optimization without holding a hosted reverse-proxy request open."""
    try:
        payload = await request.json()
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            raise ValueError("提示词优化请求缺少 request_id")
        config, cancel_event = _prepare_prompt_optimization(payload)
        task = asyncio.create_task(_run_prompt_optimization(config, payload))
        old_job = _ASYNC_OPTIMIZER_JOBS.pop(request_id, None)
        if old_job is not None:
            old_job.cancel()
        _ASYNC_OPTIMIZER_JOBS[request_id] = task
        _register_optimizer_task(request_id, task, cancel_event)
        return web.json_response({"request_id": request_id, "status": "running"}, status=202)
    except Exception as error:
        return web.json_response({"error": str(error)}, status=400)


async def prompt_optimization_status(request):
    request_id = str(request.query.get("request_id") or "")
    task = _ASYNC_OPTIMIZER_JOBS.get(request_id)
    if task is None:
        return web.json_response({"error": "找不到提示词优化任务"}, status=404)
    if not task.done():
        return web.json_response({"status": "running"})
    _ASYNC_OPTIMIZER_JOBS.pop(request_id, None)
    _ACTIVE_REQUESTS.pop(request_id, None)
    _ACTIVE_CANCEL_EVENTS.pop(request_id, None)
    try:
        return web.json_response({"status": "success", "prompt": task.result()})
    except asyncio.CancelledError:
        return web.json_response({"error": "提示词优化已取消"}, status=499)
    except Exception as error:
        return web.json_response({"error": str(error)}, status=400)


async def cancel_prompt_optimization(request):
    payload = await request.json()
    request_id = str(payload.get("request_id") or "")
    task = _ACTIVE_REQUESTS.pop(request_id, None)
    async_job = _ASYNC_OPTIMIZER_JOBS.pop(request_id, None)
    cancel_event = _ACTIVE_CANCEL_EVENTS.pop(request_id, None)
    if cancel_event is not None:
        cancel_event.set()
    if task is not None:
        task.cancel()
    if async_job is not None and async_job is not task:
        async_job.cancel()
    runninghub_task = _ACTIVE_RH_TASKS.pop(request_id, None)
    if runninghub_task is not None:
        try:
            await _runninghub_cancel(*runninghub_task)
        except Exception:
            pass
    return web.json_response({"cancelled": task is not None or async_job is not None or cancel_event is not None or runninghub_task is not None})


def register_prompt_optimizer_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    routes = instance.routes
    routes.get("/goohai/minimax-h3/prompt-optimizer/config")(get_prompt_optimizer_config)
    routes.get("/goohai/minimax-h3/prompt-optimizer/models")(list_prompt_optimizer_models)
    routes.get("/goohai/minimax-h3/prompt-optimizer/runninghub-models")(refresh_runninghub_models)
    routes.post("/goohai/minimax-h3/prompt-optimizer/config")(save_prompt_optimizer_config)
    routes.post("/goohai/minimax-h3/prompt-optimizer/optimize")(optimize_prompt)
    routes.post("/goohai/minimax-h3/prompt-optimizer/start")(start_prompt_optimization)
    routes.get("/goohai/minimax-h3/prompt-optimizer/status")(prompt_optimization_status)
    routes.post("/goohai/minimax-h3/prompt-optimizer/cancel")(cancel_prompt_optimization)
    _ROUTES_REGISTERED = True
    return True


register_prompt_optimizer_routes()
