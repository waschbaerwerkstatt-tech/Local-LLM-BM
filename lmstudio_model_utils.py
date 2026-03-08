from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

import requests


DEFAULT_MODELS = [
    "zai-org/glm-4.6v-flash",
    "google/gemma-3-12b",
    "mistralai/ministral-3-14b-reasoning",
    "qwen/qwen3-8b",
    "qwen/qwen3-14b",
    "qwen/qwen3-4b-thinking-2507",
    "qwen2.5-coder-7b-instruct",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "google/gemma-3-4b",
]

MODEL_TEMPERATURES = {
    "deepseek/deepseek-r1-0528-qwen3-8b": 0.6,
    "qwen/qwen3-8b": 0.7,
    "qwen/qwen3-14b": 0.7,
    "qwen/qwen3-4b-thinking-2507": 0.7,
    "qwen2.5-coder-7b-instruct": 0.7,
    "google/gemma-3-12b": 1.0,
    "google/gemma-3-4b": 1.0,
    "mistralai/ministral-3-14b-reasoning": 0.7,
    "zai-org/glm-4.6v-flash": 0.8,
}

MODEL_FILE_MARKERS = (
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
)


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_model_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def is_embedding_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return "embedding" in lowered or lowered.startswith("embed-") or "embed-text" in lowered


def sort_model_ids(model_ids: Sequence[str], preferred_models: Sequence[str]) -> List[str]:
    preferred = [model for model in preferred_models if model in model_ids]
    remaining = sorted(model for model in model_ids if model not in preferred)
    return preferred + remaining


def get_model_temperature(model: str, fallback: float) -> float:
    if model in MODEL_TEMPERATURES:
        return MODEL_TEMPERATURES[model]

    normalized = normalize_model_token(model)
    if "deepseek" in normalized and "r1" in normalized:
        return 0.6
    if "qwen3-5" in normalized or normalized.startswith("qwen-qwen3") or "qwen2-5-coder" in normalized:
        return 0.7
    if "gemma-3" in normalized:
        return 1.0
    if "ministral" in normalized:
        return 0.7
    if "glm-4" in normalized:
        return 0.8
    return fallback


def fetch_api_models(base_url: str, api_key: str, timeout_s: int = 10) -> List[str]:
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()

    models: List[str] = []
    for item in payload.get("data", []):
        model_id = str(item.get("id", "")).strip()
        if model_id and not is_embedding_model(model_id):
            models.append(model_id)
    return dedupe_keep_order(models)


def looks_like_model_folder(path: str) -> bool:
    try:
        for entry in os.scandir(path):
            if not entry.is_file():
                continue
            name = entry.name.lower()
            if name == "config.json" or name.endswith(MODEL_FILE_MARKERS):
                return True
    except OSError:
        return False
    return False


def load_model_name_from_config(config_path: str) -> Optional[str]:
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    model_name = payload.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


def discover_disk_models(models_root: str) -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []
    expanded_root = os.path.expanduser(models_root)
    if not os.path.isdir(expanded_root):
        return models

    try:
        providers = sorted(os.scandir(expanded_root), key=lambda entry: entry.name.lower())
    except OSError:
        return models

    for provider in providers:
        if not provider.is_dir():
            continue
        try:
            model_dirs = sorted(os.scandir(provider.path), key=lambda entry: entry.name.lower())
        except OSError:
            continue

        for model_dir in model_dirs:
            if not model_dir.is_dir() or not looks_like_model_folder(model_dir.path):
                continue
            config_model_name = load_model_name_from_config(os.path.join(model_dir.path, "config.json"))
            models.append(
                {
                    "provider": provider.name,
                    "folder": model_dir.name,
                    "path": model_dir.path,
                    "config_model_name": config_model_name or "",
                }
            )

    return models


def resolve_models(
    explicit_models: Optional[Sequence[str]],
    default_models: Sequence[str],
    model_source: str,
    base_url: str,
    api_key: str,
    timeout_s: int,
    models_root: str,
) -> Tuple[List[str], Dict[str, Any]]:
    disk_models = discover_disk_models(models_root)
    api_models: List[str] = []
    api_error: Optional[str] = None

    if model_source in {"auto", "api"} or explicit_models:
        try:
            api_models = fetch_api_models(base_url, api_key, timeout_s=timeout_s)
        except Exception as exc:
            api_error = str(exc)

    if explicit_models:
        selected_models = dedupe_keep_order(list(explicit_models))
        unavailable_models: List[str] = []
        if api_models:
            unavailable_models = [model for model in selected_models if model not in api_models]
        return selected_models, {
            "selected_source": "explicit",
            "api_models": api_models,
            "api_error": api_error,
            "disk_models": disk_models,
            "unavailable_explicit_models": unavailable_models,
        }

    if model_source == "defaults":
        selected_models = list(default_models)
    elif model_source == "api":
        if api_error is not None:
            raise RuntimeError(f"Could not fetch models from {base_url.rstrip('/')}/models: {api_error}")
        selected_models = sort_model_ids(api_models, default_models)
    elif model_source == "auto":
        if api_models:
            selected_models = sort_model_ids(api_models, default_models)
        else:
            selected_models = list(default_models)
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    return selected_models, {
        "selected_source": model_source if api_models or model_source != "auto" else "defaults-fallback",
        "api_models": api_models,
        "api_error": api_error,
        "disk_models": disk_models,
        "unavailable_explicit_models": [],
    }


def format_model_inventory(info: Dict[str, Any]) -> str:
    lines: List[str] = []

    api_models = info.get("api_models", [])
    api_error = info.get("api_error")
    if api_models:
        lines.append("Live API models (/v1/models):")
        for model in api_models:
            lines.append(f"  - {model}")
    else:
        reason = api_error or "no live model list available"
        lines.append(f"Live API models (/v1/models): unavailable ({reason})")

    disk_models = info.get("disk_models", [])
    if disk_models:
        lines.append("")
        lines.append("Installed model folders on disk:")
        for item in disk_models:
            config_model_name = item.get("config_model_name") or ""
            suffix = f" | config model_name={config_model_name}" if config_model_name else ""
            lines.append(f"  - {item['provider']}/{item['folder']}{suffix}")

    return "\n".join(lines)
