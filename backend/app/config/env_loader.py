from __future__ import annotations

import os
from pathlib import Path


def _strip_optional_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None

    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = raw_value.strip()
    # Support inline comments for unquoted values.
    if value and value[0] not in {"'", '"'} and " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    value = _strip_optional_quotes(value)
    return key, value


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_assignment(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


__all__ = ["load_env_file"]
