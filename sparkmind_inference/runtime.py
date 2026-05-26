"""Runtime helpers for optional local dependency and model search paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MODEL_ROOT_ENV_KEYS = ("INFERENCE_SDK_MODEL_ROOTS", "INFERENCE_SDK_MODEL_ROOT")
SPARKMIND_PATH_ENV_KEYS = (
    "INFERENCE_SDK_SPARKMIND_PATH",
    "SPARKMIND_PATH",
    "SPARKMIND_ROOT",
)
SPARKMIND_LOCAL_RELATIVE_PATHS = (
    Path("third_party") / "SparkMind",
    Path("SparkMind"),
)


def _normalize_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def iter_unique_paths(candidates: Iterable[Path | str | None]) -> Iterator[Path]:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = _normalize_path(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def iter_env_paths(env_keys: Sequence[str]) -> Iterator[Path]:
    raw_candidates: list[str] = []
    for key in env_keys:
        value = os.environ.get(key)
        if not value:
            continue
        raw_candidates.extend(part.strip() for part in value.split(os.pathsep) if part.strip())
    yield from iter_unique_paths(raw_candidates)


def iter_local_sparkmind_roots(repo_root: Path | None = None) -> Iterator[Path]:
    """Yield local SparkMind checkout candidates for an editable SDK checkout."""
    root_path = _normalize_path(repo_root or Path(__file__).resolve().parents[1])
    yield from iter_unique_paths(
        root_path / relative_path for relative_path in SPARKMIND_LOCAL_RELATIVE_PATHS
    )


def format_optional_dependency_error(
    dependency_label: str,
    import_error: BaseException | None = None,
    *,
    min_python: tuple[int, int] | None = None,
    install_hint: str | None = None,
) -> str:
    """Build a user-facing error message for unavailable optional dependencies."""
    parts = [f"{dependency_label} 依赖不可用。"]

    if min_python is not None and sys.version_info < min_python:
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required_python = f"{min_python[0]}.{min_python[1]}"
        parts.append(
            f"当前环境是 Python {current_python}，但该依赖需要 Python >={required_python}。"
        )

    if install_hint:
        parts.append(install_hint)

    if import_error is not None:
        parts.append(f"底层导入错误: {import_error}")

    return " ".join(parts)


def configure_optional_import_paths(
    env_keys: Sequence[str] = SPARKMIND_PATH_ENV_KEYS,
    include_local_sparkmind: bool = True,
) -> list[Path]:
    """Add optional dependency roots from environment variables and local checkouts to sys.path."""
    candidates: list[Path] = list(iter_env_paths(env_keys))
    if include_local_sparkmind:
        candidates.extend(iter_local_sparkmind_roots())

    added_paths: list[Path] = []
    for candidate in reversed(list(iter_unique_paths(candidates))):
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str in sys.path:
            continue
        sys.path.insert(0, candidate_str)
        added_paths.insert(0, candidate)
    return added_paths


def iter_model_search_roots(checkpoint_path: Path | None = None) -> Iterator[Path]:
    """Yield generic model search roots without assuming a host repository name."""
    candidates: list[Path] = list(iter_env_paths(MODEL_ROOT_ENV_KEYS))

    if checkpoint_path is not None:
        normalized_checkpoint = _normalize_path(checkpoint_path)
        candidates.append(normalized_checkpoint)

        max_parent_depth = min(4, len(normalized_checkpoint.parents))
        for idx in range(max_parent_depth):
            candidates.append(normalized_checkpoint.parents[idx])

    candidates.extend(
        [
            Path.cwd() / "models",
            Path.cwd(),
        ]
    )

    yield from iter_unique_paths(candidates)
