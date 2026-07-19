"""Shared path and scene utilities for the novel-view-synthesis scripts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator


IMAGE_SUFFIXES: frozenset[str] = frozenset({
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
})

REQUIRED_TEST_POSE_COLUMNS: frozenset[str] = frozenset({
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
})


def _resolve_supplied_path(raw_path: str, option_name: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise FileNotFoundError(f"{option_name} must name an existing path; received an empty path")

    path = Path(raw_path).expanduser().resolve()
    return path


def require_directory(raw_path: str, option_name: str) -> Path:
    """Resolve a caller-supplied path and require an existing directory."""
    path = _resolve_supplied_path(raw_path, option_name)
    if not path.is_dir():
        raise FileNotFoundError(f"{option_name} must name an existing directory: {path}")
    return path


def require_file(raw_path: str, option_name: str) -> Path:
    """Resolve a caller-supplied path and require an existing file."""
    path = _resolve_supplied_path(raw_path, option_name)
    if not path.is_file():
        raise FileNotFoundError(f"{option_name} must name an existing file: {path}")
    return path


def iter_dataset_scenes(data_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield every scene below each immediate set directory in sorted order."""
    set_dirs = (
        path
        for path in data_root.iterdir()
        if path.is_dir() and path.name.casefold() != "__macosx"
    )
    for set_dir in sorted(set_dirs, key=lambda path: (path.name.casefold(), path.name)):
        scene_dirs = (
            path
            for path in set_dir.iterdir()
            if path.is_dir() and path.name.casefold() != "__macosx"
        )
        for scene_dir in sorted(scene_dirs, key=lambda path: (path.name.casefold(), path.name)):
            yield set_dir.name, scene_dir


def safe_relative_output_path(image_name: str) -> Path:
    """Return an output path only when the supplied name is safely relative."""
    if not image_name or not image_name.strip():
        raise ValueError("Output image name must not be empty")

    posix_path = PurePosixPath(image_name)
    windows_path = PureWindowsPath(image_name)
    if (
        image_name in {".", ".."}
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.root
        or windows_path.drive
        or any(part in {".", ".."} for part in (*posix_path.parts, *windows_path.parts))
    ):
        raise ValueError(f"Unsafe relative output image name: {image_name!r}")

    return Path(image_name)
