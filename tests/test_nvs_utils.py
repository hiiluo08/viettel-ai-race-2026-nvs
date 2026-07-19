from pathlib import Path

import pytest

from nvs_utils import (
    IMAGE_SUFFIXES,
    REQUIRED_TEST_POSE_COLUMNS,
    iter_dataset_scenes,
    require_directory,
    require_file,
    safe_relative_output_path,
)


def test_public_constants_match_dataset_contract() -> None:
    assert IMAGE_SUFFIXES == frozenset({
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    })
    assert REQUIRED_TEST_POSE_COLUMNS == frozenset({
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


def test_require_directory_resolves_existing_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    assert require_directory(str(data_root), "--data-root") == data_root.resolve()


def test_require_directory_rejects_file_with_cli_option(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("content", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="--data-root"):
        require_directory(str(not_a_directory), "--data-root")


def test_require_file_rejects_missing_path_with_cli_option(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="--poses-csv"):
        require_file(str(missing_file), "--poses-csv")


def test_iter_dataset_scenes_discovers_all_sets_sorted_case_insensitively(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    (data_root / "beta_set" / "sceneA").mkdir(parents=True)
    (data_root / "alpha_set" / "SceneB").mkdir(parents=True)
    (data_root / "__MACOSX" / "ignored_scene").mkdir(parents=True)
    (data_root / "alpha_set" / "__MACOSX" / "ignored_scene").mkdir(parents=True)
    (data_root / "not-a-set").write_text("ignored", encoding="utf-8")

    discovered = list(iter_dataset_scenes(data_root))

    assert discovered == [
        ("alpha_set", data_root / "alpha_set" / "SceneB"),
        ("beta_set", data_root / "beta_set" / "sceneA"),
    ]


@pytest.mark.parametrize(
    "image_name",
    [
        "",
        "   ",
        ".",
        "../escape.png",
        "/tmp/escape.png",
        r"C:\\escape.png",
        r"\escape.png",
    ],
)
def test_safe_relative_output_path_rejects_unsafe_names(image_name: str) -> None:
    with pytest.raises(ValueError):
        safe_relative_output_path(image_name)


def test_safe_relative_output_path_accepts_nested_relative_name() -> None:
    assert safe_relative_output_path("nested/render.JPG") == Path("nested/render.JPG")
