from pathlib import Path

import pytest

from show_rendered_images import build_public_pairs, infer_scene_kind


def test_infer_scene_kind_is_case_insensitive() -> None:
    scene_id, kind = infer_scene_kind(Path("outputs/renders/HCM0031/images"))
    assert scene_id == "hcm0031"
    assert kind == "public"


def test_infer_scene_kind_identifies_private_scene() -> None:
    scene_id, kind = infer_scene_kind(Path("outputs/renders/HNI0437"))
    assert scene_id == "hni0437"
    assert kind == "private"


def test_infer_scene_kind_rejects_unknown_scene() -> None:
    with pytest.raises(ValueError, match="could not infer a known Phase 1 scene"):
        infer_scene_kind(Path("outputs/renders/unknown_scene"))


def test_build_public_pairs_requires_all_ground_truth_files(tmp_path: Path) -> None:
    rendered = tmp_path / "hcm0031" / "renders"
    ground_truth = tmp_path / "ground_truth"
    rendered.mkdir(parents=True)
    ground_truth.mkdir()
    (rendered / "a.png").write_bytes(b"not-an-image")

    with pytest.raises(FileNotFoundError, match="Missing ground-truth images"):
        build_public_pairs(rendered, ground_truth)
