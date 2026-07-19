# Taming-3DGS Portable Kaggle Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a Taming-3DGS-only, path-explicit pipeline that trains, renders test poses, displays public/private results correctly, evaluates public renders, uploads all outputs to Hugging Face, and is copied into `VAI_NVS_CODE/` for use as a Kaggle dataset.

**Architecture:** The project scripts remain small command-line tools with no implicit repository, dataset, output, or Kaggle paths. A shared `scripts/nvs_utils.py` module owns input validation, scene discovery, CSV constants, and output-path safety; rendering imports Taming only after the caller supplies `--taming-root`. Training continues to call Taming’s upstream `train.py`, but that entry point will fail before training if `-s` or `-m` was omitted.

**Tech Stack:** Python 3.10+, argparse, pathlib, csv/json, Pillow, matplotlib, PyTorch/CUDA, Taming-3DGS CUDA extensions, pytest, `huggingface_hub` in Kaggle.

## Global Constraints

- Only Taming-3DGS is used for train, render, COLMAP loading, LPIPS/SSIM evaluation, and the delivered Kaggle bundle; no script may import or discover `external/gaussian-splatting`.
- Every filesystem or repository location used by project entry-point scripts must be supplied explicitly as a CLI argument: data root, output directory/file, Taming root, model directory, CSV, rendered images, ground-truth images, and reports directory.
- Hyperparameters retain upstream defaults unless the Kaggle caller explicitly overrides them. The documented recommended train profile must explicitly show `iterations`, `budget`, `mode`, `cams`, `densification_interval`, `save_iterations`, and `checkpoint_iterations`.
- `scripts/render_test_poses.py` and `scripts/render_taming_test_poses.py` are removed. `scripts/render_poses.py` is the only project test-pose renderer.
- `render_poses.py` requires `--model-path`; `--iteration -1` is allowed and selects the highest numeric `point_cloud/iteration_*/point_cloud.ply` under that caller-supplied model directory.
- The unified renderer must instantiate `GaussianModel(..., optimizer_type="default", rendering_mode="abs")` and use `separate_sh=True`, matching `external/taming-3dgs/render.py` checkpoint opacity semantics.
- Never use a hard-coded Kaggle path, project-root fallback, environment-variable repo lookup, or default report/dataset/output path.
- Dataset scans must discover immediate set directories below the caller-supplied data root and skip `__MACOSX`; they must not assume `public_set` or `private_set1` exists.
- Render output image names must be relative safe paths: reject absolute paths and any path component equal to `..`.
- `show_rendered_images.py` infers the current Phase 1 scene type from components of `--images-dir`, case-insensitively. Public scenes require `--gt-images`; private scenes render a gallery and must not need GT.
- Kaggle code dataset mounts are already unzipped. Documentation must copy the mounted `VAI_NVS_CODE` source to a caller-declared working directory but must never unzip an archive.
- The final `VAI_NVS_CODE/` contains only source required at runtime: `scripts/`, `external/taming-3dgs/`, and `README_KAGGLE.md`. Exclude `.git`, `__pycache__`, `*.pyc`, build directories, egg-info, local datasets, outputs, checkpoints, and logs.
- Do not commit or push as part of this work unless the user asks explicitly.

---

## Planned File Structure

```text
scripts/
├── nvs_utils.py                  # shared path/CSV/dataset safety utilities
├── inspect_dataset.py             # explicit-path manifest writer
├── verify_colmap_scene.py         # explicit-path Taming COLMAP verifier
├── render_poses.py                # sole Taming test-pose renderer
├── evaluate_predictions.py        # explicit-path public evaluator
└── show_rendered_images.py        # scene-aware public/private image gallery

tests/
├── conftest.py                    # add scripts/ to test imports
├── test_nvs_utils.py              # pure validation/discovery tests
├── test_renderer_paths.py         # iteration and unsafe image-name tests
├── test_display_scenes.py         # scene classification/page pairing tests
└── test_cli_contracts.py          # `--help` and required path contract tests

external/taming-3dgs/
└── train.py                       # rejects omitted `-s` and `-m` before output setup

docs/
├── kaggle_taming_3dgs.md          # executable cell-by-cell Kaggle guide
└── superpowers/plans/
    └── 2026-07-11-taming-kaggle-portable-pipeline.md

VAI_NVS_CODE/
├── scripts/
├── external/taming-3dgs/
└── README_KAGGLE.md
```

`nvs_utils.py` is intentionally a dependency-free module. The remaining project scripts import it when launched from `scripts/`, and pytest imports it after `tests/conftest.py` prepends that directory to `sys.path`. Taming CUDA modules are imported lazily inside `main()` after `--taming-root` validation, so `--help` and pure tests do not require built extensions.

---

### Task 1: Establish tests and shared path/scene utilities

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_nvs_utils.py`
- Create: `scripts/nvs_utils.py`

**Interfaces:**
- Produces `IMAGE_SUFFIXES: frozenset[str]` and `REQUIRED_TEST_POSE_COLUMNS: frozenset[str]` for all dataset/renderer scripts.
- Produces `require_directory(raw_path: str, option_name: str) -> Path` and `require_file(raw_path: str, option_name: str`; both expand `~`, resolve paths, and raise `FileNotFoundError` with the option name when invalid.
- Produces `iter_dataset_scenes(data_root: Path) -> Iterator[tuple[str, Path]]`, yielding sorted `(set_name, scene_dir)` from immediate child directories and skipping `__MACOSX`.
- Produces `safe_relative_output_path(image_name: str) -> Path`, rejecting absolute paths, empty paths, `.` paths, and `..` components.

- [ ] **Step 1: Write the failing test.**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

Create `tests/test_nvs_utils.py`:

```python
from pathlib import Path

import pytest

from nvs_utils import (
    iter_dataset_scenes,
    require_directory,
    require_file,
    safe_relative_output_path,
)


def test_require_directory_resolves_existing_directory(tmp_path: Path) -> None:
    assert require_directory(str(tmp_path), "--data-root") == tmp_path.resolve()


def test_require_directory_rejects_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"--data-root must be an existing directory"):
        require_directory(str(file_path), "--data-root")


def test_require_file_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"--poses-csv must be an existing file"):
        require_file(str(tmp_path / "missing.csv"), "--poses-csv")


def test_iter_dataset_scenes_discovers_unknown_set_names(tmp_path: Path) -> None:
    for relative in ("alpha_set/SceneB", "beta_set/sceneA", "__MACOSX/ignored"):
        (tmp_path / relative).mkdir(parents=True)

    assert list(iter_dataset_scenes(tmp_path)) == [
        ("alpha_set", tmp_path / "alpha_set" / "SceneB"),
        ("beta_set", tmp_path / "beta_set" / "sceneA"),
    ]


@pytest.mark.parametrize("unsafe_name", ["", ".", "../escape.png", "/tmp/escape.png", r"C:\\escape.png"])
def test_safe_relative_output_path_rejects_unsafe_names(unsafe_name: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        safe_relative_output_path(unsafe_name)


def test_safe_relative_output_path_accepts_nested_image_name() -> None:
    assert safe_relative_output_path("nested/render.JPG") == Path("nested/render.JPG")
```

- [ ] **Step 2: Run test to verify it fails.**

Run:

```powershell
python -m pytest tests/test_nvs_utils.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'nvs_utils'`.

- [ ] **Step 3: Write minimal implementation.**

Create `scripts/nvs_utils.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
REQUIRED_TEST_POSE_COLUMNS = frozenset(
    {
        "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
        "fx", "fy", "cx", "cy", "width", "height",
    }
)


def require_directory(raw_path: str, option_name: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{option_name} must be an existing directory: {path}")
    return path


def require_file(raw_path: str, option_name: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{option_name} must be an existing file: {path}")
    return path


def iter_dataset_scenes(data_root: Path) -> Iterator[tuple[str, Path]]:
    for set_dir in sorted(data_root.iterdir(), key=lambda path: path.name.casefold()):
        if not set_dir.is_dir() or set_dir.name == "__MACOSX":
            continue
        for scene_dir in sorted(set_dir.iterdir(), key=lambda path: path.name.casefold()):
            if scene_dir.is_dir() and scene_dir.name != "__MACOSX":
                yield set_dir.name, scene_dir


def safe_relative_output_path(image_name: str) -> Path:
    candidate = Path(image_name)
    if (
        not image_name.strip()
        or candidate.is_absolute()
        or candidate == Path(".")
        or any(part == ".." for part in candidate.parts)
    ):
        raise ValueError(f"image_name must be a safe relative path: {image_name!r}")
    return candidate
```

- [ ] **Step 4: Run test to verify it passes.**

Run:

```powershell
python -m pytest tests/test_nvs_utils.py -v
python -m py_compile scripts/nvs_utils.py
```

Expected: all tests pass and `py_compile` emits no output.

---

### Task 2: Make dataset inspection and COLMAP verification path-explicit

**Files:**
- Modify: `scripts/inspect_dataset.py`
- Modify: `scripts/verify_colmap_scene.py`
- Create: `tests/test_cli_contracts.py`

**Interfaces:**
- Consumes `nvs_utils.require_directory`, `iter_dataset_scenes`, `IMAGE_SUFFIXES`, and `REQUIRED_TEST_POSE_COLUMNS`.
- `inspect_dataset.py` accepts required `--data-root` and `--output-dir`, writes `dataset_manifest_phase1.json` and `dataset_manifest_phase1.md` below the caller-supplied output directory, and returns non-zero for an empty scene scan or invalid required inputs.
- `verify_colmap_scene.py` accepts required `--data-root`, `--output-file`, and `--taming-root`; it imports `scene.colmap_loader` from only the supplied Taming root.

- [ ] **Step 1: Write failing CLI contract tests.**

Create `tests/test_cli_contracts.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_inspector_requires_explicit_paths() -> None:
    result = run_script("inspect_dataset.py")
    assert result.returncode == 2
    assert "--data-root" in result.stderr
    assert "--output-dir" in result.stderr


def test_colmap_verifier_requires_explicit_paths() -> None:
    result = run_script("verify_colmap_scene.py")
    assert result.returncode == 2
    assert "--data-root" in result.stderr
    assert "--output-file" in result.stderr
    assert "--taming-root" in result.stderr


def test_inspector_help_uses_hyphenated_options() -> None:
    result = run_script("inspect_dataset.py", "--help")
    assert result.returncode == 0
    assert "--data-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--data_root" not in result.stdout
```

- [ ] **Step 2: Run test to verify it fails.**

Run:

```powershell
python -m pytest tests/test_cli_contracts.py -v
```

Expected: inspector/verifier required-path tests fail because both scripts currently run with defaults.

- [ ] **Step 3: Write minimal implementation.**

Refactor `scripts/inspect_dataset.py`:

```python
parser = argparse.ArgumentParser(description="Inspect every VAI NVS scene below an explicit dataset root.")
parser.add_argument("--data-root", required=True, help="Directory whose child directories are dataset sets.")
parser.add_argument("--output-dir", required=True, help="Directory for dataset manifest JSON and Markdown.")
```

Replace pandas CSV access with `csv.DictReader`; report a missing/empty header, do not access `image_name` unless every required column exists, and preserve duplicate/resolution checks. In `main()`:

```python
data_root = require_directory(args.data_root, "--data-root")
output_dir = Path(args.output_dir).expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
manifest = [scan_scene(scene_dir, set_name, warnings) for set_name, scene_dir in iter_dataset_scenes(data_root)]
if not manifest:
    raise ValueError(f"No scene directories were found below --data-root: {data_root}")
```

Keep expected scene-relative paths (`train/images`, `train/sparse/0`, `test/test_poses.csv`) as validation rules, not as inferred top-level paths.

Refactor `scripts/verify_colmap_scene.py`: delete `repo_root_from_script()` and `add_gaussian_splatting_to_path()`. Add:

```python
def require_taming_root(raw_path: str) -> Path:
    taming_root = require_directory(raw_path, "--taming-root")
    missing = [name for name in ("scene", "gaussian_renderer") if not (taming_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"--taming-root is not a Taming-3DGS checkout; missing directories: {', '.join(missing)}"
        )
    return taming_root


def import_colmap_loader(taming_root: Path):
    if str(taming_root) not in sys.path:
        sys.path.insert(0, str(taming_root))
    from scene import colmap_loader
    return colmap_loader
```

Use this parser contract:

```python
parser.add_argument("--data-root", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--taming-root", required=True)
parser.add_argument("--strict", action="store_true")
```

Validate roots before import; use `iter_dataset_scenes(data_root)` instead of a fixed set tuple; fail for no scenes. Write to exactly `Path(args.output_file).expanduser().resolve()`, creating only its parent.

- [ ] **Step 4: Run test to verify it passes.**

Run:

```powershell
python -m pytest tests/test_nvs_utils.py tests/test_cli_contracts.py -v
python scripts/inspect_dataset.py --help
python scripts/verify_colmap_scene.py --help
```

Expected: all pytest cases pass; help lists only hyphenated explicit path flags and does not trigger CUDA/Taming imports.

---

### Task 3: Replace both renderers with the single correct Taming renderer

**Files:**
- Delete: `scripts/render_test_poses.py`
- Delete: `scripts/render_taming_test_poses.py`
- Create: `scripts/render_poses.py`
- Create: `tests/test_renderer_paths.py`

**Interfaces:**
- `render_poses.py` consumes `--taming-root`, `--model-path`, `--poses-csv`, and `--output-dir` as required arguments.
- Exposes `find_checkpoint_ply(model_path: Path, requested_iteration: int) -> tuple[int, Path]` and `read_test_poses(csv_path: Path) -> list[TestPose]` for pure unit tests.
- `--iteration` defaults to `-1`; only this checkpoint-selection value may be inferred from the caller-supplied model directory.

- [ ] **Step 1: Write the failing test.**

Create `tests/test_renderer_paths.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from render_poses import find_checkpoint_ply, output_path_for_pose


def write_checkpoint(model_path: Path, iteration: int) -> Path:
    checkpoint = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ply")
    return checkpoint


def test_find_checkpoint_ply_uses_largest_complete_iteration(tmp_path: Path) -> None:
    older = write_checkpoint(tmp_path, 7000)
    write_checkpoint(tmp_path, 30000)
    (tmp_path / "point_cloud" / "iteration_40000").mkdir()

    iteration, checkpoint = find_checkpoint_ply(tmp_path, -1)

    assert iteration == 30000
    assert checkpoint.name == "point_cloud.ply"
    assert older.exists()


def test_find_checkpoint_ply_requires_requested_ply(tmp_path: Path) -> None:
    (tmp_path / "point_cloud" / "iteration_7000").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="point_cloud.ply"):
        find_checkpoint_ply(tmp_path, 7000)


def test_output_path_for_pose_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        output_path_for_pose(tmp_path, "../outside.jpg")


def test_output_path_for_pose_keeps_nested_relative_name(tmp_path: Path) -> None:
    assert output_path_for_pose(tmp_path, "nested/result.jpg") == tmp_path / "nested/result.jpg"
```

- [ ] **Step 2: Run test to verify it fails.**

Run:

```powershell
python -m pytest tests/test_renderer_paths.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'render_poses'`.

- [ ] **Step 3: Write minimal implementation.**

Create `scripts/render_poses.py` by porting only Taming logic: retain `TestPose`, `TestPoseCamera`, projection matrix, CSV parser, tensor-to-Pillow conversion, and JPEG behavior from the current Taming renderer. Do **not** keep `REPO_ROOT`, `TAMING_3DGS_ROOT`, `find_taming_root`, import-time Taming imports, `cfg_args` reading, or any Gaussian-Splatting import.

Use this parser:

```python
parser.add_argument("--taming-root", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--poses-csv", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--iteration", type=int, default=-1)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--white-background", action="store_true")
parser.add_argument("--black-background", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--limit", type=int)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--jpeg-quality", type=int, default=95)
parser.add_argument("--quiet", action="store_true")
```

Implement:

```python
def find_checkpoint_ply(model_path: Path, requested_iteration: int) -> tuple[int, Path]:
    point_cloud_root = model_path / "point_cloud"
    if requested_iteration >= 0:
        checkpoint = point_cloud_root / f"iteration_{requested_iteration}" / "point_cloud.ply"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint point cloud does not exist: {checkpoint}")
        return requested_iteration, checkpoint

    candidates: list[tuple[int, Path]] = []
    if point_cloud_root.is_dir():
        for iteration_dir in point_cloud_root.glob("iteration_*"):
            try:
                iteration = int(iteration_dir.name.rsplit("_", 1)[1])
            except ValueError:
                continue
            checkpoint = iteration_dir / "point_cloud.ply"
            if checkpoint.is_file():
                candidates.append((iteration, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint point_cloud.ply files found in: {point_cloud_root}")
    return max(candidates, key=lambda candidate: candidate[0])


def output_path_for_pose(output_dir: Path, image_name: str) -> Path:
    return output_dir / safe_relative_output_path(image_name)
```

Validate all required paths before CUDA initialization. Reject non-positive `--limit`, JPEG quality outside `1..100`, conflicting backgrounds, non-CUDA device when CUDA is unavailable, and invalid Taming root. Lazy-import Taming from only the supplied root. Use:

```python
gaussians = GaussianModel(args.sh_degree, optimizer_type="default", rendering_mode="abs")
pipeline = Namespace(separate_sh=True, convert_SHs_python=False, compute_cov3D_python=False, debug=False)
```

- [ ] **Step 4: Run test to verify it passes.**

Delete both old renderer files after `render_poses.py` exists. Run:

```powershell
python -m pytest tests/test_nvs_utils.py tests/test_renderer_paths.py -v
python scripts/render_poses.py --help
```

Expected: all tests pass; help lists `--taming-root`; legacy renderer files no longer exist.

- [ ] **Step 5: Run a conditional CUDA smoke render.**

With real manual values:

```powershell
python scripts/render_poses.py --taming-root "<TAMING_ROOT>" --model-path "<MODEL_DIR>" --poses-csv "<TEST_POSES_CSV>" --output-dir "<RENDER_DIR>" --iteration -1 --limit 1
```

Expected: prints selected PLY, writes the requested image name below `<RENDER_DIR>`, and matches CSV dimensions. If extensions/checkpoint are unavailable, record this as skipped; do not claim it passed.

---

### Task 4: Refactor evaluation and add scene-aware result display

**Files:**
- Modify: `scripts/evaluate_predictions.py`
- Create: `scripts/show_rendered_images.py`
- Create: `tests/test_display_scenes.py`
- Modify: `tests/test_cli_contracts.py`

**Interfaces:**
- Evaluator requires `--taming-root`, `--pred-images`, `--gt-images`, `--experiment-name`, `--output-dir`.
- Display script requires `--images-dir`; `--gt-images` is optional in parser but required after public-scene classification.
- Display script exposes `infer_scene_kind(images_dir: Path) -> tuple[str, str]` and `build_public_pairs(images_dir: Path, gt_images: Path) -> list[str]`.

- [ ] **Step 1: Write failing display behavior tests.**

Create `tests/test_display_scenes.py`:

```python
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
```

Append to `tests/test_cli_contracts.py`:

```python
def test_evaluator_requires_taming_and_output_paths() -> None:
    result = run_script("evaluate_predictions.py")
    assert result.returncode == 2
    for option in ("--taming-root", "--pred-images", "--gt-images", "--output-dir"):
        assert option in result.stderr


def test_display_script_requires_images_dir() -> None:
    result = run_script("show_rendered_images.py")
    assert result.returncode == 2
    assert "--images-dir" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails.**

Run:

```powershell
python -m pytest tests/test_display_scenes.py tests/test_cli_contracts.py -v
```

Expected: display module collection fails; evaluator required-path contract fails.

- [ ] **Step 3: Write minimal evaluator implementation.**

Remove evaluator project-root/Original-3DGS imports. Require:

```python
parser.add_argument("--taming-root", required=True)
parser.add_argument("--pred-images", required=True)
parser.add_argument("--gt-images", required=True)
parser.add_argument("--experiment-name", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--psnr-max", type=float, default=40.0)
parser.add_argument("--lpips-net", choices=["alex", "squeeze", "vgg"], default="vgg")
parser.add_argument("--device", default="cuda")
parser.add_argument("--max-images", type=int)
```

Add `import_metric_modules(taming_root: Path)` that validates Taming and lazily imports `LPIPS` and `ssim`. Reject nonpositive PSNR/max image count and unsafe experiment names:

```python
if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in {"", ".", ".."}:
    raise ValueError("--experiment-name must be a single safe file-name component.")
```

Retain strict GT filenames, dimension checks, metric formula, and report format.

- [ ] **Step 4: Write minimal display implementation.**

Create `scripts/show_rendered_images.py` with these scene sets:

```python
PUBLIC_SCENES = frozenset({"hcm0031", "hcm0034", "hcm0181", "hcm0193", "hcm0204"})
PRIVATE_SCENES = frozenset({"hcm0249", "hcm0254", "hcm0276", "hcm1439", "hni0131", "hni0265", "hni0366", "hni0437"})
```

Infer the type from reversed resolved path components, case-insensitively. Public requires `--gt-images`, validates same names and dimensions, and displays all images paginated as Render | Ground truth pairs. Private displays all rendered images in paginated four-column galleries. Use `PIL`, `matplotlib.pyplot`, and `plt.show()` once per page; do not write a montage or infer GT location.

- [ ] **Step 5: Run test to verify it passes.**

Run:

```powershell
python -m pytest tests/test_nvs_utils.py tests/test_display_scenes.py tests/test_cli_contracts.py -v
python scripts/evaluate_predictions.py --help
python scripts/show_rendered_images.py --help
```

Expected: all tests pass; evaluator help has no Original-3DGS path; display help documents images/GT/page arguments.

---

### Task 5: Eliminate implicit training output and author the Kaggle runbook

**Files:**
- Modify: `external/taming-3dgs/train.py`
- Modify: `docs/kaggle_taming_3dgs.md`
- Modify: `configs/taming_3dgs_hcm0031.yaml`

**Interfaces:**
- Taming `train.py` rejects missing `-s/--source_path` and `-m/--model_path` with parser exit 2 before CUDA/output setup.
- Kaggle runbook contains paste-ready cells and manually declared paths.

- [ ] **Step 1: Write the failing test.**

Append to `tests/test_cli_contracts.py`:

```python
def test_taming_train_requires_source_and_model_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "external" / "taming-3dgs" / "train.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "-s/--source_path is required" in result.stderr
    assert "-m/--model_path is required" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails.**

Run:

```powershell
python -m pytest tests/test_cli_contracts.py::test_taming_train_requires_source_and_model_paths -v
```

Expected: fails because current training starts instead of rejecting absent paths.

- [ ] **Step 3: Write minimal training validation.**

In `external/taming-3dgs/train.py`, after `parse_args` and before saving iterations/CUDA:

```python
if not args.source_path:
    parser.error("-s/--source_path is required")
if not args.model_path:
    parser.error("-m/--model_path is required")
source_path = Path(args.source_path).expanduser().resolve()
if not source_path.is_dir():
    parser.error(f"-s/--source_path must be an existing directory: {source_path}")
args.source_path = str(source_path)
args.model_path = str(Path(args.model_path).expanduser().resolve())
```

Import `Path`; replace UUID fallback in `prepare_output_and_logger` with a defensive `ValueError`; remove `uuid` import. Do not alter upstream hyperparameter defaults in `arguments/__init__.py`.

- [ ] **Step 4: Write the Kaggle runbook and path-free reference profile.**

Rewrite `docs/kaggle_taming_3dgs.md` in Vietnamese with ordered cells: GPU/Internet settings; manually declare `CODE_ROOT`, `WORK_ROOT`, `TAMING_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`; copy mounted source without unzip; build CUDA extensions; declare direct `TRAIN_DIR`, `POSES_CSV`, `GT_IMAGES_DIR`, `MODEL_DIR`, `RENDER_DIR`, `REPORT_DIR`; audit/verify; train; render; public/private display; public evaluation; Hugging Face upload.

The train cell must set and pass:

```bash
ITERATIONS=30000
BUDGET=3500000
MODE=final_count
CAMS=10
DENSIFICATION_INTERVAL=500
SAVE_ITERATIONS="7000 15000 30000"
CHECKPOINT_ITERATIONS="7000 15000 30000"
```

Use this train invocation, without defaults/fallback paths:

```bash
cd "$TAMING_ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  -s "$TRAIN_DIR" -m "$MODEL_DIR" \
  --iterations "$ITERATIONS" --budget "$BUDGET" --mode "$MODE" \
  --cams "$CAMS" --densification_interval "$DENSIFICATION_INTERVAL" \
  --save_iterations $SAVE_ITERATIONS \
  --checkpoint_iterations $CHECKPOINT_ITERATIONS \
  --test_iterations -1
```

Explain `BUDGET` semantics in final-count vs multiplier mode, state that other hyperparameters use upstream defaults, and use Kaggle Secret `HF_TOKEN` with manually supplied `HF_REPO_ID`, `HF_REPO_TYPE`, `HF_PATH_IN_REPO` for `HfApi.upload_folder` of all `OUTPUT_ROOT`.

Rewrite `configs/taming_3dgs_hcm0031.yaml` to be path-free:

```yaml
training:
  iterations: 30000
  budget: 3500000
  mode: final_count
  cams: 10
  densification_interval: 500
  save_iterations: [7000, 15000, 30000]
  checkpoint_iterations: [7000, 15000, 30000]
  test_iterations: [-1]
notes:
  - This file is a reference profile only; no project script reads it automatically.
  - Pass all filesystem paths directly to Kaggle commands.
  - budget is a final Gaussian count only when mode is final_count.
```

- [ ] **Step 5: Run test to verify it passes.**

Run:

```powershell
python -m pytest tests/test_cli_contracts.py -v
python -m py_compile external/taming-3dgs/train.py scripts/*.py
rg "external/gaussian-splatting|TAMING_3DGS_ROOT|/kaggle/working/taming-3dgs" scripts docs/kaggle_taming_3dgs.md configs/taming_3dgs_hcm0031.yaml
```

Expected: tests pass; no prohibited runtime path fallback remains.

---

### Task 6: Assemble the source-only Kaggle bundle and final verification

**Files:**
- Create: `VAI_NVS_CODE/scripts/`
- Create: `VAI_NVS_CODE/external/taming-3dgs/`
- Create: `VAI_NVS_CODE/README_KAGGLE.md`

**Interfaces:**
- Bundle scripts include `nvs_utils.py`, `render_poses.py`, and `show_rendered_images.py`; no legacy renderers.
- Bundle Taming tree has source/submodules, not VCS/cache/build/data/output artifacts.

- [ ] **Step 1: Inspect any existing bundle before replacement.**

Run:

```powershell
Get-ChildItem -Force "F:\Projects\ViettelAIRace2026\VAI_NVS_CODE"
```

Expected: directory absent or its contents are reviewed. If it has non-generated user files, stop and ask before deletion.

- [ ] **Step 2: Create/copy clean bundle source.**

After confirmation, create the requested tree, copy `scripts/` and `external/taming-3dgs/`, then remove only verified generated metadata (`.git`, `__pycache__`, `build`, `dist`, `*.egg-info`, `*.pyc`). Never copy datasets, checkpoints, outputs, logs, Original Gaussian Splatting, or local environments.

- [ ] **Step 3: Create `VAI_NVS_CODE/README_KAGGLE.md`.**

State that Kaggle mounts the dataset already extracted; copy it to a manually chosen work directory; use `render_poses.py` only; call display with GT for public and without GT for private; follow the complete cells in `docs/kaggle_taming_3dgs.md`. Exclude Windows paths, dataset slugs, HF IDs/tokens, default paths, and unzip instructions.

- [ ] **Step 4: Run final verification.**

Run:

```powershell
python -m pytest tests -v
python -m py_compile scripts/*.py external/taming-3dgs/train.py
Test-Path "F:\Projects\ViettelAIRace2026\VAI_NVS_CODE\scripts\render_poses.py"
Test-Path "F:\Projects\ViettelAIRace2026\VAI_NVS_CODE\scripts\render_test_poses.py"
Test-Path "F:\Projects\ViettelAIRace2026\VAI_NVS_CODE\scripts\render_taming_test_poses.py"
git diff --check
git status --short
```

Expected: pytest and compilation pass; unified renderer exists; both legacy renderer paths are `False`; `git diff --check` has no whitespace errors. Report any skipped CUDA integration check honestly and do not commit/push unless asked.

---

## Plan Self-Review

### Requirement coverage

- Explicit paths: Tasks 1, 2, 3, 4, and 5 remove script defaults and validate required CLI/train inputs.
- Taming-only rendering: Task 3 deletes both old renderers; uses lazy Taming imports, `separate_sh=True`, and `rendering_mode="abs"`.
- Automatic checkpoint search from a manual model folder: Task 3 tests and implements largest complete iteration lookup.
- Train hyperparameters: Task 5 keeps upstream defaults yet documents/passes requested profile variables.
- Public/private display: Task 4 infers known scene IDs and creates pair/gallery layouts.
- Kaggle/Hugging Face: Task 5 provides no-unzip cells and safe secret-based upload.
- Bundle: Task 6 creates `VAI_NVS_CODE` under the requested project path.
- Testing: Tasks 1–6 specify unit, CLI, compilation, structural, and conditional CUDA smoke checks.

### Placeholder scan

Angle-bracket values only occur in commands explicitly marked as manually supplied user paths. No implementation step leaves an unspecified function or validation rule.

### Type/interface consistency

Later tasks use `nvs_utils`, `find_checkpoint_ply`, `output_path_for_pose`, `infer_scene_kind`, and `build_public_pairs` with signatures defined by their creating tasks. All path flags are hyphenated consistently.
