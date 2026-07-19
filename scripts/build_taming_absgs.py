#!/usr/bin/env python
"""Build a runnable Taming-AbsGS hybrid without modifying either source checkout.

The generated directory is a copy of Taming-3DGS with three deliberate changes:

* its rasterizer accumulates AbsGS's per-pixel absolute 2D position gradients;
* small Gaussians remain eligible for cloning through Taming's normal gradient;
* large Gaussians become eligible for splitting through AbsGS's absolute gradient.

Taming's score-based sampling and final Gaussian budget are intentionally left in
place.  Run this script in a writable directory (for example Kaggle working
storage), not inside the read-only dataset mount.
"""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path
from typing import Callable, Iterable, Sequence


class HybridBuildError(RuntimeError):
    """Raised when the checked-out source does not match the supported revisions."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Taming-3DGS + AbsGS hybrid training checkout."
    )
    parser.add_argument("--taming-root", required=True, help="Original Taming-3DGS checkout.")
    parser.add_argument("--absgs-root", required=True, help="Original AbsGS checkout.")
    parser.add_argument("--output-root", required=True, help="Writable directory for the generated hybrid.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing --output-root. The two source checkouts are never changed.",
    )
    return parser.parse_args(argv)


def require_directory(raw_path: str, flag: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise HybridBuildError(f"{flag} must be an existing directory: {path}")
    return path


def require_files(root: Path, relative_paths: Iterable[str], source_name: str) -> None:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise HybridBuildError(f"{source_name} is incomplete at {root}; missing: {joined}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise HybridBuildError(
            f"Could not apply {label}: expected one matching source block, found {count}."
        )
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str, expected_count: int, label: str) -> str:
    count = text.count(old)
    if count != expected_count:
        raise HybridBuildError(
            f"Could not apply {label}: expected {expected_count} matches, found {count}."
        )
    return text.replace(old, new)


def patch_file(path: Path, patch: Callable[[str], str]) -> None:
    write_text(path, patch(read_text(path)))


def copy_taming_source(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise HybridBuildError(
                f"--output-root already exists: {destination}. Pass --overwrite to replace it."
            )
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {".git", "__pycache__", "build", "dist"}
        ignored.update(name for name in names if name.endswith((".egg-info", ".so", ".pyd")))
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def validate_sources(taming_root: Path, absgs_root: Path) -> None:
    require_files(
        taming_root,
        (
            "arguments/__init__.py",
            "train.py",
            "gaussian_renderer/__init__.py",
            "scene/gaussian_model.py",
            "submodules/diff-gaussian-rasterization/rasterize_points.cu",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h",
            "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
        ),
        "Taming-3DGS",
    )
    require_files(
        absgs_root,
        (
            "scene/gaussian_model.py",
            "submodules/diff-gaussian-rasterization-abs/rasterize_points.cu",
            "submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu",
        ),
        "AbsGS",
    )

    abs_rasterizer = read_text(
        absgs_root / "submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu"
    )
    if "fabs(dL_dG * dG_ddelx * ddelx_dx)" not in abs_rasterizer:
        raise HybridBuildError("AbsGS rasterizer does not expose the expected homodirectional gradient.")


def patch_arguments(path: Path) -> None:
    def patch(text: str) -> str:
        return replace_once(
            text,
            "        self.densify_grad_threshold = 0.0002\n"
            "        self.random_background = False\n",
            "        self.densify_grad_threshold = 0.0002\n"
            "        # AbsGS threshold for splitting large, under-represented Gaussians.\n"
            "        self.densify_grad_abs_threshold = 0.0004\n"
            "        self.random_background = False\n",
            "AbsGS optimization argument",
        )

    patch_file(path, patch)


def patch_gaussian_model(path: Path) -> None:
    def patch(text: str) -> str:
        text = replace_once(
            text,
            "        self.xyz_gradient_accum = torch.empty(0)\n"
            "        self.denom = torch.empty(0)\n",
            "        self.xyz_gradient_accum = torch.empty(0)\n"
            "        # Per-pixel absolute view-space gradients required by AbsGS.\n"
            "        self.xyz_gradient_accum_abs = torch.empty(0)\n"
            "        self.denom = torch.empty(0)\n",
            "AbsGS gradient buffer declaration",
        )
        text = replace_once(
            text,
            "    def modify_functions(self):\n"
            "        old_opacities = self.get_opacity.clone()\n"
            "        self.opacity_activation = torch.abs\n"
            "        self.inverse_opacity_activation = identity_gate\n"
            "        self._opacity = self.opacity_activation(old_opacities)\n",
            "    def modify_functions(self):\n"
            "        old_opacities = self.get_opacity.clone()\n"
            "        self.opacity_activation = torch.abs\n"
            "        self.inverse_opacity_activation = identity_gate\n"
            "        self._opacity = self.opacity_activation(old_opacities)\n\n"
            "    def restore_absolute_opacity_mode(self):\n"
            "        \"\"\"Restore a post-HO checkpoint without transforming saved opacities.\"\"\"\n"
            "        self.opacity_activation = torch.abs\n"
            "        self.inverse_opacity_activation = identity_gate\n",
            "post-HO opacity-mode restore helper",
        )
        text = replace_once(
            text,
            "            self.xyz_gradient_accum,\n"
            "            self.denom,\n",
            "            self.xyz_gradient_accum,\n"
            "            self.xyz_gradient_accum_abs,\n"
            "            self.denom,\n",
            "AbsGS checkpoint state",
        )

        old_restore = '''    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)
'''
        new_restore = '''    def restore(self, model_args, training_args):
        # Support checkpoints written by unmodified Taming-3DGS as well as this hybrid.
        if len(model_args) == 13:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             shopt_dict,
             self.spatial_lr_scale) = model_args
            xyz_gradient_accum_abs = None
        elif len(model_args) == 14:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             xyz_gradient_accum_abs,
             denom,
             opt_dict,
             shopt_dict,
             self.spatial_lr_scale) = model_args
        else:
            raise ValueError(f"Unsupported checkpoint tuple with {len(model_args)} entries")

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = (
            torch.zeros_like(self.xyz_gradient_accum)
            if xyz_gradient_accum_abs is None
            else xyz_gradient_accum_abs
        )
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)
'''
        text = replace_once(text, old_restore, new_restore, "checkpoint restore compatibility")
        text = replace_once(
            text,
            "    def training_setup(self, training_args):\n"
            "        self.percent_dense = training_args.percent_dense\n"
            "        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n",
            "    def training_setup(self, training_args):\n"
            "        self.percent_dense = training_args.percent_dense\n"
            "        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n",
            "AbsGS gradient buffer initialization",
        )
        text = replace_once(
            text,
            "        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]\n\n"
            "        self.denom = self.denom[valid_points_mask]\n",
            "        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]\n"
            "        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]\n\n"
            "        self.denom = self.denom[valid_points_mask]\n",
            "AbsGS gradient buffer pruning",
        )
        text = replace_once(
            text,
            "        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))\n"
            "        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n",
            "        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))\n"
            "        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n"
            "        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")\n",
            "AbsGS gradient buffer reset after densification",
        )
        text = replace_once(
            text,
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter):\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)\n"
            "        self.denom[update_filter] += 1\n",
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter):\n"
            "        # Channels 0:2 are Taming's normal 2D gradient; channels 2:4 are\n"
            "        # the per-pixel absolute gradient accumulated in the patched rasterizer.\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(\n"
            "            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True\n"
            "        )\n"
            "        self.xyz_gradient_accum_abs[update_filter] += torch.norm(\n"
            "            viewspace_point_tensor.grad[update_filter, 2:4], dim=-1, keepdim=True\n"
            "        )\n"
            "        self.denom[update_filter] += 1\n",
            "AbsGS gradient accumulation",
        )
        text = replace_once(
            text,
            "    def densify_with_score(self, scores, max_screen_size, min_opacity, extent, budget, radii, iter_num=None):\n"
            "        \n"
            "        grad_vars = self.xyz_gradient_accum / self.denom\n"
            "        grad_vars[grad_vars.isnan()] = 0.0\n"
            "        self.tmp_radii = radii\n\n"
            "        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= 0.0002, True, False)\n"
            "        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= self.percent_dense*extent\n"
            "        split_qualifiers = torch.max(self.get_scaling, dim=1).values > self.percent_dense*extent\n\n"
            "        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)\n"
            "        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers)\n",
            "    def densify_with_score(\n"
            "        self, scores, max_screen_size, min_opacity, extent, budget, radii,\n"
            "        iter_num=None, grad_threshold=0.0002, abs_grad_threshold=0.0004\n"
            "    ):\n"
            "        \n"
            "        grad_vars = self.xyz_gradient_accum / self.denom\n"
            "        grad_vars[grad_vars.isnan()] = 0.0\n"
            "        abs_grad_vars = self.xyz_gradient_accum_abs / self.denom\n"
            "        abs_grad_vars[abs_grad_vars.isnan()] = 0.0\n"
            "        self.tmp_radii = radii\n\n"
            "        normal_grad_qualifiers = torch.where(\n"
            "            torch.norm(grad_vars, dim=-1) >= grad_threshold, True, False\n"
            "        )\n"
            "        abs_grad_qualifiers = torch.where(\n"
            "            torch.norm(abs_grad_vars, dim=-1) >= abs_grad_threshold, True, False\n"
            "        )\n"
            "        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= self.percent_dense*extent\n"
            "        split_qualifiers = torch.max(self.get_scaling, dim=1).values > self.percent_dense*extent\n\n"
            "        # Taming scores still rank the candidates. AbsGS only changes which\n"
            "        # large Gaussians are eligible for splitting.\n"
            "        all_clones = torch.logical_and(clone_qualifiers, normal_grad_qualifiers)\n"
            "        all_splits = torch.logical_and(split_qualifiers, abs_grad_qualifiers)\n",
            "hybrid clone and split qualification",
        )
        return text

    patch_file(path, patch)


def patch_renderer(path: Path) -> None:
    def patch(text: str) -> str:
        return replace_once(
            text,
            "    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=\"cuda\") + 0\n",
            "    # The first two channels receive the usual 2D mean gradient. The patched\n"
            "    # rasterizer writes AbsGS's absolute per-pixel gradient into channels 2:4.\n"
            "    screenspace_points = torch.zeros(\n"
            "        (pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device=\"cuda\"\n"
            "    ) + 0\n",
            "four-channel screen-space gradient placeholder",
        )

    patch_file(path, patch)


def patch_train(path: Path) -> None:
    def patch(text: str) -> str:
        text = replace_once(
            text,
            "    if checkpoint:\n"
            "        (model_params, first_iter) = torch.load(checkpoint)\n"
            "        gaussians.restore(model_params, opt)\n",
            "    if checkpoint:\n"
            "        # The checkpoint includes optimizer and NumPy state. Load only trusted files.\n"
            "        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)\n"
            "        gaussians.restore(model_params, opt)\n"
            "        # Post-HO checkpoints store absolute opacities, not sigmoid logits.\n"
            "        if first_iter >= args.ho_iteration:\n"
            "            gaussians.restore_absolute_opacity_mode()\n",
            "post-HO checkpoint opacity-mode restore",
        )
        return replace_once(
            text,
            "                                                budget=target_count, \n"
            "                                                radii=radii,\n"
            "                                                iter_num=completed_densification_steps + densify_iter_num)\n",
            "                                                budget=target_count, \n"
            "                                                radii=radii,\n"
            "                                                iter_num=completed_densification_steps + densify_iter_num,\n"
            "                                                grad_threshold=opt.densify_grad_threshold,\n"
            "                                                abs_grad_threshold=opt.densify_grad_abs_threshold)\n",
            "hybrid densification thresholds",
        )

    patch_file(path, patch)


def patch_native_rasterizer(root: Path) -> None:
    native_root = root / "submodules" / "diff-gaussian-rasterization"

    def patch_rasterize_points(text: str) -> str:
        return replace_once(
            text,
            "  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());\n",
            "  // xy: normal gradient; zw: AbsGS absolute per-pixel gradient.\n"
            "  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());\n",
            "four-channel rasterizer gradient tensor",
        )

    def patch_backward_header(text: str) -> str:
        return replace_all(
            text,
            "float3* dL_dmean2D",
            "float4* dL_dmean2D",
            2,
            "four-channel rasterizer header declarations",
        )

    def patch_backward_cuda(text: str) -> str:
        text = replace_all(
            text,
            "float3* dL_dmean2D",
            "float4* dL_dmean2D",
            3,
            "four-channel rasterizer CUDA declarations",
        )
        text = replace_all(
            text,
            "float3* __restrict__ dL_dmean2D",
            "float4* __restrict__ dL_dmean2D",
            2,
            "four-channel rasterizer CUDA restricted declarations",
        )
        text = replace_once(
            text,
            "\tfloat Register_dL_dmean2D_x = 0.0f;\n"
            "\tfloat Register_dL_dmean2D_y = 0.0f;\n",
            "\tfloat Register_dL_dmean2D_x = 0.0f;\n"
            "\tfloat Register_dL_dmean2D_y = 0.0f;\n"
            "\tfloat Register_dL_dmean2D_abs_x = 0.0f;\n"
            "\tfloat Register_dL_dmean2D_abs_y = 0.0f;\n",
            "AbsGS per-Gaussian gradient accumulators",
        )
        text = replace_once(
            text,
            "\t\t\tconst float tmp_x = dL_dG * dG_ddelx * ddelx_dx;\n"
            "\t\t\tRegister_dL_dmean2D_x += tmp_x;\n"
            "\t\t\tconst float tmp_y = dL_dG * dG_ddely * ddely_dy;\n"
            "\t\t\tRegister_dL_dmean2D_y += tmp_y;\n",
            "\t\t\tconst float tmp_x = dL_dG * dG_ddelx * ddelx_dx;\n"
            "\t\t\tRegister_dL_dmean2D_x += tmp_x;\n"
            "\t\t\tRegister_dL_dmean2D_abs_x += fabs(tmp_x);\n"
            "\t\t\tconst float tmp_y = dL_dG * dG_ddely * ddely_dy;\n"
            "\t\t\tRegister_dL_dmean2D_y += tmp_y;\n"
            "\t\t\tRegister_dL_dmean2D_abs_y += fabs(tmp_y);\n",
            "AbsGS per-Gaussian absolute gradient accumulation",
        )
        text = replace_once(
            text,
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);\n"
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);\n",
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);\n"
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);\n"
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].z, Register_dL_dmean2D_abs_x);\n"
            "\t\tatomicAdd(&dL_dmean2D[gaussian_idx].w, Register_dL_dmean2D_abs_y);\n",
            "AbsGS per-Gaussian absolute gradient output",
        )
        text = replace_once(
            text,
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);\n"
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);\n",
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);\n"
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);\n"
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));\n"
            "\t\t\tatomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));\n",
            "AbsGS per-pixel absolute gradient output",
        )
        text = replace_all(
            text,
            "(float3*)dL_dmean2D",
            "(float4*)dL_dmean2D",
            1,
            "four-channel rasterizer CUDA cast",
        )
        return text

    def patch_rasterizer_impl(text: str) -> str:
        return replace_all(
            text,
            "(float3*)dL_dmean2D",
            "(float4*)dL_dmean2D",
            2,
            "four-channel rasterizer implementation casts",
        )

    patch_file(native_root / "rasterize_points.cu", patch_rasterize_points)
    patch_file(native_root / "cuda_rasterizer" / "backward.h", patch_backward_header)
    patch_file(native_root / "cuda_rasterizer" / "backward.cu", patch_backward_cuda)
    patch_file(native_root / "cuda_rasterizer" / "rasterizer_impl.cu", patch_rasterizer_impl)


def write_build_info(output_root: Path, taming_root: Path, absgs_root: Path) -> None:
    info = f"""# Taming-AbsGS hybrid workspace

This directory was generated by `scripts/build_taming_absgs.py`.

- Taming source: `{taming_root}`
- AbsGS reference source: `{absgs_root}`

It keeps Taming's score-based budgeted densification and modifies only the copied
workspace:

1. The Taming rasterizer exposes the regular XY gradient plus AbsGS's per-pixel
   absolute XY gradient in a four-channel `means2D` gradient tensor.
2. Normal gradient controls candidate cloning for small Gaussians.
3. Absolute gradient controls candidate splitting for large Gaussians.
4. Taming's scores retain control of candidate ranking and budget allocation.

Do not edit the source repositories to reproduce this workspace; regenerate it
from the builder instead.
"""
    write_text(output_root / "TAMING_ABSGS_BUILD_INFO.md", textwrap.dedent(info))


def verify_generated_hybrid(root: Path) -> None:
    checks = {
        "arguments/__init__.py": ("densify_grad_abs_threshold",),
        "scene/gaussian_model.py": (
            "xyz_gradient_accum_abs",
            "abs_grad_qualifiers",
            "all_splits = torch.logical_and(split_qualifiers, abs_grad_qualifiers)",
            "restore_absolute_opacity_mode",
        ),
        "gaussian_renderer/__init__.py": ("(pc.get_xyz.shape[0], 4)",),
        "train.py": (
            "abs_grad_threshold=opt.densify_grad_abs_threshold",
            "if first_iter >= args.ho_iteration:",
            "gaussians.restore_absolute_opacity_mode()",
            "torch.load(checkpoint, weights_only=False)",
        ),
        "submodules/diff-gaussian-rasterization/rasterize_points.cu": ("zeros({P, 4}",),
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu": (
            "Register_dL_dmean2D_abs_x",
            "fabs(dL_dG * dG_ddelx * ddelx_dx)",
            "float4* dL_dmean2D",
        ),
    }
    failures: list[str] = []
    for relative_path, markers in checks.items():
        path = root / relative_path
        content = read_text(path) if path.is_file() else ""
        absent = [marker for marker in markers if marker not in content]
        if absent:
            failures.append(f"{relative_path}: missing {', '.join(absent)}")
    if failures:
        raise HybridBuildError("Generated hybrid verification failed:\n- " + "\n- ".join(failures))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    taming_root = require_directory(args.taming_root, "--taming-root")
    absgs_root = require_directory(args.absgs_root, "--absgs-root")
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root == taming_root or output_root == absgs_root:
        raise HybridBuildError("--output-root must be different from both source checkouts.")

    validate_sources(taming_root, absgs_root)
    copy_taming_source(taming_root, output_root, args.overwrite)

    patch_arguments(output_root / "arguments" / "__init__.py")
    patch_gaussian_model(output_root / "scene" / "gaussian_model.py")
    patch_renderer(output_root / "gaussian_renderer" / "__init__.py")
    patch_train(output_root / "train.py")
    patch_native_rasterizer(output_root)
    write_build_info(output_root, taming_root, absgs_root)
    verify_generated_hybrid(output_root)

    print("Taming-AbsGS hybrid workspace created successfully.")
    print(f"Output: {output_root}")
    print("Next: build the CUDA extensions inside that output directory, then train with train.py.")


if __name__ == "__main__":
    main()
