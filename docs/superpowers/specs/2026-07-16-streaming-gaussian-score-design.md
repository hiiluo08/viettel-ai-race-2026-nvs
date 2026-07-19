# Streaming Gaussian Score Accumulator Design

## Goal

Allow `CAMS=-1` to score all training cameras during Taming-3DGS densification without allocating a score matrix proportional to `num_cameras × num_gaussians`.

## Scope

Modify only `external/taming-3dgs/utils/taming_utils.py::compute_gaussian_score` and add focused unit coverage. Preserve the public function signature and its call site in `external/taming-3dgs/train.py`.

## Current behavior

For every selected camera, `compute_gaussian_score` computes an `agg_importance` vector for visible Gaussians and writes it to one row of a tensor shaped `(len(camlist), num_points)`. It then returns the sum across camera rows.

At 4,000,000 Gaussians and 200 cameras, the float32 score matrix alone needs about 2.98 GiB.

## Design

Replace the camera-by-Gaussian matrix with a single float32 CUDA vector:

1. Allocate `gaussian_importance` with shape `(num_points,)`, initialized to zero.
2. For every camera, retain the existing two renders and all existing importance calculations.
3. Add `agg_importance[visibility_filter]` directly into the corresponding entries of the accumulator.
4. Return the accumulator without a final reduction.

The result is mathematically equivalent to summing the former matrix over cameras, up to ordinary floating-point reduction-order differences.

## Compatibility contract

- Preserve the function signature and CUDA/device behavior.
- Return a one-dimensional `torch.float32` tensor of length `num_points`.
- Preserve camera traversal order and all importance formulae.
- Preserve zero contribution for Gaussians invisible in a camera.
- An empty camera list returns a zero vector.
- No changes to renderer behavior, autograd mode, loss weighting, camera selection, or Taming budget logic.

## Memory impact

For 4,000,000 Gaussians:

| Configuration | Score storage before | Score storage after |
|---|---:|---:|
| `CAMS=25` | ~0.37 GiB | ~15.3 MiB |
| `CAMS=200` | ~2.98 GiB | ~15.3 MiB |

Temporary per-view renderer buffers remain unchanged. Runtime remains approximately proportional to the number of scored cameras; this change addresses the avoidable score-matrix memory cost.

## Validation

Add a self-contained test that mocks the renderer and photometric-loss calculation, then:

1. Builds deterministic per-camera visibility masks and importance values.
2. Calculates the expected result with the legacy matrix-and-sum algorithm.
3. Asserts that the streaming implementation matches with `torch.testing.assert_close`.
4. Covers overlapping visibility, a camera with no visible Gaussian, output shape, dtype, and device.

## Out of scope

- Camera batching or parallel scoring.
- `torch.no_grad()` changes.
- Modifying score coefficients, threshold tuning, opacity logic, or densification schedules.
- Changes to the duplicate packaging/source trees unless the active training command proves they are used.
