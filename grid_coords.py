import argparse
from pathlib import Path

import numpy as np

from config import get_config


def build_grid_coords(nx: int, ny: int, dx: float, dy: float, dtype=np.float32):
    """Build x/y coordinate grids with shape [Nx, Ny].

    Coordinate convention:
    - Origin at image center.
    - +x points to image bottom.
    - +y points to image right.
    """
    # Anchor both x=0 and y=0 at the same pixel index (nx//2, ny//2).
    # Keep +x downward (row index increases) and +y rightward (col index increases).
    i0 = nx // 2
    j0 = ny // 2
    x_idx = np.arange(nx, dtype=np.float64) - i0
    y_idx = np.arange(ny, dtype=np.float64) - j0

    x_axis = x_idx * dx
    y_axis = y_idx * dy

    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

    if dtype is not None:
        x_grid = x_grid.astype(dtype, copy=False)
        y_grid = y_grid.astype(dtype, copy=False)

    return x_grid, y_grid


def _validate_bounds(bounds: tuple[float, float, float, float] | None) -> None:
    if bounds is None:
        return
    if len(bounds) != 4:
        raise ValueError("bound must be (x_min, x_max, y_min, y_max).")

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    if not (0.0 <= x_min <= 1.0 and 0.0 <= x_max <= 1.0 and 0.0 <= y_min <= 1.0 and 0.0 <= y_max <= 1.0):
        raise ValueError(f"bound must be within [0,1], got {bounds}.")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"bound must satisfy x_max>x_min and y_max>y_min, got {bounds}.")


def _compute_bound_indices(shape_2d: tuple[int, int], bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Map normalized bounds (x_min, x_max, y_min, y_max) to indices (r0, r1, c0, c1)."""
    h, w = int(shape_2d[0]), int(shape_2d[1])
    x_min, x_max, y_min, y_max = [float(v) for v in bounds]

    c0 = int(np.floor(x_min * w))
    c1 = int(np.ceil(x_max * w))
    r0 = int(np.floor(y_min * h))
    r1 = int(np.ceil(y_max * h))

    c0 = max(0, min(c0, w - 1))
    c1 = max(c0 + 1, min(c1, w))
    r0 = max(0, min(r0, h - 1))
    r1 = max(r0 + 1, min(r1, h))

    return r0, r1, c0, c1


def _apply_bound_crop(grid_xy: np.ndarray, bound_indices: tuple[int, int, int, int] | None) -> np.ndarray:
    if bound_indices is None:
        return grid_xy
    r0, r1, c0, c1 = bound_indices
    return grid_xy[r0:r1, c0:c1, :]


def save_grid_coords(output_path: Path, factor: float = 1.0, bound: tuple[float, float, float, float] | None = None):
    """Generate and save grid coordinates to a single .npy file.

    Saved array shape: [Nx, Ny, 2]
    - arr[..., 0] is x_grid
    - arr[..., 1] is y_grid
    """
    _validate_bounds(bound)

    cfg = get_config(factor=factor)
    x_grid, y_grid = build_grid_coords(cfg.Nx, cfg.Ny, cfg.dx, cfg.dy)

    grid_xy = np.stack([x_grid, y_grid], axis=-1)
    bound_indices = _compute_bound_indices((cfg.Nx, cfg.Ny), bound) if bound is not None else None
    grid_xy = _apply_bound_crop(grid_xy, bound_indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, grid_xy)

    return grid_xy, bound_indices


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate centered grid coordinate matrices.")
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Scale factor passed to get_config (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temp/grid_xy.npy"),
        help="Output .npy path (default: temp/grid_xy.npy)",
    )
    parser.add_argument(
        "--bound",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Normalized crop bound in [0,1]: x_min x_max y_min y_max",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    bound = tuple(args.bound) if args.bound is not None else None
    grid_xy, bound_indices = save_grid_coords(output_path=args.output, factor=args.factor, bound=bound)

    x_grid = grid_xy[..., 0]
    y_grid = grid_xy[..., 1]

    x_idx_axis = np.arange(x_grid.shape[0]) - (x_grid.shape[0] // 2)
    y_idx_axis = np.arange(x_grid.shape[1]) - (x_grid.shape[1] // 2)
    i0 = int(np.where(x_idx_axis == 0)[0][0])
    j0 = int(np.where(y_idx_axis == 0)[0][0])

    print(f"Saved: {args.output}")
    print(f"grid_xy shape: {grid_xy.shape} (expected [Nx, Ny, 2])")
    if bound is not None:
        print(f"bound (normalized): {bound}")
        print(f"bound indices (r0, r1, c0, c1): {bound_indices}")
    print(f"x_grid shape: {x_grid.shape}, y_grid shape: {y_grid.shape}")
    print(f"x index range: [{x_idx_axis.min()}, {x_idx_axis.max()}]")
    print(f"y index range: [{y_idx_axis.min()}, {y_idx_axis.max()}]")
    print(f"origin index (x=0,y=0): ({i0}, {j0})")
    print(f"value at origin: x={x_grid[i0, j0]:.6e}, y={y_grid[i0, j0]:.6e}")
    print(f"x at top-left: {x_grid[0, 0]:.6e}, bottom-left: {x_grid[-1, 0]:.6e}")
    print(f"y at top-left: {y_grid[0, 0]:.6e}, top-right: {y_grid[0, -1]:.6e}")


if __name__ == "__main__":
    main()

