import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_sensor_coords(npz_path: str) -> np.ndarray:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as data:
        if "sensor_coords" not in data:
            keys = ", ".join(data.files)
            raise KeyError(f"'sensor_coords' not found in {npz_path}. Available keys: {keys}")
        sensor_coords = data["sensor_coords"]

    return np.asarray(sensor_coords)


def normalize_xy(sensor_coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(sensor_coords)

    # Common outputs: (2, N) or (N, 2). Also tolerate 3D and keep first 2 axes.
    if coords.ndim != 2:
        raise ValueError(f"sensor_coords must be a 2D array, got shape {coords.shape}")

    if coords.shape[0] in (2, 3) and coords.shape[1] >= 2:
        xy = coords[:2, :].T
    elif coords.shape[1] in (2, 3) and coords.shape[0] >= 2:
        xy = coords[:, :2]
    else:
        raise ValueError(
            "Unsupported sensor_coords shape. Expected (2, N), (N, 2), (3, N), or (N, 3). "
            f"Got: {coords.shape}"
        )

    return xy.astype(float)


def plot_sensor_coords(
    xy: np.ndarray,
    save_path: str,
    title: str,
    annotate: bool,
    dpi: int,
    marker_size: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, c="#1f77b4", edgecolors="black", linewidths=0.4)

    if annotate:
        for idx, (x_val, y_val) in enumerate(xy):
            ax.text(x_val, y_val, str(idx), fontsize=7, ha="left", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sensor_coords from a .npz file.")
    parser.add_argument("--npz", required=True, help="Path to input .npz file containing sensor_coords.")
    parser.add_argument("--out", default="./temp/sensor_coords_plot.png", help="Path to save output figure.")
    parser.add_argument("--title", default="Sensor Coordinates", help="Figure title.")
    parser.add_argument("--annotate", action="store_true", help="Annotate each sensor with its index.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for export.")
    parser.add_argument("--marker-size", type=float, default=42.0, help="Scatter marker size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sensor_coords = load_sensor_coords(args.npz)
    xy = normalize_xy(sensor_coords)

    print(f"Loaded sensor_coords shape: {sensor_coords.shape} -> plotting {xy.shape[0]} points")
    plot_sensor_coords(
        xy=xy,
        save_path=args.out,
        title=args.title,
        annotate=args.annotate,
        dpi=args.dpi,
        marker_size=args.marker_size,
    )
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()

