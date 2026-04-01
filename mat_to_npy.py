import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import scipy.io as io

from config import get_config


RF_DATA_PATTERN = re.compile(r"^RF_data(\d{5})\.mat$")


def build_sensor_coords_no_random_shift(dtype=np.float32) -> np.ndarray:
    """Reproduce KWave.py element coordinates with rand_shift_grid=0."""
    cfg = get_config(factor=1.0)

    pml_size = cfg.PMLSize_base
    grid_y_size = cfg.Ny * cfg.dy
    t_y = grid_y_size / 2 - (cfg.array_offset_y_grids + pml_size) * cfg.dy

    translation_x = 0.0
    translation_y = -t_y

    array_width_total = (cfg.element_num - 1) * cfg.element_pitch
    x_start = -array_width_total / 2

    sensor_coords = np.zeros((2, cfg.element_num), dtype=dtype)
    for ind in range(cfg.element_num):
        # Keep the same index formula as Kwave.py for strict compatibility.
        x_pos = x_start + (ind - 1) * cfg.element_pitch
        sensor_coords[0, ind] = x_pos + translation_x
        sensor_coords[1, ind] = translation_y

    return sensor_coords


def _extract_time_data_cat(model_input: np.ndarray) -> np.ndarray:
    """Extract a 3D time_data_cat array from MATLAB-loaded model_input."""
    if isinstance(model_input, np.ndarray) and np.issubdtype(model_input.dtype, np.number):
        if model_input.ndim != 3:
            raise ValueError(f"model_input must be 3D, got shape {model_input.shape}")
        return model_input

    # Common MATLAB struct container patterns.
    if isinstance(model_input, np.ndarray) and model_input.dtype.names:
        if "time_data_cat" not in model_input.dtype.names:
            raise KeyError(
                "model_input is a MATLAB struct but missing field 'time_data_cat'. "
                f"Available fields: {model_input.dtype.names}"
            )
        data = model_input["time_data_cat"]
        if isinstance(data, np.ndarray) and data.size == 1:
            data = data.item()
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"Extracted time_data_cat must be 3D, got shape {data.shape}")
        return data

    if isinstance(model_input, np.ndarray) and model_input.size == 1:
        obj = model_input.item()
        if hasattr(obj, "time_data_cat"):
            data = np.asarray(obj.time_data_cat)
            if data.ndim != 3:
                raise ValueError(f"Extracted time_data_cat must be 3D, got shape {data.shape}")
            return data

    raise TypeError(
        "Unsupported model_input format. Expected a 3D numeric array or a MATLAB struct containing 'time_data_cat'."
    )


def convert_mat_to_kwave_npz(mat_path: str, output_npz_path: str) -> None:
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat_data = io.loadmat(mat_path)
    if "model_input" not in mat_data:
        raise KeyError(f"'model_input' not found in {mat_path}. Available keys: {list(mat_data.keys())}")

    time_data_cat = _extract_time_data_cat(mat_data["model_input"])
    sensor_coords = build_sensor_coords_no_random_shift(dtype=np.float32)

    os.makedirs(os.path.dirname(output_npz_path) or ".", exist_ok=True)
    np.savez(output_npz_path, time_data_cat=time_data_cat, sensor_coords=sensor_coords)

    print(f"Saved NPZ: {output_npz_path}")
    print(f"time_data_cat: shape={time_data_cat.shape}, dtype={time_data_cat.dtype}")
    print(f"sensor_coords: shape={sensor_coords.shape}, dtype={sensor_coords.dtype}")


def _scan_rf_data_mats(input_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    matches: List[Tuple[int, str]] = []
    for name in os.listdir(input_dir):
        m = RF_DATA_PATTERN.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        matches.append((idx, os.path.join(input_dir, name)))

    matches.sort(key=lambda item: item[0])
    return matches


def convert_batch_rf_data(input_dir: str, output_dir: str) -> None:
    rf_files = _scan_rf_data_mats(input_dir)
    if not rf_files:
        raise ValueError(f"No files matching RF_dataxxxxx.mat found in: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(rf_files)} RF_data mat files in: {input_dir}")
    for index, mat_path in rf_files:
        output_name = f"RF_data{index:05d}.npz"
        output_path = os.path.join(output_dir, output_name)
        print(f"Converting {os.path.basename(mat_path)} -> {output_name}")
        convert_mat_to_kwave_npz(mat_path, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MATLAB .mat (field: model_input) to KWave-style .npz with sensor_coords."
    )
    parser.add_argument(
        "mat_file",
        nargs="?",
        help="Path to one input .mat file (single-file mode).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Single-file mode output .npz path. Default: same folder/name as input .mat",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Batch mode input folder containing RF_dataxxxxx.mat files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Batch mode output folder for RF_dataxxxxx.npz files. Default: same as --input-dir",
    )
    args = parser.parse_args()

    if args.input_dir is None and args.mat_file is None:
        parser.error("Provide either mat_file (single mode) or --input-dir (batch mode).")

    if args.input_dir is not None and args.mat_file is not None:
        parser.error("Use either mat_file or --input-dir, not both.")

    if args.output is not None and args.input_dir is not None:
        parser.error("--output is only valid in single-file mode.")

    return args


def main() -> int:
    args = _parse_args()

    if args.input_dir is not None:
        output_dir = args.output_dir if args.output_dir is not None else args.input_dir
        convert_batch_rf_data(args.input_dir, output_dir)
        return 0

    output_npz = args.output
    if output_npz is None:
        root, _ = os.path.splitext(args.mat_file)
        output_npz = root + ".npz"

    convert_mat_to_kwave_npz(args.mat_file, output_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
