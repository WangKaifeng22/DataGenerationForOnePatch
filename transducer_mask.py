from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import SimulationConfig, get_config


@dataclass(frozen=True)
class ElementGeometry:
    """单个阵元的几何信息（物理坐标系，单位：米）。"""

    index: int
    center_xy: tuple[float, float]
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


def _build_rotation_matrix(angle_rad: float) -> np.ndarray:
    """构造 2x2 旋转矩阵。rotation 与 Kwave.py 保持相同单位（弧度）。"""
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64)


def _get_array_translation(cfg: SimulationConfig, rand_shift_grid: int = 0) -> np.ndarray:
    """复现 Kwave.py 中 set_array_position 使用的平移量。"""
    pml_size = cfg.PMLSize_base
    shift_x_meter = rand_shift_grid * cfg.dx
    grid_y_size = cfg.Ny * cfg.dy
    t_y = grid_y_size / 2 - (cfg.array_offset_y_grids + pml_size) * cfg.dy
    return np.array([shift_x_meter, -t_y], dtype=np.float64)


def build_array_geometry(cfg: SimulationConfig, rand_shift_grid: int = 0) -> list[ElementGeometry]:
    """
    根据 Kwave.py 中的 add_rect_element + set_array_position 逻辑重建阵元几何。

    注意：这里故意保留 `x_pos = x_start + (ind - 1) * cfg.element_pitch`
    的写法，以严格复现当前 `Kwave.py` 的放置方式。
    """
    rotation_matrix = _build_rotation_matrix(cfg.rotation)
    translation = _get_array_translation(cfg, rand_shift_grid=rand_shift_grid)

    array_width_total = (cfg.element_num - 1) * cfg.element_pitch
    x_start = -array_width_total / 2
    half_length = cfg.element_width / 2

    geometry: list[ElementGeometry] = []
    for ind in range(cfg.element_num):
        x_pos = x_start + (ind - 1) * cfg.element_pitch

        local_center = np.array([x_pos, 0.0], dtype=np.float64)
        local_start = np.array([x_pos - half_length, 0.0], dtype=np.float64)
        local_end = np.array([x_pos + half_length, 0.0], dtype=np.float64)

        center = rotation_matrix @ local_center + translation
        start = rotation_matrix @ local_start + translation
        end = rotation_matrix @ local_end + translation

        geometry.append(
            ElementGeometry(
                index=ind,
                center_xy=(float(center[0]), float(center[1])),
                start_xy=(float(start[0]), float(start[1])),
                end_xy=(float(end[0]), float(end[1])),
            )
        )

    return geometry


def kwave_xy_to_rc(x: float, y: float, cfg: SimulationConfig) -> tuple[float, float]:
    """
    将 k-Wave 物理坐标 (x, y) 映射到数组索引 (row, col)。

    - 物理坐标: x 正方向为图片下方，y 正方向为图片右方
    - ndarray: row 向下增大，col 向右增大
    """
    row = cfg.Nx // 2 + x / cfg.dx
    col = cfg.Ny // 2 + y / cfg.dy
    return float(row), float(col)


def _physical_axis_to_pixel_direction(start_xy: tuple[float, float], end_xy: tuple[float, float],
                                      cfg: SimulationConfig) -> np.ndarray:
    """把阵元长度方向从物理坐标转换到像素坐标方向。"""
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    physical_direction = end - start
    pixel_direction = np.array(
        [
            physical_direction[0] / cfg.dx,
            physical_direction[1] / cfg.dy,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(pixel_direction)
    if norm == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return pixel_direction / norm


def _rounded_length_in_pixels(start_xy: tuple[float, float], end_xy: tuple[float, float],
                              cfg: SimulationConfig) -> int:
    """按物理长度换算为像素长度，并执行四舍五入。"""
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    physical_direction = end - start
    pixel_span = np.array(
        [
            physical_direction[0] / cfg.dx,
            physical_direction[1] / cfg.dy,
        ],
        dtype=np.float64,
    )
    return max(1, int(np.rint(np.linalg.norm(pixel_span))))


def _bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Bresenham 线段栅格化，保证掩膜厚度为 1 像素。"""
    points: list[tuple[int, int]] = []
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    err = d_col - d_row

    row, col = row0, col0
    while True:
        points.append((row, col))
        if row == row1 and col == col1:
            break
        twice_err = 2 * err
        if twice_err > -d_row:
            err -= d_row
            col += step_col
        if twice_err < d_col:
            err += d_col
            row += step_row
    return points


def _rasterize_single_element(mask: np.ndarray, element: ElementGeometry, cfg: SimulationConfig) -> None:
    """把单个阵元栅格化到 mask 上。"""
    center_row, center_col = kwave_xy_to_rc(element.center_xy[0], element.center_xy[1], cfg)
    axis_direction = _physical_axis_to_pixel_direction(element.start_xy, element.end_xy, cfg)
    length_pixels = _rounded_length_in_pixels(element.start_xy, element.end_xy, cfg)

    if np.allclose(axis_direction, 0.0):
        row = int(np.rint(center_row))
        col = int(np.rint(center_col))
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            mask[row, col] = 1
        return

    half_span = (length_pixels - 1) / 2
    start_rc = np.array([center_row, center_col], dtype=np.float64) - axis_direction * half_span
    end_rc = np.array([center_row, center_col], dtype=np.float64) + axis_direction * half_span

    row0, col0 = np.rint(start_rc).astype(int)
    row1, col1 = np.rint(end_rc).astype(int)

    for row, col in _bresenham_line(row0, col0, row1, col1):
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            mask[row, col] = 1


def generate_transducer_mask(
    factor: float = 1.0,
    rand_shift_grid: int = 0,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    生成阵元二值掩膜。

    返回的 mask 满足：
    - 阵元厚度固定为 1 像素
    - 阵元长度按物理长度折算后的像素长度四舍五入
    - 阵元中心、平移、旋转与 Kwave.py 保持一致
    """
    cfg = get_config(factor=factor)
    mask = np.zeros((cfg.Nx, cfg.Ny), dtype=dtype)

    for element in build_array_geometry(cfg, rand_shift_grid=rand_shift_grid):
        _rasterize_single_element(mask, element, cfg)

    return mask


def overlay_mask_on_sos(
    sound_speed_map: np.ndarray,
    mask: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "Sound speed map with transducer mask",
) -> None:
    """把阵元掩膜叠加到声速图上，便于论文作图。"""
    if sound_speed_map.shape != mask.shape:
        raise ValueError(f"Shape mismatch: sound_speed_map={sound_speed_map.shape}, mask={mask.shape}")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sound_speed_map, cmap="viridis", origin="upper")
    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(
        overlay,
        cmap=ListedColormap(["#ff2d55"]),
        origin="upper",
        interpolation="nearest",
        alpha=0.95,
    )
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax, label="SoS [m/s]")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_mask(mask: np.ndarray, save_path: str | Path) -> None:
    """保存掩膜为 .npy 文件。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, mask)


def _summarize_mask(mask: np.ndarray, geometry: list[ElementGeometry], cfg: SimulationConfig) -> str:
    """生成简要文本摘要，便于命令行快速检查。"""
    centers_rc = [kwave_xy_to_rc(element.center_xy[0], element.center_xy[1], cfg) for element in geometry]
    center_rows = [row for row, _ in centers_rc]
    center_cols = [col for _, col in centers_rc]
    return (
        f"mask shape={mask.shape}, foreground_pixels={int(mask.sum())}, "
        f"element_num={cfg.element_num}, rounded_length_px={_rounded_length_in_pixels(geometry[0].start_xy, geometry[0].end_xy, cfg)}, "
        f"center_row_range=({min(center_rows):.2f}, {max(center_rows):.2f}), "
        f"center_col_range=({min(center_cols):.2f}, {max(center_cols):.2f})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成单像素厚度的换能器阵元二值掩膜")
    parser.add_argument("--factor", type=float, default=1.0, help="配置缩放因子，默认 1.0")
    parser.add_argument("--rand-shift-grid", type=int, default=0, help="沿 x 方向的随机平移（单位：grid）")
    parser.add_argument("--save-mask", type=str, default=None, help="保存 .npy 掩膜的路径")
    parser.add_argument("--sos-map", type=str, default=None, help="可选：用于叠加显示的声速图 .npy 路径")
    parser.add_argument("--save-figure", type=str, default=None, help="可选：叠加图保存路径（如 .png）")
    parser.add_argument("--show", action="store_true", help="是否显示图像窗口")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = get_config(factor=args.factor)
    geometry = build_array_geometry(cfg, rand_shift_grid=args.rand_shift_grid)
    mask = generate_transducer_mask(factor=args.factor, rand_shift_grid=args.rand_shift_grid)

    print(_summarize_mask(mask, geometry, cfg))

    if args.save_mask:
        save_mask(mask, args.save_mask)
        print(f"Mask saved to: {args.save_mask}")

    if args.sos_map:
        sound_speed_map = np.load(args.sos_map)
        overlay_mask_on_sos(
            sound_speed_map=sound_speed_map,
            mask=mask,
            save_path=args.save_figure,
            show=args.show,
            title=f"Transducer mask overlay (shift={args.rand_shift_grid} grid)",
        )
        if args.save_figure:
            print(f"Overlay figure saved to: {args.save_figure}")


if __name__ == "__main__":
    main()

