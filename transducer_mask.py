from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import SimulationConfig, get_config


@dataclass(frozen=True)
class ElementGeometry:
    """单个阵元的几何信息（物理坐标系，单位：米）。"""

    index: int
    center_xy: tuple[float, float]
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


@dataclass(frozen=True)
class MaskGridConfig:
    """用于掩膜栅格化的最小网格参数集合。"""

    Nx: int
    Ny: int
    dx: float
    dy: float


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


def kwave_xy_to_rc(x: float, y: float, cfg: SimulationConfig | MaskGridConfig) -> tuple[float, float]:
    """
    将 k-Wave 物理坐标 (x, y) 映射到数组索引 (row, col)。

    - 物理坐标: x 正方向为图片下方，y 正方向为图片右方
    - ndarray: row 向下增大，col 向右增大
    """
    row = cfg.Nx // 2 + x / cfg.dx
    col = cfg.Ny // 2 + y / cfg.dy
    return float(row), float(col)


def _physical_axis_to_pixel_direction(start_xy: tuple[float, float], end_xy: tuple[float, float],
                                      cfg: SimulationConfig | MaskGridConfig) -> np.ndarray:
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
                              cfg: SimulationConfig | MaskGridConfig) -> int:
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


def _paint_disk(mask: np.ndarray, center_row: int, center_col: int, radius: float) -> None:
    """在 mask 上绘制圆盘，半径单位为像素。"""
    if radius <= 0:
        if 0 <= center_row < mask.shape[0] and 0 <= center_col < mask.shape[1]:
            mask[center_row, center_col] = 1
        return

    r = int(np.ceil(radius))
    r2 = radius * radius
    row_min = max(0, center_row - r)
    row_max = min(mask.shape[0] - 1, center_row + r)
    col_min = max(0, center_col - r)
    col_max = min(mask.shape[1] - 1, center_col + r)
    for row in range(row_min, row_max + 1):
        dr2 = (row - center_row) ** 2
        for col in range(col_min, col_max + 1):
            if dr2 + (col - center_col) ** 2 <= r2:
                mask[row, col] = 1


def _rasterize_single_element(
    mask: np.ndarray,
    element: ElementGeometry,
    cfg: SimulationConfig | MaskGridConfig,
    line_thickness_px: int = 1,
) -> None:
    """把单个阵元栅格化到 mask 上。"""
    center_row, center_col = kwave_xy_to_rc(element.center_xy[0], element.center_xy[1], cfg)
    axis_direction = _physical_axis_to_pixel_direction(element.start_xy, element.end_xy, cfg)
    length_pixels = _rounded_length_in_pixels(element.start_xy, element.end_xy, cfg)

    if np.allclose(axis_direction, 0.0):
        row = int(np.rint(center_row))
        col = int(np.rint(center_col))
        _paint_disk(mask, row, col, radius=(line_thickness_px - 1) / 2)
        return

    half_span = (length_pixels - 1) / 2
    start_rc = np.array([center_row, center_col], dtype=np.float64) - axis_direction * half_span
    end_rc = np.array([center_row, center_col], dtype=np.float64) + axis_direction * half_span

    row0, col0 = np.rint(start_rc).astype(int)
    row1, col1 = np.rint(end_rc).astype(int)

    for row, col in _bresenham_line(row0, col0, row1, col1):
        _paint_disk(mask, row, col, radius=(line_thickness_px - 1) / 2)


def _build_mask_grid(cfg: SimulationConfig, oversample: int) -> MaskGridConfig:
    """构造用于掩膜绘制的网格，保持物理尺寸不变。"""
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    return MaskGridConfig(
        Nx=cfg.Nx * oversample,
        Ny=cfg.Ny * oversample,
        dx=cfg.dx / oversample,
        dy=cfg.dy / oversample,
    )


def _area_downsample_to_shape(array_2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """用面积平均把 2D 数组降采样到目标尺寸。"""
    src_rows, src_cols = array_2d.shape
    dst_rows, dst_cols = target_shape
    if src_rows % dst_rows != 0 or src_cols % dst_cols != 0:
        raise ValueError(
            f"Cannot area-downsample from {array_2d.shape} to {target_shape}: non-integer scale ratio"
        )

    scale_r = src_rows // dst_rows
    scale_c = src_cols // dst_cols
    if scale_r < 1 or scale_c < 1:
        raise ValueError(
            f"Cannot area-downsample from {array_2d.shape} to {target_shape}: target is larger than source"
        )

    if scale_r == 1 and scale_c == 1:
        return array_2d.astype(np.float64, copy=False)

    reshaped = array_2d.reshape(dst_rows, scale_r, dst_cols, scale_c)
    return reshaped.mean(axis=(1, 3), dtype=np.float64)


def _to_binary_mask(mask: np.ndarray, target_shape: tuple[int, int], mask_threshold: float) -> np.ndarray:
    """将任意分辨率掩膜转换为目标尺寸的二值掩膜。"""
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError(f"mask_threshold must be in [0, 1], got {mask_threshold}")

    mask_float = np.asarray(mask, dtype=np.float64)
    if mask_float.shape != target_shape:
        mask_float = _area_downsample_to_shape(mask_float, target_shape)
    return (mask_float >= mask_threshold).astype(np.uint8)


def generate_transducer_mask(
    factor: float = 1.0,
    rand_shift_grid: int = 0,
    oversample: int = 8,
    mask_threshold: float = 0.5,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    生成阵元二值掩膜。

    生成流程：
    - 先在 oversample 倍高分辨率网格中栅格化阵元
    - 再面积平均回采样到 SoS 网格
    - 最后按 mask_threshold 二值化，得到清晰边界掩膜
    """
    cfg = get_config(factor=factor)
    grid_cfg = _build_mask_grid(cfg, oversample=oversample)
    mask_hr = np.zeros((grid_cfg.Nx, grid_cfg.Ny), dtype=np.uint8)

    for element in build_array_geometry(cfg, rand_shift_grid=rand_shift_grid):
        _rasterize_single_element(mask_hr, element, grid_cfg, line_thickness_px=oversample)

    mask_lr = _to_binary_mask(mask_hr, target_shape=(cfg.Nx, cfg.Ny), mask_threshold=mask_threshold)
    return mask_lr.astype(dtype, copy=False)


def overlay_mask_on_sos(
    sound_speed_map: np.ndarray,
    mask: np.ndarray,
    mask_threshold: float = 0.5,
    mm_per_pixel: float = 1.0,
    origin: str = "lower",
    axis_unit: str = "mm",
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "Sound speed map with transducer mask",
) -> None:
    """把阵元掩膜叠加到声速图上，便于论文作图。

    坐标范围按 mm_per_pixel 转换为物理坐标，范围与 my_test.py 保持一致：
    [0, Ny * mm_per_pixel] x [0, Nx * mm_per_pixel]。
    """
    binary_mask = _to_binary_mask(mask, target_shape=sound_speed_map.shape, mask_threshold=mask_threshold)

    if mm_per_pixel <= 0:
        raise ValueError(f"mm_per_pixel must be > 0, got {mm_per_pixel}")
    if origin not in {"lower", "upper"}:
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin}")

    rows, cols = sound_speed_map.shape
    extent = [0.0, cols * mm_per_pixel, 0.0, rows * mm_per_pixel]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sound_speed_map, cmap="viridis", origin=origin, extent=extent, aspect="equal")
    overlay = np.ma.masked_where(binary_mask == 0, binary_mask)
    ax.imshow(
        overlay,
        cmap=ListedColormap(["#ff2d55"]),
        origin=origin,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
        aspect="equal",
    )
    #ax.set_title(title)
    ax.set_xlabel(f"Y ({axis_unit})", fontsize=14)
    ax.set_ylabel(f"X ({axis_unit})", fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("SoS [m/s]", fontsize=14)
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
    parser.add_argument("--mask-oversample", type=int, default=8, help="掩膜超采样倍数，默认 8")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="回采样后二值化阈值，默认 0.5")
    parser.add_argument("--save-mask", type=str, default=None, help="保存 .npy 掩膜的路径")
    parser.add_argument("--sos-map", type=str, default=None, help="可选：用于叠加显示的声速图 .npy 路径")
    parser.add_argument("--mm-per-pixel", type=float, default=0.1, help="物理坐标缩放（每像素毫米数），与 my_test.py 一致")
    parser.add_argument("--origin", type=str, default="lower", choices=["lower", "upper"], help="imshow 原点方向，默认 lower")
    parser.add_argument("--axis-unit", type=str, default="mm", help="坐标轴物理单位标签，默认 mm")
    parser.add_argument("--save-figure", type=str, default=None, help="可选：叠加图保存路径（如 .png）")
    parser.add_argument("--show", action="store_true", help="是否显示图像窗口")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = get_config(factor=args.factor)
    geometry = build_array_geometry(cfg, rand_shift_grid=args.rand_shift_grid)
    mask = generate_transducer_mask(
        factor=args.factor,
        rand_shift_grid=args.rand_shift_grid,
        oversample=args.mask_oversample,
        mask_threshold=args.mask_threshold,
    )

    print(_summarize_mask(mask, geometry, cfg))

    if args.save_mask:
        save_mask(mask, args.save_mask)
        print(f"Mask saved to: {args.save_mask}")

    if args.sos_map:
        sound_speed_map = np.load(args.sos_map)
        overlay_mask_on_sos(
            sound_speed_map=sound_speed_map,
            mask=mask,
            mask_threshold=args.mask_threshold,
            mm_per_pixel=args.mm_per_pixel,
            origin=args.origin,
            axis_unit=args.axis_unit,
            save_path=args.save_figure,
            show=args.show,
            title=f"Transducer mask overlay (shift={args.rand_shift_grid} grid)",
        )
        if args.save_figure:
            print(f"Overlay figure saved to: {args.save_figure}")


if __name__ == "__main__":
    main()

