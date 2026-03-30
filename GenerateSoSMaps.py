import os
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from kwave.utils.filters import sharpness
from scipy.ndimage import gaussian_filter


from GRF_KL import generate_grf
from config import get_config


# =========================
#  主函数
# =========================
def generate_sos_maps(
    output_dir: str = "./SoSMap",
    data_num: int = 10,
    start_num: int = 0,
    length_scale_bg: float = 3e-3,
    length_scale_inc: float = 2e-3,
    plot_samples: bool = False,
    use_single: bool = True,
    noise_level: float = 0.05,
    ellipses_range: Tuple[int, int] = (1, 2),
    pool_size: Optional[int] = None,
    prob_no_inclusion: float = 0.1,
    sharpness: float = 3.5,
    texture_strength: float = 0.2,
):
    """
    Python 版本：GenerateSoSMaps
    """
    print("=== 数据集生成开始 ===")
    print(f"保存目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 1) 检查存在性
    print("正在检查文件存在性...")
    all_requested_indices = list(range(start_num, start_num + data_num))
    todo_indices = []
    for idx in all_requested_indices:
        fname = os.path.join(output_dir, f"sample_{idx:06d}.npy")
        if not os.path.exists(fname):
            todo_indices.append(idx)

    num_todo = len(todo_indices)
    if num_todo == 0:
        print(f"所有 {data_num} 个样本文件已存在，跳过生成。")
        return
    elif num_todo < data_num:
        print(f"检测到 {data_num - num_todo} 个样本已存在。实际将生成 {num_todo} 个新样本。")
    else:
        print(f"将生成全部 {num_todo} 个样本。")

    # 2) 加载配置
    try:
        cfg = get_config()
    except Exception as exc:
        raise RuntimeError(f"无法调用 get_config()。错误: {exc}")

    HR_size = (cfg.Nx, cfg.Ny)
    speed_range = (cfg.minSoS, cfg.maxOtherSoS)
    speed_range_inclusion = (cfg.minInclusionSoS, cfg.maxSoS)
    grid_spacing = cfg.dx

    # 3) 生成掩膜（并行）
    print("=== 步骤 1: 并行生成掩膜 ===")
    all_masks = np.zeros((num_todo, HR_size[0], HR_size[1]), dtype=np.float32 if use_single else np.float64)

    with ProcessPoolExecutor(max_workers=pool_size) as ex:
        futures = []
        for _ in range(num_todo):
            futures.append(ex.submit(
                generate_inclusion_mask,
                HR_size,
                (0.2, 0.8, 0.2, 0.8),
                use_single,
                prob_no_inclusion,
                ellipses_range,
                noise_level,
            ))

        for i, fut in enumerate(as_completed(futures)):
            all_masks[i, :, :] = fut.result()

    # 4) 生成随机场
    print("=== 步骤 2: 批量生成随机场 ===")
    all_bg_fields, all_inc_fields = generate_batch_fields(
        HR_size,
        num_todo,
        speed_range,
        speed_range_inclusion,
        grid_spacing,
        length_scale_bg,
        length_scale_inc,
        use_single,
        sharpness=sharpness,
        texture_strength=texture_strength,
    )

    # 5) 合成与保存（并行）
    print("=== 步骤 3: 并行合成与保存 ===")
    with ProcessPoolExecutor(max_workers=pool_size) as ex:
        futures = []
        for i in range(num_todo):
            real_sample_idx = todo_indices[i]
            mask = all_masks[i, :, :]
            bg = all_bg_fields[i, :, :]
            inc = all_inc_fields[i, :, :]
            futures.append(ex.submit(
                save_sample_simple_par,
                output_dir,
                real_sample_idx,
                mask,
                bg,
                inc,
                (speed_range[0], speed_range_inclusion[1]),
                plot_samples,
            ))

        for i, fut in enumerate(as_completed(futures), start=1):
            fut.result()
            if i % 10 == 0:
                print(f"已完成 {i}/{num_todo}")

    print("数据集生成完成！")


# =========================
#  随机场生成
# =========================
def generate_batch_fields(
    img_size: Tuple[int, int],
    n_samples: int,
    speed_range: Tuple[float, float],
    speed_range_inclusion: Tuple[float, float],
    grid_spacing: float,
    phy_length_scale: float,
    phy_length_scale_inc: float,
    use_single: bool,
    sharpness: float,
    texture_strength: float,
):
    try:
        print(f"正在调用 GRF 生成 {n_samples} 个样本...")
        bg_batch = generate_grf(
            img_size=img_size,
            num_fields=n_samples,
            speed_range=speed_range,
            physical_length_scale=phy_length_scale,
            grid_spacing=grid_spacing,
            plot=False,
            sharpness=sharpness,
            texture_strength=texture_strength,
        )
        inc_batch = generate_grf(
            img_size=img_size,
            num_fields=n_samples,
            speed_range=speed_range_inclusion,
            physical_length_scale=phy_length_scale_inc,
            grid_spacing=grid_spacing,
            plot=False,
            sharpness=sharpness,
            texture_strength=texture_strength,
        )
        if use_single:
            bg_batch = bg_batch.astype(np.float32)
            inc_batch = inc_batch.astype(np.float32)
        return bg_batch, inc_batch
    except Exception as exc:
        print(f"随机场生成出错！错误信息: {exc}")
        raise RuntimeError("请检查 generate_grf 函数逻辑") from exc


# =========================
#  保存函数
# =========================
def save_sample_simple_par(
    output_dir: str,
    sample_idx: int,
    mask: np.ndarray,
    bg: np.ndarray,
    inc: np.ndarray,
    speed_range: Tuple[float, float],
    plot_flag: bool,
):
    sound_speed_map = bg * (1.0 - mask) + inc * mask
    filename = os.path.join(output_dir, f"sample_{sample_idx:06d}.npy")
    np.save(filename, sound_speed_map)

    if plot_flag and sample_idx <= 10:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(sound_speed_map, origin="lower")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)
        im.set_clim(speed_range[0], speed_range[1])
        ax.set_title(f"Sample {sample_idx}")
        fig.savefig(os.path.join(output_dir, f"sample_{sample_idx:06d}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


# =========================
#  掩膜生成
# =========================
def generate_inclusion_mask(
    img_size: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    use_single: bool,
    prob_no_inc: float,
    num_ellipses_rng: Tuple[int, int],
    noise_lvl: float,
):
    dtype = np.float32 if use_single else np.float64
    inclusion_mask = np.zeros(img_size, dtype=dtype)

    if np.random.rand() < prob_no_inc:
        return inclusion_mask

    num_ellipses = np.random.randint(num_ellipses_rng[0], num_ellipses_rng[1] + 1)

    x = np.linspace(0, 1, img_size[1])
    y = np.linspace(0, 1, img_size[0])
    X, Y = np.meshgrid(x, y)

    bound_width = bounds[1] - bounds[0]
    bound_height = bounds[3] - bounds[2]
    max_radius = min(bound_width, bound_height) * 0.4

    for _ in range(num_ellipses):
        center_x = np.random.rand() * bound_width + bounds[0]
        center_y = np.random.rand() * bound_height + bounds[2]

        radius_x = np.random.rand() * max_radius * 0.3 + max_radius * 0.1
        radius_y = np.random.rand() * max_radius * 0.3 + max_radius * 0.1

        radius_x = min(radius_x, center_x - bounds[0], bounds[1] - center_x)
        radius_y = min(radius_y, center_y - bounds[2], bounds[3] - center_y)

        rotation = np.random.rand() * math.pi

        X_rot = (X - center_x) * math.cos(rotation) + (Y - center_y) * math.sin(rotation)
        Y_rot = -(X - center_x) * math.sin(rotation) + (Y - center_y) * math.cos(rotation)

        single_ellipse = (X_rot / radius_x) ** 2 + (Y_rot / radius_y) ** 2 <= 1
        inclusion_mask = np.maximum(inclusion_mask, single_ellipse.astype(dtype))

    inclusion_mask = gaussian_filter(inclusion_mask, sigma=1)
    max_val = inclusion_mask.max()
    if max_val > 0:
        inclusion_mask = inclusion_mask / max_val

    inclusion_mask = inclusion_mask + noise_lvl * np.random.randn(*img_size).astype(dtype) * inclusion_mask
    inclusion_mask = np.clip(inclusion_mask, 0, 1)

    return inclusion_mask


if __name__ == "__main__":
    generate_sos_maps(data_num=1,plot_samples=True)