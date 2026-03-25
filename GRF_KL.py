import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sys


def rbf_kernel_1d(x1, x2, length_scale=1.0, variance=1.0):
    """计算1D RBF核矩阵"""
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * sqdist / length_scale ** 2)


def generate_2d_grf_via_kl(size_x, size_y, num_fields, length_scale_relative, variance=1.0, num_modes=100):
    """
    基于KL展开的2D高斯随机场生成

    参数:
        length_scale_relative: 相对长度尺度。
                               如果你原图物理尺寸是 L，相关长度是 l，这里填 l/L。
                               例如：原图宽 0.0384米，length_scale是0.01米，这里填 0.01/0.0384 ≈ 0.26
    """
    # 1. 建立归一化网格 [0, 1]
    grid_x = np.linspace(0, 1, size_x).reshape(-1, 1)
    grid_y = np.linspace(0, 1, size_y).reshape(-1, 1)

    # 2. 计算核矩阵 (只需计算1D)
    K_x = rbf_kernel_1d(grid_x, grid_x, length_scale=length_scale_relative, variance=variance)
    K_y = rbf_kernel_1d(grid_y, grid_y, length_scale=length_scale_relative, variance=variance)

    # 3. 特征值分解
    # 注意：如果 size 很大 (如 >1000)，这里可能会慢。但对于 384x384 很快。
    evals_x, evecs_x = eigh(K_x)
    evals_y, evecs_y = eigh(K_y)

    # 4. 截断模态 (取最大的 num_modes 个)
    idx_x = np.argsort(evals_x)[::-1][:num_modes]
    idx_y = np.argsort(evals_y)[::-1][:num_modes]

    evals_x, evecs_x = evals_x[idx_x], evecs_x[:, idx_x]
    evals_y, evecs_y = evals_y[idx_y], evecs_y[:, idx_y]

    # 5. 准备生成
    fields = np.zeros((num_fields, size_x, size_y), dtype=np.float64)
    # 2D特征值权重矩阵 sqrt(lambda_x * lambda_y)
    sqrt_eig_2d = np.sqrt(np.outer(evals_x, evals_y))

    for n in range(num_fields):
        # 生成标准正态随机数
        xi = np.random.normal(0, 1, (num_modes, num_modes))

        # 核心计算：Field = Evec_x @ (Weight * Random) @ Evec_y.T
        coeffs = sqrt_eig_2d * xi
        field = evecs_x @ coeffs @ evecs_y.T

        fields[n, :, :] = field

    return fields


def normalize_fields(fields, target_min, target_max):
    """
    将场线性映射到指定范围 [target_min, target_max]
    """
    fields_normalized = np.zeros_like(fields)
    for i in range(fields.shape[0]):
        field = fields[i]
        f_min, f_max = np.min(field), np.max(field)

        # 防止除以0（虽然在高斯场中极少见）
        if f_max == f_min:
            fields_normalized[i] = np.full_like(field, (target_min + target_max) / 2)
        else:
            fields_normalized[i] = (field - f_min) / (f_max - f_min) * (target_max - target_min) + target_min

    return fields_normalized


def generate_grf(img_size=(256, 256), num_fields=1, speed_range=(1430, 1600),
                 sharpness=3.5, texture_strength=0.2,
                 physical_length_scale=4e-3, grid_spacing=1e-4, plot = False):
    # --- 参数设置 ---
    size_x, size_y = img_size

    # ！！！关键转换！！！
    # KL方法通常在 [0,1] 域上计算。
    # 假设你的物理 grid_spacing = 1e-4 (0.1mm)
    # 物理总宽度 L = 384 * 1e-4 = 0.0384 米
    # 你想要的物理 length_scale = 10e-3 = 0.01 米
    # 那么相对 length_scale = 0.01 / 0.0384 ≈ 0.26
    physical_L = size_x * grid_spacing

    relative_l_scale = physical_length_scale / physical_L

    #print(f"物理相关长度: {physical_length_scale}m, 相对相关长度: {relative_l_scale:.4f} (相对于图像边长)")

    # 1. 生成原始高斯场 (KL方法)
    # num_modes 设为 100-200 足够满足平滑的 RBF 核
    raw_fields = generate_2d_grf_via_kl(
        size_x, size_y,
        num_fields=num_fields,
        length_scale_relative=relative_l_scale,
        variance=1.0,
        num_modes=128
    )

    if sharpness > 0:
        # A. 结构层：使用 tanh 制造明显的“相分离”边界
        # raw_fields 约为 N(0,1)，sharpness*raw 放大后过 tanh 会趋向于 -1 和 1
        structure_part = np.tanh(sharpness * raw_fields)

        # B. 纹理层：保留一部分原始随机变化
        # 如果只用 A，组织内部是纯平的颜色；加上 B，内部会有自然的斑点纹理
        combined_fields = structure_part + (texture_strength * raw_fields)
    else:
        combined_fields = raw_fields

    # 2. 映射到指定速度范围 (Min-Max Normalization)
    fields_controlled = normalize_fields(combined_fields, speed_range[0], speed_range[1])
    fields_controlled = fields_controlled.astype(np.float32)
    # 3. 可视化检查
    if plot:
        for i in range(num_fields):
            plt.figure(figsize=(5, 4))
            plt.imshow(fields_controlled[i], extent=[0, size_x, 0, size_y], origin='lower')
            plt.title(
                f'KL-Generated Field {i + 1}\nRange: [{np.min(fields_controlled[i]):.1f}, {np.max(fields_controlled[i]):.1f}]')
            plt.colorbar(label='Speed (m/s)')
            plt.show()

    return fields_controlled

if __name__ == "__main__":
    print(generate_grf(plot = True))

    """if len(sys.argv) < 10:
        print("Error: Not enough arguments.")
        result = np.zeros((1, 10, 10))
        inclusion = np.zeros((1, 10, 10))
    else:
        try:
            # 读取原有参数
            arg_inc_min = int(sys.argv[1])
            arg_inc_max = int(sys.argv[2])
            arg_bg_min = int(sys.argv[3])
            arg_bg_max = int(sys.argv[4])
            arg_nx = int(sys.argv[5])
            arg_ny = int(sys.argv[6])
            arg_grid = float(sys.argv[7])
            arg_len = float(sys.argv[8])

            arg_num_samples = int(sys.argv[9])
            arg_len_inclusion = float(sys.argv[10])

            img_size_tuple = (arg_nx, arg_ny)

            # 调用 main，传入 num_fields = arg_num_samples
            # 这会一次性生成 N 个场，返回 shape 为 (N, H, W) 的数组
            result = main(img_size=img_size_tuple,
                          num_fields=arg_num_samples,
                          speed_range=(arg_bg_min, arg_bg_max),
                          physical_length_scale=arg_len,
                          grid_spacing=arg_grid)

            inclusion = main(img_size=img_size_tuple,
                             num_fields=arg_num_samples,
                             speed_range=(arg_inc_min, arg_inc_max),
                             physical_length_scale=arg_len_inclusion,
                             grid_spacing=arg_grid)

        except Exception as e:
            print(f"Python Error: {e}")
            result = np.zeros((1, 10, 10))
            inclusion = np.zeros((1, 10, 10))"""