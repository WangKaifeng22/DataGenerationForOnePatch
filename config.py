import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """k-Wave仿真配置参数管理类"""

    # --- 1. 网格与全局控制 ---
    factor: float = 1.0  # 缩放因子
    PMLSize_base: int = 20  # 基准 PML 大小
    PMLAlpha_default: float = 2.5  # 默认 PML 吸收系数
    dt = 1.8182e-8
    Nt = 1900

    # 计算得到的网格参数（属性）
    @property
    def Nx(self) -> int:
        return int(256 * self.factor)

    @property
    def Ny(self) -> int:
        return int(256 * self.factor)

    @property
    def dx(self) -> float:
        return 0.1e-3 / self.factor

    @property
    def dy(self) -> float:
        return 0.1e-3 / self.factor

    # --- 2. 介质参数 (Medium) ---
    rho0: np.float32 = np.float32(1000)  # 水的密度 [kg/m^3]
    alpha_coeff: np.float32 = np.float32(0.75)  # [dB/(MHz^y cm)]
    alpha_power: np.float32 = np.float32(1.5)
    BonA: np.float32 = np.float32(6.0)

    # 声速范围参数
    minSoS: float = 1430.0  # 其它区域最低声速
    maxSoS: float = 1650.0  # 包涵体最高声速
    minInclusionSoS: float = 1520.0  # 包涵体最低声速
    maxOtherSoS: float = 1600.0  # 其它区域最高声速

    # --- 3. 换能器阵列 (Array) ---
    element_num: int = 32  # 阵元数量
    element_pitch: float = 3e-4  # 阵元间距 [m]
    element_width: float = 2.74e-4  # 阵元宽度 [m]
    upsampling_rate: int = 100  # kArray 上采样率
    bli_tolerance: float = 0.035  # kArray 容差

    # 阵列位置偏移
    array_offset_y_grids: float = 15.0  # 初始值，会在__post_init__中乘以factor
    rotation: float = 0.0  # 旋转角度

    # --- 4. 信号源 (Source) ---
    source_amp: float = 2.5e5
    source_f0: float = 3e6  # 中心频率 [Hz]
    source_cycles: int = 4  # 周期数
    transmitting_ind: int = 0  # 发射阵元索引
    cfl: float = 0.3  # CFL条件数

    # --- 5. 仿真时间控制 ---
    t_end_scaler: float = 1.8  # 时间长度倍数

    # --- 6. 散射体几何 ---
    radius: float = 15.0
    scat_radius_grid: Optional[float] = None  # 会在__post_init__中计算

    def __post_init__(self):
        """初始化后计算依赖参数"""
        self.array_offset_y_grids *= self.factor
        self.scat_radius_grid = self.radius * self.factor

    def get_grid_size(self):
        """获取网格尺寸"""
        return self.Nx, self.Ny

    def get_grid_spacing(self):
        """获取网格间距"""
        return self.dx, self.dy

    def get_speed_of_sound_range(self):
        """获取声速范围"""
        return {
            'minSoS': self.minSoS,
            'maxSoS': self.maxSoS,
            'minInclusionSoS': self.minInclusionSoS,
            'maxOtherSoS': self.maxOtherSoS
        }

    def get_array_params(self):
        """获取换能器阵列参数"""
        return {
            'element_num': self.element_num,
            'element_pitch': self.element_pitch,
            'element_width': self.element_width,
            'upsampling_rate': self.upsampling_rate,
            'bli_tolerance': self.bli_tolerance,
            'array_offset_y_grids': self.array_offset_y_grids,
            'rotation': self.rotation
        }

    def get_source_params(self):
        """获取信号源参数"""
        return {
            'source_amp': self.source_amp,
            'source_f0': self.source_f0,
            'source_cycles': self.source_cycles,
            'transmitting_ind': self.transmitting_ind,
            'cfl': self.cfl
        }

    def to_dict(self):
        """将配置转换为字典"""
        return {
            'factor': self.factor,
            'PMLSize_base': self.PMLSize_base,
            'PMLAlpha_default': self.PMLAlpha_default,
            'Nx': self.Nx,
            'Ny': self.Ny,
            'dx': self.dx,
            'dy': self.dy,
            'rho0': self.rho0,
            'alpha_coeff': self.alpha_coeff,
            'alpha_power': self.alpha_power,
            'BonA': self.BonA,
            'minSoS': self.minSoS,
            'maxSoS': self.maxSoS,
            'minInclusionSoS': self.minInclusionSoS,
            'maxOtherSoS': self.maxOtherSoS,
            'element_num': self.element_num,
            'element_pitch': self.element_pitch,
            'element_width': self.element_width,
            'upsampling_rate': self.upsampling_rate,
            'bli_tolerance': self.bli_tolerance,
            'array_offset_y_grids': self.array_offset_y_grids,
            'rotation': self.rotation,
            'source_amp': self.source_amp,
            'source_f0': self.source_f0,
            'source_cycles': self.source_cycles,
            'transmitting_ind': self.transmitting_ind,
            'cfl': self.cfl,
            't_end_scaler': self.t_end_scaler,
            'radius': self.radius,
            'scat_radius_grid': self.scat_radius_grid
        }


def get_config(factor: float = 1.0) -> SimulationConfig:
    """
    获取k-Wave仿真配置

    参数:
        factor: 缩放因子，默认为1.0

    返回:
        SimulationConfig: 配置对象
    """
    return SimulationConfig(factor=factor)


# 为了保持与MATLAB接口的兼容性，也可以提供简单的函数版本
def get_config_simple(factor: float = 1.0) -> dict:
    """
    简化的配置获取函数（返回字典，与MATLAB结构体类似）
    """
    cfg = SimulationConfig(factor=factor)
    return cfg.to_dict()


if __name__ == "__main__":
    # 示例用法
    cfg = get_config(factor=2.0)

    print("=== 仿真配置 ===")
    print(f"网格尺寸: {cfg.get_grid_size()}")
    print(f"网格间距: {cfg.get_grid_spacing()}")
    print(f"声速范围: {cfg.get_speed_of_sound_range()}")
    print(f"阵列参数: {cfg.get_array_params()}")
    print(f"信号源参数: {cfg.get_source_params()}")

    # 使用字典形式
    cfg_dict = get_config_simple(factor=1.0)
    print(f"\n配置字典: {cfg_dict}")