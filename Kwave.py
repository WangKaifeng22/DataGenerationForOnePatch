import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
import random
import os
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D, kspace_first_order_2d_gpu
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kspaceLineRecon import kspaceLineRecon
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.plot import voxel_plot

from config import get_config


def per_SoSMap_Kwave(use_single: bool = True, worker_temp_dir: str = None, rand_shift_grid: int = 0,
                     sample_idx: int = 0, dataset_dir: str = "./SoSMap",
                     save_dir: str = "./KwaveResult", use_cpu=False):
    # --------------------
    # SIMULATION
    # --------------------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_{sample_idx:06d}.npz")

    cfg = get_config(factor=1.0)

    # create the computational grid
    PML_size = cfg.PMLSize_base  # size of the PML in grid points
    N = Vector([cfg.Nx, cfg.Ny])  # number of grid points
    d = Vector([cfg.dx, cfg.dy])  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    data_path = f"{dataset_dir}/sample_{sample_idx:06d}.npy"
    sos_map = np.load(data_path)
    medium = kWaveMedium(
        sound_speed=sos_map,
        density=cfg.rho0,
        alpha_coeff=cfg.alpha_coeff,  # dB/(MHz^alpha_power·cm)
        alpha_power=cfg.alpha_power,  # typical for soft tissue
        BonA=cfg.BonA
    )  # [m/s]

    # create the time array
    kgrid.Nt = cfg.Nt
    kgrid.dt = cfg.dt

    # set the input arguments: force the PML to be outside the computational grid
    
    if use_cpu:
        num_threads = 1
        kwave_function_name = None
        is_gpu = False
    else:
        num_threads = 1
        kwave_function_name = "kspaceFirstOrder2DG"  
        is_gpu = True  

    simulation_options = SimulationOptions(
        output_filename="my_acoustic_results.h5",  # ← 仿真结果将保存为此文件
        input_filename="simulation_input.h5",  # 输入数据保存名（需配合 save_to_disk=True）
        data_path=worker_temp_dir,  # 相对路径基准目录
        save_to_disk=True,
        pml_inside=True,
        pml_size=PML_size,
        pml_alpha=cfg.PMLAlpha_default,
        data_cast="single",
    )

    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=is_gpu, binary_name=None, num_threads=num_threads,
        show_sim_log=False, kwave_function_name=kwave_function_name)

    shift_x_meter = rand_shift_grid * cfg.dx

    karray = kWaveArray(bli_tolerance=cfg.bli_tolerance, upsampling_rate=cfg.upsampling_rate)
    t_y = kgrid.y_size / 2 - (cfg.array_offset_y_grids + PML_size) * kgrid.dy
    translation = [shift_x_meter, -t_y]

    array_width_total = (cfg.element_num - 1) * cfg.element_pitch
    x_start = -array_width_total / 2

    for ind in range(cfg.element_num):
        x_pos = x_start + (ind - 1) * cfg.element_pitch
        karray.add_rect_element([x_pos, 0], cfg.element_width, kgrid.dy, cfg.rotation)

    karray.set_array_position(translation, cfg.rotation)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)

    sensor_mask = karray.get_array_binary_mask(kgrid)
    sensor = kSensor(sensor_mask, record=["p"])

    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # axes[0].imshow(sensor_mask, origin='lower', cmap='gray')

    # define the input signal
    source_sig = cfg.source_amp * tone_burst(1 / kgrid.dt, cfg.source_f0, cfg.source_cycles, signal_length=kgrid.Nt)

    if use_single:
        sensor_coords = karray.get_element_positions().astype(np.float32)
        time_data_cat = np.zeros((cfg.element_num, cfg.element_num, kgrid.Nt), dtype=np.float32)
    else:
        sensor_coords = karray.get_element_positions().astype(np.float64)
        time_data_cat = np.zeros((cfg.element_num, cfg.element_num, kgrid.Nt), dtype=np.float64)

    for ind in range(0, cfg.element_num):
        source_sig_array = np.zeros((cfg.element_num, kgrid.Nt))
        source_sig_array[ind, :] = source_sig

        source.p = karray.get_distributed_source_signal(kgrid, source_sig_array, order="C")

        if use_cpu:
            sensor_data = kspaceFirstOrder(
                kgrid=kgrid, medium=medium, source=source, sensor=sensor,
                data_path=worker_temp_dir,  # 相对路径基准目录
                pml_inside=True,
                pml_size=PML_size,
                pml_alpha=cfg.PMLAlpha_default,
                use_sg=True,
                backend="python",
                device="cpu"
            )
        else:
            sensor_data = kspace_first_order_2d_gpu(
                kgrid=kgrid, medium=medium, source=source, sensor=sensor, simulation_options=simulation_options,
                execution_options=execution_options
            )

        sensor_data = sensor_data["p"]
        if not use_cpu:
            sensor_data = sensor_data.T

        # Explicitly pin order to avoid future default-order changes affecting training data.
        combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data, order="C")

        time_data_cat[ind, :, :] = combined_sensor_data

        """plt.figure(dpi=500)
        plt.imshow(combined_sensor_data, aspect='auto', vmin=-1, vmax=1, cmap=get_color_map())
        plt.ylabel('Transducer index',fontsize=14)
        plt.xlabel('Time Step',fontsize=14)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Amplitude", fontsize=14)
        plt.show()"""

    np.savez(save_path, time_data_cat=time_data_cat, sensor_coords=sensor_coords)


def _run_single_sample(args):
    sample_idx, shift_grid, dataset_dir, output_dir, use_single = args
    worker_temp_dir = os.path.join(tempfile.gettempdir(), f"kwave_worker_{os.getpid()}_{sample_idx}")
    os.makedirs(worker_temp_dir, exist_ok=True)
    try:
        per_SoSMap_Kwave(
            use_single=use_single,
            worker_temp_dir=worker_temp_dir,
            rand_shift_grid=int(shift_grid),
            sample_idx=sample_idx,
            dataset_dir=dataset_dir,
            save_dir=output_dir,
        )
        return None
    except Exception as exc:
        return f"Sample {sample_idx} failed: {exc}"
    finally:
        try:
            shutil.rmtree(worker_temp_dir, ignore_errors=True)
        except Exception:
            pass


def batch_generate_kwavedata_parallel_2(
        dataset_dir: str,
        output_dir: str,
        start_idx: int,
        num_samples: int,
        use_single: bool = True,
        rng_seed: int = None,
        max_shift_allowance: int = 20,
        pool_size: int = None,
        cpu_workers: int = 8
):
    plt.close('all')
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"dataset_dir not found: {dataset_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if cpu_workers > 0:
        gpu_samples = 0
        cpu_samples = num_samples - gpu_samples
    else:
        gpu_samples = num_samples
        cpu_samples = 0

    half_max_shift = int(max_shift_allowance // 2)
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()
    rand_shifts_grid = rng.integers(-half_max_shift, half_max_shift + 1, size=num_samples)

    print(f"开始并行仿真任务(gpu): 样本 {start_idx} 到 {start_idx + gpu_samples - 1}")
    futures = []
    if pool_size > 0:
        with ProcessPoolExecutor(max_workers=pool_size) as ex:
            for idx in range(gpu_samples):
                real_sample_idx = start_idx + idx
                save_path = os.path.join(output_dir, f"sample_{real_sample_idx:06d}.npz")
                if os.path.exists(save_path):
                    print(f"Sample {real_sample_idx} exists. Skipping...")
                    continue
                args = (
                    real_sample_idx,
                    rand_shifts_grid[idx],
                    dataset_dir,
                    output_dir,
                    use_single,
                )
                futures.append(ex.submit(_run_single_sample, args))

            for fut in as_completed(futures):
                err = fut.result()
                if err:
                    print(err)

    if cpu_workers > 0:
        cpu_start = start_idx + gpu_samples
        print(f"使用CPU处理 {cpu_samples} 个样本（{cpu_workers}个进程并行）")

        # 使用ProcessPoolExecutor并行处理CPU任务
        with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            for i in range(cpu_samples):
                sample_idx = cpu_start + i
                save_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.npz")
                if os.path.exists(save_path):
                    print(f"Sample {sample_idx} exists. Skipping...")
                    continue
                args = (sample_idx,
                        rand_shifts_grid[cpu_start + i],
                        dataset_dir,
                        output_dir,
                        use_single,)
                futures.append(executor.submit(_run_single_sample_cpu, args))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    print(f"处理失败: {result}")


def _run_single_sample_cpu(args):
    sample_idx, shift_grid, dataset_dir, output_dir, use_single = args
    worker_temp_dir = os.path.join(tempfile.gettempdir(), f"kwave_worker_{os.getpid()}_{sample_idx}")
    os.makedirs(worker_temp_dir, exist_ok=True)
    try:
        per_SoSMap_Kwave(
            use_single=use_single,
            worker_temp_dir=worker_temp_dir,
            rand_shift_grid=int(shift_grid),
            sample_idx=sample_idx,
            dataset_dir=dataset_dir,
            save_dir=output_dir,
            use_cpu=True,
        )
        return None
    except Exception as exc:
        return f"Sample {sample_idx} failed: {exc}"
    finally:
        try:
            shutil.rmtree(worker_temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    max_shift_grid = 20
    half_shift_grid = max_shift_grid // 2
    rand_shift_grid = random.randint(-half_shift_grid, half_shift_grid)
    per_SoSMap_Kwave(use_single=True,
                     worker_temp_dir=r"./temp",
                     rand_shift_grid=rand_shift_grid,
                     sample_idx=0,
                     dataset_dir=r"./SoSMap", save_dir="./KwaveResult", )