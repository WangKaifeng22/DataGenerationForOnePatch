#import numpy as np
import os
import signal
import subprocess
import multiprocessing as mp
from GenerateSoSMaps import generate_sos_maps
from Kwave import batch_generate_kwavedata_parallel_2


def _cleanup_local_children():
    # 只清理当前 Python 主进程创建的子进程，避免误杀其他任务。
    for child in mp.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in mp.active_children():
        try:
            child.join(timeout=1)
        except Exception:
            pass


def _kill_stale_kwave_binaries():
    # 兜底清理上次异常退出后可能残留的 k-Wave 可执行进程
    names = ["kspaceFirstOrder2DG", "kspaceFirstOrder2D"]
    try:
        if os.name == "nt":
            for n in names:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", f"{n}.exe"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        else:
            # Linux/macOS: 按命令行匹配杀进程
            for n in names:
                subprocess.run(
                    ["pkill", "-f", n],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    except Exception:
        pass

def _graceful_shutdown(*_args):
    _cleanup_local_children()
    _kill_stale_kwave_binaries()


def main():
    # 启动前先做一次兜底清理，避免上次 Ctrl+C 留下的进程继续占用内存。
    _kill_stale_kwave_binaries()

    pool_size = 1
    Generate_SoSMaps = True  # 是否生成声速图
    data_num = 1 # 每个idx生成声速图的数量
    start_num = 1
    use_single = True
    length_scale_bg_list = [2e-3]
    length_scale_inc_list = [2e-3]

    output_dir = r"./dataset"  # 输出目录

    for i in range(len(length_scale_bg_list)):
        length_scale_bg = length_scale_bg_list[i]
        length_scale_inc = length_scale_inc_list[i]
        dir_name = f"{length_scale_bg:.1e}andInc{length_scale_inc:.1e}"
        SoSMaps_output_dir = os.path.join(output_dir, 'SoSMap', dir_name)

        if Generate_SoSMaps:
            if not os.path.exists(SoSMaps_output_dir):
                os.makedirs(SoSMaps_output_dir)
            generate_sos_maps(SoSMaps_output_dir,data_num,start_num,length_scale_bg,length_scale_inc,
                              use_single=use_single,ellipses_range=(1, 2), pool_size=pool_size, plot_samples=False
                              ,sharpness=3.5, texture_strength=0.2)


        Kwave_output_dir = os.path.join(output_dir, 'KwaveResult', dir_name)
        if not os.path.exists(Kwave_output_dir):
            os.makedirs(Kwave_output_dir)

        batch_generate_kwavedata_parallel_2(SoSMaps_output_dir, Kwave_output_dir, start_num, data_num,
                                            use_single=use_single, rng_seed=114514, max_shift_allowance=20,
                                            pool_size=pool_size, cpu_workers=0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, _graceful_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在强制清理并退出...")
        _graceful_shutdown()
        raise SystemExit(130)
    finally:
        _graceful_shutdown()
