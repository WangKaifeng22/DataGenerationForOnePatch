import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from config import get_config
from kwave.utils.colormap import get_color_map

def log_transform(data, k=1, c=0):
    """ 对数变换，用于处理跨度大的频域数据 """
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def minmax_normalize(vid, vmin, vmax, scale=2):
    """ 归一化到 [-1, 1] (scale=2) 或 [0, 1] (scale=1) """
    vid = vid - vmin
    vid = vid / (vmax - vmin)
    if scale == 2:
        return (vid - 0.5) * 2
    return vid


def suppress_early_incident(
    data: np.ndarray,
    t_axis: np.ndarray,
    gate_center: float,
    gate_ramp: float,
    compress_alpha: float,
) -> np.ndarray:
    """Suppress strong early incident wave and compress dynamic range."""
    out = data.astype(np.float64, copy=True)

    ramp = max(float(gate_ramp), 1e-12)
    gate = 1.0 / (1.0 + np.exp(-(t_axis - float(gate_center)) / ramp))
    out *= gate[None, None, :]

    if compress_alpha > 0:
        a = float(compress_alpha)
        out = np.tanh(a * out) / np.tanh(a)
    return out

def plot_pseudo3d_tx_stack(
    time_data_cat: np.ndarray,
    time_step: float = 1.0,
    time_start: float = 0.0,
    time_unit: str = "sample",
    tx_step: int = 2,
    rx_step: int = 1,
    t_step_plot: int = 4,
    layer_gap: float = 1.0,
    amp_scale: float = 0.2,
    normalize_amplitude: bool = True,
    suppress_incident: bool = False,
    gate_center: float = 2.0,
    gate_ramp: float = 0.35,
    compress_alpha: float = 1.8,
    mode: str = "slice",
    elev: float = 28, azim: float = -58,
    out_png: str = "tx_rx_t_pseudo3d.png",
    out_pdf: str = "tx_rx_t_pseudo3d.pdf",
):
    assert time_data_cat.ndim == 3, "time_data_cat must be (Tx, Rx, T)"
    n_tx, n_rx, n_t = time_data_cat.shape

    data = time_data_cat.astype(np.float64, copy=False)
    t_full = time_start + np.arange(n_t, dtype=np.float64) * time_step
    if suppress_incident:
        data = suppress_early_incident(
            data,
            t_axis=t_full,
            gate_center=gate_center,
            gate_ramp=gate_ramp,
            compress_alpha=compress_alpha,
        )

    if normalize_amplitude:
        vmax = np.max(np.abs(data))
        if vmax > 0:
            data = data / vmax

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.78, bottom=0.08, top=0.93)
    ax = fig.add_subplot(111, projection="3d")

    t_idx = np.arange(0, n_t, max(1, t_step_plot), dtype=np.int64)
    t = t_full[t_idx]
    tx_indices = list(range(0, n_tx, max(1, tx_step)))
    rx_indices = np.arange(0, n_rx, max(1, rx_step), dtype=np.int64)

    cmap = get_color_map()

    if mode == "line":
        n_colors = max(2, len(tx_indices))
        for color_idx, tx in enumerate(tx_indices):
            # Reuse the same base colormap for all plotting modes.
            c = cmap(color_idx / (n_colors - 1))
            z_base = tx * layer_gap
            for rx in rx_indices:
                x = t
                y = np.full_like(t, rx, dtype=np.float64)
                z = z_base + amp_scale * data[tx, rx, t_idx]
                ax.plot(x, y, z, color=c, linewidth=0.6, alpha=0.85)
    else:
        # A clearer pseudo-3D: each Tx is one Rx-Time color slice stacked on z.
        robust = np.percentile(np.abs(data[:, rx_indices][:, :, t_idx]), 99.0)
        lim = robust if robust > 0 else 1.0
        norm = Normalize(vmin=-lim, vmax=lim)

        t_grid, rx_grid = np.meshgrid(t, rx_indices)
        for tx in tx_indices:
            z_grid = np.full_like(t_grid, tx * layer_gap, dtype=np.float64)
            slice_data = data[tx, rx_indices][:, t_idx]
            facecolors = cmap(norm(slice_data))
            facecolors[..., -1] = 0.92
            ax.plot_surface(
                t_grid,
                rx_grid,
                z_grid,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                shade=False,
                linewidth=0,
                antialiased=False,
            )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.10)
        cbar.set_label("Amplitude (a.u.)")

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Receiver index")
    ax.set_zlabel("Transmitter index", labelpad=10)
    ax.zaxis.set_label_coords(-0.10, 0.5)
    ax.set_title("Pseudo-3D stacked sensor data (Tx-Rx-T)")
    ax.view_init(elev=elev, azim=azim)

    tick_tx = np.array(tx_indices[::max(1, len(tx_indices) // 8)] or [0], dtype=int)
    ax.set_zticks(tick_tx * layer_gap)
    ax.set_zticklabels([str(v) for v in tick_tx])
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(float(rx_indices[0]), float(rx_indices[-1]))
    ax.set_zlim(-0.5 * layer_gap, (n_tx - 0.2) * layer_gap)

    try:
        ax.set_box_aspect((2.6, 1.3, 1.7))
    except Exception:
        pass

    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    d = np.load("./KwaveResult/sample_000000.npz")
    time_data_cat = d["time_data_cat"]
    cfg = get_config()

    time_data_cat = log_transform(time_data_cat)
    d_min = np.nanmin(time_data_cat)
    d_max = np.nanmax(time_data_cat)
    time_data_cat = minmax_normalize(time_data_cat, d_min, d_max, scale=2)

    plot_pseudo3d_tx_stack(
        time_data_cat,
        time_step=cfg.dt * 1e6,
        time_start=0.0,
        time_unit="us",
        tx_step=2,
        rx_step=1,
        t_step_plot=4,
        mode="slice",
        layer_gap=1.0,
        amp_scale=0.2,
        suppress_incident=False,
        gate_center=2.5,
        gate_ramp=0.3,
        compress_alpha=3.5,
        out_png="./temp/tx_rx_t_pseudo3d.png",
        out_pdf="./temp/tx_rx_t_pseudo3d.pdf",
        normalize_amplitude=False,
    )
