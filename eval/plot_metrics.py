import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from matplotlib.axes import Axes


epochs = 300

eval_dir = Path(__file__).resolve().parent
root_dir = eval_dir.parent

def split_run_path(run: str) -> tuple[Path, str]:
    """Split a run path into `(sub_dir, run_name)`."""
    parts = re.split(r"[\\/]", run)
    name = parts[-1]
    sub_dir = Path(*parts[:-1])
    return sub_dir, name


def get_history(run_name: str) -> pd.DataFrame:
    """Load and type-coerce one run's metrics history CSV."""
    sub_dir, name = split_run_path(run_name)
    path = root_dir / "runs" / sub_dir / name / "metrics" / "history.csv"

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)

    for col in df.columns:
        if col not in ["CodebookUsage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["epoch", "bw_idx"])


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Apply a simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _metric_series(
    df: pd.DataFrame,
    bitrates: list[int] | tuple[int, ...],
    metric: str,
    avg: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return epoch/value arrays for one metric over selected bitrates."""
    selected = df[df["Bitrate(bps)"].isin(bitrates)]

    pivot = selected.pivot_table(
        index="epoch",
        columns="Bitrate(bps)",
        values=metric,
        aggfunc="last",
    )

    scores = pivot.sum(axis=1, min_count=1)

    if avg:
        scores = scores / len(bitrates)

    scores = scores.replace(0, np.nan)

    return scores.index.to_numpy(), scores.to_numpy()


def _plot_metric(
    ax: Axes,
    df: pd.DataFrame,
    bitrates: list[int] | tuple[int, ...],
    metric: str,
    color: str = "#1f77b4",
    avg: bool = False,
    dotted: bool = True,
) -> None:
    """Plot one metric series on an axis."""
    x, y = _metric_series(df, bitrates, metric, avg=avg)
    mask = np.isfinite(y)

    if dotted:
        ax.plot(x[mask], y[mask], "o", markersize=4, color=color)
    else:
        ax.plot(x[mask], y[mask], color=color)


def plot_metric_dict(
    ax: Axes,
    bitrates: list[int] | tuple[int, ...],
    runs: dict[str, str],
    metric: str,
    ylabel: str | None = None,
    avg: bool = True,
    dotted: bool = True,
) -> None:
    """Plot one metric for several named runs."""
    for run_name, color in runs.items():
        df = get_history(run_name)
        _plot_metric(ax, df, bitrates, metric, color=color, avg=avg, dotted=dotted)

    ax.set_ylabel(ylabel or metric)
    ax.set_xlim([0, epochs])


def plot_estoi_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot ESTOI across runs."""
    plot_metric_dict(ax, bitrates, runs, "ESTOI", "Total ESTOI Score", avg, dotted)


def plot_pesq_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot PESQ across runs."""
    plot_metric_dict(ax, bitrates, runs, "PESQ", "Total PESQ Score", avg, dotted)


def plot_dis_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot discriminator loss across runs."""
    plot_metric_dict(ax, bitrates, runs, "DisLoss(e-3)", "Discriminator Loss", avg, dotted)


def plot_commit_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot commitment loss across runs."""
    plot_metric_dict(ax, bitrates, runs, "CommitLoss(e-6)", "Commitment Loss", avg, dotted)
    ax.set_yscale("log")


def plot_feat_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot feature matching loss across runs."""
    plot_metric_dict(ax, bitrates, runs, "FeatLoss(e-3)", "Feature Loss", avg, dotted)


def plot_total_score_dict(ax: Axes, bitrates: list[int] | tuple[int, ...], runs: dict[str, str], avg: bool = True, dotted: bool = True) -> None:
    """Plot combined PESQ/ESTOI score across runs."""
    for run_name, color in runs.items():
        df = get_history(run_name)

        x_pesq, pesq = _metric_series(df, bitrates, "PESQ", avg=False)
        x_estoi, estoi = _metric_series(df, bitrates, "ESTOI", avg=False)

        assert np.array_equal(x_pesq, x_estoi)

        scores = pesq / 5 + estoi

        if avg:
            scores /= len(bitrates) * 2

        mask = np.isfinite(scores)

        if dotted:
            ax.plot(x_pesq[mask], scores[mask], "o", markersize=4, color=color)
        else:
            ax.plot(x_pesq[mask], scores[mask], color=color)

    ax.set_ylabel("Total Score")
    ax.set_xlim([0, epochs])


def plot_total_score_smooth_dict(
    ax: Axes,
    bitrates: list[int] | tuple[int, ...],
    runs: dict[str, str],
    avg: bool = True,
    dotted: bool = False,
    window: int = 9,
) -> None:
    """Plot smoothed combined PESQ/ESTOI score across runs."""
    for run_name, color in runs.items():
        df = get_history(run_name)

        x_pesq, pesq = _metric_series(df, bitrates, "PESQ", avg=False)
        x_estoi, estoi = _metric_series(df, bitrates, "ESTOI", avg=False)

        assert np.array_equal(x_pesq, x_estoi)

        scores = pesq / 5 + estoi

        if avg:
            scores /= len(bitrates) * 2

        mask = np.isfinite(scores)
        x = x_pesq[mask]
        scores = scores[mask]

        if len(scores) < window:
            continue

        smooth_scores = smooth(scores, window)
        smooth_x = x[window // 2: len(x) - window // 2]

        if dotted:
            ax.plot(smooth_x, smooth_scores, "o", markersize=4, color=color)
        else:
            ax.plot(smooth_x, smooth_scores, color=color)

    ax.set_ylabel("Total Score (Smoothed)")
    ax.set_xlim([0, epochs])


def set_grid(ax: Axes, checkpoint_ticks: int) -> None:
    """Apply major/minor grid lines for checkpoint plots."""
    ax.xaxis.set_major_locator(MultipleLocator(checkpoint_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")


if __name__ == "__main__":
    bitrates = [300,600,1200]

    to_plot = {
        #"nevo_lite": "blue",
        "main/nevo": "green",
        "main/nevo_lite":"blue",
        #"80ms_lookahead": "red",
    }

    to_plot = {k: v for k, v in to_plot.items() if v is not None}

    fig, axs = plt.subplots(5, sharex=True)

    plot_estoi_dict(axs[0], bitrates, to_plot, dotted=False, avg=True)
    plot_pesq_dict(axs[1], bitrates, to_plot, dotted=False, avg=True)
    plot_total_score_dict(axs[2], bitrates, to_plot, dotted=False)
    plot_total_score_smooth_dict(axs[3], bitrates, to_plot, dotted=False, window=5)
    plot_commit_dict(axs[4], bitrates, to_plot, dotted=False)

    for ax in axs:
        set_grid(ax, 10)

    axs[0].set_ylim(0.0, 1.0)  # ESTOI
    axs[1].set_ylim(0.5, 3.5)  # PESQ
    #axs[2].set_ylim(0.0, 1.0)  # Total score
    #axs[3].set_ylim(0.0, 1.0)  # Smoothed total score

    axs[-1].set_xlabel("Epochs")
    plt.show()
