"""Generate publication-ready figures for the Results and Discussion chapter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", pad: float = 0.002) -> None:
    for bar in bars:
        value = bar.get_width() if bar.get_width() > bar.get_height() else bar.get_height()
        if bar.get_width() > bar.get_height():
            ax.text(
                bar.get_x() + bar.get_width() + pad,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=6.5,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() + pad,
                fmt.format(value),
                va="bottom",
                ha="center",
                fontsize=6.5,
            )


def stage1_model_comparison() -> None:
    models = [
        "Flow-statistics MLP",
        "No encoding",
        "BiLSTM",
        "CNN1D",
        "LSTM",
        "GRU",
        "CNN1D + time",
        "LSTM + time",
        "Position only",
        "Time only",
        "Position + time",
    ]
    macro_f1 = np.array([0.7762, 0.8507, 0.8522, 0.8579, 0.8582, 0.8595, 0.8670, 0.8677, 0.8739, 0.8818, 0.8882])
    class1_f1 = np.array([0.5723, 0.7134, 0.7160, 0.7274, 0.7273, 0.7296, 0.7443, 0.7455, 0.7575, 0.7728, 0.7853])
    pr_auc = np.array([0.5604, 0.8003, 0.7903, 0.7975, 0.7976, 0.8009, 0.7821, 0.8077, 0.8482, 0.8510, 0.8564])

    y = np.arange(len(models))
    h = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    ax.axhspan(9.5, 10.5, color=BLUE, alpha=0.08, zorder=0)
    b1 = ax.barh(y - h, macro_f1, height=h, color=BLUE, label="Macro-F1")
    b2 = ax.barh(y, class1_f1, height=h, color=ORANGE, label="Class-1 F1")
    b3 = ax.barh(y + h, pr_auc, height=h, color=GREEN, label="PR-AUC")
    ax.set_yticks(y, models)
    ax.set_xlim(0.52, 0.91)
    ax.set_xlabel("Score (axis starts at 0.52)")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=3, frameon=False)
    label_bars(ax, b1)
    label_bars(ax, b2)
    label_bars(ax, b3)
    fig.tight_layout()
    save(fig, "stage1_model_comparison")


def stage1_error_comparison() -> None:
    models = [
        "Flow-statistics MLP",
        "No encoding",
        "BiLSTM",
        "CNN1D",
        "LSTM",
        "GRU",
        "CNN1D + time",
        "LSTM + time",
        "Position only",
        "Time only",
        "Position + time",
    ]
    fp = np.array([853, 420, 372, 423, 315, 282, 307, 286, 294, 300, 296])
    fn = np.array([501, 394, 415, 361, 423, 437, 391, 401, 370, 332, 306])
    fpr = fp / 34003 * 100

    y = np.arange(len(models))
    height = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axhspan(9.5, 10.5, color=BLUE, alpha=0.08, zorder=0)
    bfp = ax.barh(y - height / 2, fp, height, color=RED, label="False positives")
    bfn = ax.barh(y + height / 2, fn, height, color=PURPLE, label="False negatives")
    ax.set_yticks(y, models)
    ax.set_xlim(0, 950)
    ax.set_xlabel("Number of test errors")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    for bar, rate in zip(bfp, fpr):
        ax.text(
            bar.get_width() + 7,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())} ({rate:.3f}%)",
            va="center",
            ha="left",
            fontsize=6.5,
        )
    for bar in bfn:
        ax.text(
            bar.get_width() + 7,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
    fig.tight_layout()
    save(fig, "stage1_error_comparison")


def stage1_sensitivity() -> None:
    lengths = np.array([8, 16, 32, 64, 128, 256])
    macro_f1 = np.array([0.8814458, 0.8803349, 0.8817057, 0.8860806, 0.8882279, 0.8819939])
    class1_f1 = np.array([0.7722560, 0.7700612, 0.7727920, 0.7813268, 0.7853067, 0.7734923])
    pr_auc = np.array([0.8368694, 0.8454853, 0.8571312, 0.8568880, 0.8564024, 0.8492834])

    strategies = ["head", "head-tail", "random"]
    strategy_metrics = {
        "Macro-P": [0.8895604, 0.8891173, 0.8958889],
        "Macro-R": [0.8869054, 0.8840772, 0.8707326],
        "Macro-F1": [0.8882279, 0.8865792, 0.8828568],
        "PR-AUC": [0.8564024, 0.8566055, 0.8446620],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    ax = axes[0]
    ax.plot(lengths, macro_f1, marker="o", color=BLUE, label="Macro-F1")
    ax.plot(lengths, class1_f1, marker="s", color=ORANGE, label="Class-1 F1")
    ax.plot(lengths, pr_auc, marker="^", color=GREEN, label="PR-AUC")
    ax.axvline(128, color=GRAY, linestyle="--", linewidth=0.9)
    ax.annotate("selected", xy=(128, 0.8882), xytext=(76, 0.866), arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [str(v) for v in lengths])
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylim(0.75, 0.90)
    ax.set_xlabel("Maximum packet-sequence length, L")
    ax.set_ylabel("Score")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=3)

    ax = axes[1]
    x = np.arange(len(strategies))
    width = 0.19
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    for idx, ((name, values), color) in enumerate(zip(strategy_metrics.items(), colors)):
        ax.bar(x + (idx - 1.5) * width, values, width, label=name, color=color)
    ax.set_xticks(x, strategies)
    ax.set_ylim(0.83, 0.91)
    ax.set_ylabel("Score (axis starts at 0.83)")
    ax.set_xlabel("Packet-selection strategy")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout()
    save(fig, "stage1_sensitivity")


def stage1_fusion_comparison() -> None:
    schemes = ["A: tiled flow stats", "B: packet only", "C: token-FiLM"]
    metrics = {
        "Macro-P": [0.8919033, 0.8895604, 0.8990258],
        "Macro-R": [0.8600128, 0.8869054, 0.8919516],
        "Macro-F1": [0.8752143, 0.8882279, 0.8954539],
        "ROC-AUC": [0.9800629, 0.9859678, 0.9890373],
        "PR-AUC": [0.8326108, 0.8564024, 0.8762558],
    }
    fp = np.array([264, 296, 267])
    fn = np.array([383, 306, 293])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ax = axes[0]
    x = np.arange(len(schemes))
    width = 0.15
    colors = [BLUE, ORANGE, GREEN, SKY, PURPLE]
    for idx, ((name, values), color) in enumerate(zip(metrics.items(), colors)):
        ax.bar(x + (idx - 2) * width, values, width, label=name, color=color)
    ax.set_xticks(x, ["A", "B", "C"])
    ax.set_ylim(0.81, 1.00)
    ax.set_xlabel("Stage 1 integration scheme")
    ax.set_ylabel("Score (axis starts at 0.81)")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.005))

    ax = axes[1]
    y = np.arange(len(schemes))
    height = 0.35
    bfp = ax.barh(y - height / 2, fp, height, color=RED, label="FP")
    bfn = ax.barh(y + height / 2, fn, height, color=PURPLE, label="FN")
    ax.set_yticks(y, schemes)
    ax.invert_yaxis()
    ax.set_xlim(0, 430)
    ax.set_xlabel("Number of test errors")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    label_bars(ax, bfp, fmt="{:.0f}", pad=5)
    label_bars(ax, bfn, fmt="{:.0f}", pad=5)
    fig.tight_layout()
    save(fig, "stage1_fusion_comparison")


def stage2_context_analysis() -> None:
    contexts = ["Source host", "Destination host", "Time only", "Endpoint"]
    macro_f1 = np.array([0.927539, 0.919625, 0.920109, 0.921457])
    pr_auc = np.array([0.934499, 0.926171, 0.921543, 0.928712])
    fpr = np.array([0.5429, 0.6876, 0.5701, 0.8682])
    recall1 = np.array([0.850284, 0.848864, 0.830398, 0.886080])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    ax = axes[0]
    x = np.arange(len(contexts))
    width = 0.35
    b1 = ax.bar(x - width / 2, macro_f1, width, color=BLUE, label="Macro-F1")
    b2 = ax.bar(x + width / 2, pr_auc, width, color=GREEN, label="PR-AUC")
    ax.set_xticks(x, ["Source", "Destination", "Time", "Endpoint"])
    ax.set_ylim(0.90, 0.945)
    ax.set_ylabel("Score (axis starts at 0.90)")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    label_bars(ax, b1, pad=0.0006)
    label_bars(ax, b2, pad=0.0006)

    ax = axes[1]
    colors = [BLUE, SKY, GRAY, ORANGE]
    markers = ["*", "o", "s", "^"]
    for name, xval, yval, color, marker in zip(contexts, fpr, recall1, colors, markers):
        ax.scatter(xval, yval, s=115 if name == "Source host" else 58, color=color, marker=marker, edgecolor="white", linewidth=0.7, zorder=3)
        offset = (5, 5) if name != "Time only" else (5, -11)
        ax.annotate(name, (xval, yval), xytext=offset, textcoords="offset points", fontsize=7)
    ax.set_xlim(0.50, 0.93)
    ax.set_ylim(0.82, 0.895)
    ax.set_xlabel(r"False-positive rate (%)  $\leftarrow$ lower")
    ax.set_ylabel(r"Class-1 recall  ($\uparrow$ higher)")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    save(fig, "stage2_context_analysis")


def stage2_window_sensitivity() -> None:
    windows = np.array([16, 32, 64, 128, 256])
    macro_f1 = np.array([0.913367, 0.912877, 0.912602, 0.927539, 0.915965])
    pr_auc = np.array([0.918065, 0.920788, 0.916339, 0.934499, 0.922061])
    fp = np.array([614, 775, 625, 439, 560])
    fn = np.array([563, 452, 564, 527, 572])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))
    ax = axes[0]
    ax.plot(windows, macro_f1, marker="o", color=BLUE, label="Macro-F1")
    ax.plot(windows, pr_auc, marker="^", color=GREEN, label="PR-AUC")
    ax.axvline(128, color=GRAY, linestyle="--", linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, [str(v) for v in windows])
    ax.set_ylim(0.905, 0.94)
    ax.set_xlabel("Context window, W")
    ax.set_ylabel("Score")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(windows, fp, marker="o", color=RED, label="False positives")
    ax.plot(windows, fn, marker="s", color=PURPLE, label="False negatives")
    ax.axvline(128, color=GRAY, linestyle="--", linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, [str(v) for v in windows])
    ax.set_xlabel("Context window, W")
    ax.set_ylabel("Error count")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "stage2_window_sensitivity")


def stage2_architecture_comparison() -> None:
    models = ["No context", "Vanilla Tr.", "LSTM", "GRU", "CNN+LSTM", "Proposed"]
    macro_f1 = np.array([0.894396, 0.906259, 0.908866, 0.909448, 0.909657, 0.927539])
    class1_f1 = np.array([0.797915, 0.820308, 0.825635, 0.826368, 0.826819, 0.861047])
    pr_auc = np.array([0.878606, 0.907286, 0.913259, 0.914091, 0.905036, 0.934499])
    fp = np.array([861, 616, 777, 566, 514, 439])
    fn = np.array([612, 644, 499, 643, 677, 527])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.25
    b1 = ax.bar(x - width, macro_f1, width, color=BLUE, label="Macro-F1")
    b2 = ax.bar(x, class1_f1, width, color=ORANGE, label="Class-1 F1")
    b3 = ax.bar(x + width, pr_auc, width, color=GREEN, label="PR-AUC")
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylim(0.77, 0.95)
    ax.set_ylabel("Score (axis starts at 0.77)")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax = axes[1]
    y = np.arange(len(models))
    height = 0.35
    bfp = ax.barh(y - height / 2, fp, height, color=RED, label="FP")
    bfn = ax.barh(y + height / 2, fn, height, color=PURPLE, label="FN")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel("Number of test errors")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    label_bars(ax, bfp, fmt="{:.0f}", pad=8)
    label_bars(ax, bfn, fmt="{:.0f}", pad=8)
    ax.set_xlim(0, 950)
    fig.tight_layout()
    save(fig, "stage2_architecture_comparison")


def stage2_efficiency() -> None:
    models = ["No context", "Vanilla Tr.", "LSTM", "GRU", "CNN+LSTM", "Proposed"]
    params = np.array([514, 132994, 922626, 692226, 1218306, 283522])
    latency = np.array([0.1811979, 0.1967497, 0.3463924, 0.3402049, 0.3584734, 0.1965021])
    macro_f1 = np.array([0.894396, 0.906259, 0.908866, 0.909448, 0.909657, 0.927539])

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for name, xval, yval, score in zip(models, params, latency, macro_f1):
        proposed = name == "Proposed"
        ax.scatter(
            xval,
            yval,
            s=190 if proposed else 85,
            color=BLUE if proposed else GRAY,
            marker="*" if proposed else "o",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        annotation_style = {
            "No context": ((5, 6), "left", "bottom"),
            "Vanilla Tr.": ((-8, -9), "right", "top"),
            "LSTM": ((8, -8), "left", "top"),
            "GRU": ((-8, 5), "right", "bottom"),
            "CNN+LSTM": ((-3, 12), "right", "bottom"),
            "Proposed": ((8, 5), "left", "bottom"),
        }
        offset, ha, va = annotation_style[name]
        ax.annotate(
            f"{name}\nF1={score:.3f}",
            (xval, yval),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha=ha,
            va=va,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test inference time (ms per flow)")
    ax.set_xlim(350, 1_700_000)
    ax.set_ylim(0.16, 0.38)
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    save(fig, "stage2_efficiency")


if __name__ == "__main__":
    stage1_model_comparison()
    stage1_error_comparison()
    stage1_sensitivity()
    stage1_fusion_comparison()
    stage2_context_analysis()
    stage2_window_sensitivity()
    stage2_architecture_comparison()
    stage2_efficiency()
    print(f"Wrote figures to {OUT_DIR}")
