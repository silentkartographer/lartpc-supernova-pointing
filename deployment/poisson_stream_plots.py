#!/usr/bin/env python3
"""Figures for the Poisson-stream study (reads the CSV logs written by larpoint.deployment.poisson_stream).
"""
# Generates figures from the simulation output CSVs.
# Produces (under <output_dir>/plots/):
#   01_score_distributions.png
#   02_confusion_matrix.png
#   03_cumulative_outcomes.png
#   04_rolling_metrics.png
#   05_memory_timeseries.png
#   05b_utilization_all_events.png       (unless --no_util_all_events)
#   05c_utilization_fired_runs.png       (unless --no_util_fired_runs)
#   06a_gpu_memory_distributions.png
#   06b_cpu_memory_distributions.png
#   07_timing_distributions.png
#   07b_fired_event_total_timing.png
#   08_costheta_histograms.png           (if direction_log.csv present)
#   09_costheta_by_score.png             (if direction_log.csv present)
#   10_interarrival_times.png
#   11_summary_stats.png
#   gallery_fired_events.pdf             (if raw/preproc NPZs present)

from __future__ import annotations
import os
import argparse
import math

from larpoint.paths import OUTPUT_ROOT

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from scipy import stats as scipy_stats


# Global style
COLORS = {
    "TP":          "#2ecc71",
    "TN":          "#3498db",
    "FP":          "#e74c3c",
    "FN":          "#f39c12",
    "Michel":      "#9b59b6",
    "noMichel":    "#1abc9c",
    "GPU":         "#e74c3c",
    "CPU":         "#3498db",
    "classifier":  "#f39c12",
    "pipeline":    "#27ae60",
    "threshold":   "#c0392b",
    "mean":        "#e67e22",
    "GT":          "#2ecc71",
    "pred":        "#e74c3c",
    "baseline":    "#7f8c8d",
    "combined":    "#2c3e50",
}

VIEW_COLORS = {"U": "#3498db", "V": "#9b59b6", "Z": "#e67e22"}

plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#555555",
    "axes.labelcolor":     "#222222",
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "axes.grid":           True,
    "grid.color":          "#dddddd",
    "grid.linewidth":      0.6,
    "xtick.color":         "#555555",
    "ytick.color":         "#555555",
    "font.family":         "sans-serif",
    "font.size":           10,
    "legend.fontsize":     9,
    "legend.framealpha":   0.85,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
})

_DPI_SAVE = 300


def _save(fig, path):
    fig.savefig(path, dpi=_DPI_SAVE, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")


def _prepare_event_df(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()

    ev["combined_gpu_util_max_pct"] = ev[
        ["classifier_gpu_util_max_pct", "pipeline_gpu_util_max_pct"]
    ].max(axis=1)

    ev["combined_cpu_proc_util_max_pct"] = ev[
        ["classifier_cpu_proc_util_max_pct", "pipeline_cpu_proc_util_max_pct"]
    ].max(axis=1)

    return ev


def _prepare_memory_df(mem: pd.DataFrame) -> pd.DataFrame:
    mem = mem.copy()

    mem["combined_peak_gpu_gb"] = mem[
        ["classifier_peak_gpu_gb", "pipeline_peak_gpu_gb"]
    ].max(axis=1)

    mem["combined_peak_cpu_rss_gb"] = mem[
        ["classifier_peak_cpu_rss_gb", "pipeline_peak_cpu_rss_gb"]
    ].max(axis=1)

    return mem


def _clip_for_display(arr: np.ndarray, percentile: float):
    x_hi = float(np.percentile(arr, percentile))
    clipped = arr[arr <= x_hi]
    n_clip = int(np.sum(arr > x_hi))
    return clipped, x_hi, n_clip


def _nice_util_ylim(series_list):
    finite = []
    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        if s.size:
            finite.append(float(np.max(s)))
    if not finite:
        return 0.0, 105.0
    maxv = max(finite)
    upper = max(105.0, 10.0 * math.ceil((1.10 * maxv) / 10.0))
    return 0.0, upper


def _focused_hist_params(data: np.ndarray, nbins: int = 20):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return np.linspace(0, 1, nbins + 1), (0.0, 1.0)

    dmin = float(np.min(data))
    dmax = float(np.max(data))
    spread = dmax - dmin
    center = float(np.median(data))

    if spread <= 0:
        pad = max(abs(center) * 5e-4, 5e-4)
        x_lo = center - pad
        x_hi = center + pad
    else:
        pad = max(spread * 0.25, abs(center) * 2e-4, 2e-4)
        x_lo = dmin - pad
        x_hi = dmax + pad

    bins = np.linspace(x_lo, x_hi, nbins + 1)
    return bins, (x_lo, x_hi)


# 01 — Classifier score distributions
def plot_score_distributions(ev: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    michel_scores   = ev.loc[ev["true_label"] == 1, "classifier_score"]
    nomichel_scores = ev.loc[ev["true_label"] == 0, "classifier_score"]

    bins = np.linspace(0, 1, 51)
    ax.hist(nomichel_scores, bins=bins, color=COLORS["noMichel"], alpha=0.7,
            label=f"No-Michel (n={len(nomichel_scores)})", density=True, edgecolor="white", lw=0.4)
    ax.hist(michel_scores,   bins=bins, color=COLORS["Michel"],   alpha=0.7,
            label=f"Michel (n={len(michel_scores)})", density=True, edgecolor="white", lw=0.4)

    thr = ev.get("threshold_used", pd.Series([0.5])).iloc[0] if "threshold_used" in ev.columns else 0.5
    ax.axvline(thr, color=COLORS["threshold"], lw=2, ls="--",
               label=f"Threshold = {thr:.3f}")

    ax.set_xlabel("Classifier Score (P(Michel))")
    ax.set_ylabel("Density")
    ax.set_title("Classifier Score Distributions")
    ax.legend()
    ax.set_xlim(0, 1)

    ax_ins = ax.inset_axes([0.35, 0.45, 0.38, 0.45])
    zoom_lo, zoom_hi = max(0, thr - 0.25), min(1, thr + 0.25)
    zoom_bins = np.linspace(zoom_lo, zoom_hi, 31)
    m_zoom  = michel_scores[(michel_scores >= zoom_lo) & (michel_scores <= zoom_hi)]
    nm_zoom = nomichel_scores[(nomichel_scores >= zoom_lo) & (nomichel_scores <= zoom_hi)]
    ax_ins.hist(nm_zoom, bins=zoom_bins, color=COLORS["noMichel"], alpha=0.7,
                density=True, edgecolor="white", lw=0.3)
    ax_ins.hist(m_zoom, bins=zoom_bins, color=COLORS["Michel"], alpha=0.7,
                density=True, edgecolor="white", lw=0.3)
    ax_ins.axvline(thr, color=COLORS["threshold"], lw=1.5, ls="--")
    ax_ins.set_title("Zoom near threshold", fontsize=8)
    ax_ins.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "01_score_distributions.png"))


# 02 — Confusion matrix
def plot_confusion_matrix(ev: pd.DataFrame, plots_dir: str):
    tp = int(ev["cumulative_TP"].iloc[-1])
    tn = int(ev["cumulative_TN"].iloc[-1])
    fp = int(ev["cumulative_FP"].iloc[-1])
    fn = int(ev["cumulative_FN"].iloc[-1])

    cm_raw  = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True).clip(1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cm, title, fmt in zip(
        axes,
        [cm_raw, cm_norm],
        ["Confusion Matrix (raw counts)", "Confusion Matrix (row-normalised)"],
        [".0f", ".3f"]
    ):
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: no-Michel", "Pred: Michel"])
        ax.set_yticklabels(["True: no-Michel", "True: Michel"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#222222",
                        fontsize=13, fontweight="bold")
        ax.grid(False)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    bacc = 0.5 * (sensitivity + specificity)
    fig.suptitle(
        f"Sensitivity={sensitivity:.3f}   Specificity={specificity:.3f}   "
        f"Balanced Acc={bacc:.3f}   "
        f"Pipeline fired={tp+fp}  (TP={tp} FP={fp}  FN={fn})",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "02_confusion_matrix.png"))


# 03 — Cumulative outcomes over simulated time
def plot_cumulative_outcomes(ev: pd.DataFrame, plots_dir: str):
    t = ev["sim_time_s"].values / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, col, label, color in zip(
        axes.flat,
        ["cumulative_TP", "cumulative_FP", "cumulative_FN", "cumulative_TN"],
        ["True Positives (Michel, fired)",
         "False Positives (no-Michel, fired)",
         "False Negatives (Michel, missed)",
         "True Negatives (no-Michel, silent)"],
        [COLORS["TP"], COLORS["FP"], COLORS["FN"], COLORS["TN"]],
    ):
        ax.plot(t, ev[col].values, color=color, lw=1.8)
        ax.fill_between(t, ev[col].values, alpha=0.15, color=color)
        ax.set_ylabel("Cumulative count")
        ax.set_title(label)

    for ax in axes[1, :]:
        ax.set_xlabel("Simulated time (min)")

    fig.suptitle("Cumulative Confusion Outcomes over Simulated Time", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "03_cumulative_outcomes.png"))


# 04 — Rolling sensitivity & specificity
def plot_rolling_metrics(ev: pd.DataFrame, plots_dir: str, window: int = 100):
    t = ev["sim_time_s"].values / 60.0

    def _rolling_sensitivity(df, w):
        tp_roll = df["cumulative_TP"].diff().fillna(df["cumulative_TP"]).rolling(w, min_periods=1).sum()
        fn_roll = df["cumulative_FN"].diff().fillna(df["cumulative_FN"]).rolling(w, min_periods=1).sum()
        return tp_roll / (tp_roll + fn_roll).clip(lower=1e-9)

    def _rolling_specificity(df, w):
        tn_roll = df["cumulative_TN"].diff().fillna(df["cumulative_TN"]).rolling(w, min_periods=1).sum()
        fp_roll = df["cumulative_FP"].diff().fillna(df["cumulative_FP"]).rolling(w, min_periods=1).sum()
        return tn_roll / (tn_roll + fp_roll).clip(lower=1e-9)

    sens = _rolling_sensitivity(ev, window)
    spec = _rolling_specificity(ev, window)
    bacc = 0.5 * (sens + spec)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, sens, color=COLORS["TP"],   lw=1.8, label="Sensitivity (recall for Michel)")
    ax.plot(t, spec, color=COLORS["TN"],   lw=1.8, label="Specificity (recall for no-Michel)")
    ax.plot(t, bacc, color=COLORS["mean"], lw=1.8, ls="--", label="Balanced accuracy")
    ax.axhline(float(sens.mean()), color=COLORS["TP"], lw=0.8, ls=":", alpha=0.6)
    ax.axhline(float(spec.mean()), color=COLORS["TN"], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Simulated time (min)")
    ax.set_ylabel("Metric value")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(f"Rolling Metrics (window = {window} events)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "04_rolling_metrics.png"))


# 05 — Memory time series
def plot_memory_timeseries(mem: pd.DataFrame, plots_dir: str):
    idx = np.arange(1, len(mem) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(idx, mem["baseline_gpu_gb"], color=COLORS["baseline"], lw=1.2, ls="-.",
            zorder=1, label="Pre-event GPU allocated")
    ax.plot(idx, mem["combined_peak_gpu_gb"], color=COLORS["combined"], lw=2.2,
            zorder=2, label="Combined event peak GPU")
    ax.plot(idx, mem["gpu_allocated_gb"], color=COLORS["GPU"], lw=1.8,
            zorder=2, label="Post-run GPU allocated")
    # dotted/dashed on top
    ax.plot(idx, mem["classifier_peak_gpu_gb"], color=COLORS["classifier"], lw=1.5, ls=":",
            zorder=3, label="Classifier peak GPU")
    ax.plot(idx, mem["pipeline_peak_gpu_gb"], color=COLORS["pipeline"], lw=1.5, ls="--",
            zorder=4, label="Direction pipeline peak GPU")
    ax.set_ylabel("GPU Memory (GB)", color=COLORS["GPU"])
    ax.tick_params(axis="y", colors=COLORS["GPU"])
    ax.set_title("GPU memory per fired run (pre-event, stage peaks, post-run)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(idx, mem["baseline_cpu_rss_gb"], color=COLORS["baseline"], lw=1.2, ls="-.",
            label="Pre-event CPU RSS")
    ax.plot(idx, mem["classifier_peak_cpu_rss_gb"], color=COLORS["classifier"], lw=1.5, ls=":",
            label="Classifier peak CPU RSS")
    ax.plot(idx, mem["pipeline_peak_cpu_rss_gb"], color=COLORS["pipeline"], lw=1.5, ls="--",
            label="Direction pipeline peak CPU RSS")
    ax.plot(idx, mem["combined_peak_cpu_rss_gb"], color=COLORS["combined"], lw=2.2,
            label="Combined event peak CPU RSS")
    ax.plot(idx, mem["cpu_rss_gb"], color=COLORS["CPU"], lw=1.8,
            label="Post-run CPU RSS")
    ax.set_ylabel("CPU RSS (GB)", color=COLORS["CPU"])
    ax.tick_params(axis="y", colors=COLORS["CPU"])
    ax.set_xlabel("Fired run index")
    ax.set_title("CPU memory per fired run (pre-event, stage peaks, post-run)")
    ax.legend(loc="upper left")

    fig.suptitle("Stage-separated memory views", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "05_memory_timeseries.png"))


# 05b — Utilization across all events (simulated time axis)
def plot_utilization_all_events(ev: pd.DataFrame, plots_dir: str):
    t = ev["sim_time_s"].values / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, ev["pipeline_gpu_util_max_pct"].values,
            color=COLORS["pipeline"], lw=1.0, alpha=0.65, zorder=2,
            label="Pipeline max GPU util (0 when not fired)")
    ax.plot(t, ev["classifier_gpu_util_max_pct"].values,
            color=COLORS["classifier"], lw=1.8, alpha=0.98, zorder=3,
            label="Classifier max GPU util")
    ax.plot(t, ev["combined_gpu_util_max_pct"].values,
            color=COLORS["combined"], lw=1.8, ls="--", alpha=0.9, zorder=4,
            label="Combined event max GPU util")
    ymin, ymax = _nice_util_ylim([
        ev["classifier_gpu_util_max_pct"].values,
        ev["pipeline_gpu_util_max_pct"].values,
        ev["combined_gpu_util_max_pct"].values,
    ])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("Device GPU utilization across all events (simulated time axis)")
    # legend: classifier, pipeline, combined
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in [1, 0, 2]], [labels[i] for i in [1, 0, 2]], loc="upper left")

    ax = axes[1]
    ax.plot(t, ev["combined_cpu_proc_util_max_pct"].values,
            color=COLORS["combined"], lw=0.9, alpha=0.45, zorder=1,
            label="Combined event max process CPU util")
    ax.plot(t, ev["pipeline_cpu_proc_util_max_pct"].values,
            color=COLORS["pipeline"], lw=1.0, alpha=0.65, zorder=2,
            label="Pipeline max process CPU util (0 when not fired)")
    ax.plot(t, ev["classifier_cpu_proc_util_max_pct"].values,
            color=COLORS["classifier"], lw=1.8, alpha=0.98, zorder=3,
            label="Classifier max process CPU util")
    ymin, ymax = _nice_util_ylim([
        ev["classifier_cpu_proc_util_max_pct"].values,
        ev["pipeline_cpu_proc_util_max_pct"].values,
        ev["combined_cpu_proc_util_max_pct"].values,
    ])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Simulated time (min)")
    ax.set_ylabel("Process CPU Utilization (%)")
    ax.set_title("Process CPU utilization across all events (simulated time axis)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in [2, 1, 0]], [labels[i] for i in [2, 1, 0]], loc="upper left")

    fig.suptitle("Utilization across all events", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "05b_utilization_all_events.png"))


# 05c — Utilization across fired runs only
def plot_utilization_fired_runs(ev: pd.DataFrame, plots_dir: str):
    fired = ev.loc[ev["pipeline_fired"] == 1].copy()
    if len(fired) == 0:
        print("  [skip] no fired events for fired-run utilization plot")
        return

    idx = np.arange(1, len(fired) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(idx, fired["pipeline_gpu_util_mean_pct"].values,
            color=COLORS["pipeline"], lw=1.0, alpha=0.55, zorder=2,
            label="Pipeline mean GPU util")
    ax.plot(idx, fired["pipeline_gpu_util_max_pct"].values,
            color=COLORS["pipeline"], lw=1.1, ls="--", alpha=0.75, zorder=2,
            label="Pipeline max GPU util")
    ax.plot(idx, fired["classifier_gpu_util_mean_pct"].values,
            color=COLORS["classifier"], lw=1.6, alpha=0.95, zorder=3,
            label="Classifier mean GPU util")
    ax.plot(idx, fired["classifier_gpu_util_max_pct"].values,
            color=COLORS["classifier"], lw=1.8, ls="--", alpha=0.98, zorder=4,
            label="Classifier max GPU util")
    ymin, ymax = _nice_util_ylim([
        fired["classifier_gpu_util_mean_pct"].values,
        fired["classifier_gpu_util_max_pct"].values,
        fired["pipeline_gpu_util_mean_pct"].values,
        fired["pipeline_gpu_util_max_pct"].values,
    ])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("Device GPU utilization across fired runs")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1]
    ax.plot(idx, fired["pipeline_cpu_proc_util_mean_pct"].values,
            color=COLORS["pipeline"], lw=1.0, alpha=0.55, zorder=2,
            label="Pipeline mean process CPU util")
    ax.plot(idx, fired["pipeline_cpu_proc_util_max_pct"].values,
            color=COLORS["pipeline"], lw=1.1, ls="--", alpha=0.75, zorder=2,
            label="Pipeline max process CPU util")
    ax.plot(idx, fired["classifier_cpu_proc_util_mean_pct"].values,
            color=COLORS["classifier"], lw=1.6, alpha=0.95, zorder=3,
            label="Classifier mean process CPU util")
    ax.plot(idx, fired["classifier_cpu_proc_util_max_pct"].values,
            color=COLORS["classifier"], lw=1.8, ls="--", alpha=0.98, zorder=4,
            label="Classifier max process CPU util")
    ymin, ymax = _nice_util_ylim([
        fired["classifier_cpu_proc_util_mean_pct"].values,
        fired["classifier_cpu_proc_util_max_pct"].values,
        fired["pipeline_cpu_proc_util_mean_pct"].values,
        fired["pipeline_cpu_proc_util_max_pct"].values,
    ])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Fired run index")
    ax.set_ylabel("Process CPU Utilization (%)")
    ax.set_title("Process CPU utilization across fired runs")
    ax.legend(loc="upper left", ncol=2)

    fig.suptitle("Sampled stage utilization across fired runs", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "05c_utilization_fired_runs.png"))


# 06a — GPU memory distributions
def plot_gpu_memory_distributions(mem: pd.DataFrame, plots_dir: str):
    cols = [
        "baseline_gpu_gb",
        "classifier_peak_gpu_gb",
        "pipeline_peak_gpu_gb",
        "combined_peak_gpu_gb",
        "gpu_allocated_gb",
    ]
    labels = [
        "Pre-event GPU\nallocated",
        "Classifier\npeak GPU",
        "Direction pipeline\npeak GPU",
        "Combined event\npeak GPU",
        "Post-run GPU\nallocated",
    ]
    colors = [
        COLORS["baseline"],
        COLORS["classifier"],
        COLORS["pipeline"],
        COLORS["combined"],
        COLORS["GPU"],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = np.array(axes).flatten()

    for ax, col, label, color in zip(axes, cols, labels, colors):
        data = mem[col].dropna().values
        bins, xlim = _focused_hist_params(data, nbins=20)
        ax.hist(data, bins=bins, color=color, edgecolor="white", lw=0.4, alpha=0.85)
        ax.axvline(data.mean(), color=COLORS["mean"], lw=2, ls="--",
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(np.median(data), color=COLORS["threshold"], lw=1.5, ls=":",
                   label=f"Median={np.median(data):.3f}")
        ax.set_xlim(*xlim)
        ax.set_xlabel("Memory (GB)")
        ax.set_ylabel("Fired runs")
        ax.set_title(label)
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("GPU memory distributions by stage", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "06a_gpu_memory_distributions.png"))


# 06b — CPU memory distributions
def plot_cpu_memory_distributions(mem: pd.DataFrame, plots_dir: str):
    cols = [
        "baseline_cpu_rss_gb",
        "classifier_peak_cpu_rss_gb",
        "pipeline_peak_cpu_rss_gb",
        "combined_peak_cpu_rss_gb",
        "cpu_rss_gb",
    ]
    labels = [
        "Pre-event\nCPU RSS",
        "Classifier\npeak CPU RSS",
        "Direction pipeline\npeak CPU RSS",
        "Combined event\npeak CPU RSS",
        "Post-run\nCPU RSS",
    ]
    colors = [
        COLORS["baseline"],
        COLORS["classifier"],
        COLORS["pipeline"],
        COLORS["combined"],
        COLORS["CPU"],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = np.array(axes).flatten()

    for ax, col, label, color in zip(axes, cols, labels, colors):
        data = mem[col].dropna().values
        bins, xlim = _focused_hist_params(data, nbins=20)
        ax.hist(data, bins=bins, color=color, edgecolor="white", lw=0.4, alpha=0.85)
        ax.axvline(data.mean(), color=COLORS["mean"], lw=2, ls="--",
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(np.median(data), color=COLORS["threshold"], lw=1.5, ls=":",
                   label=f"Median={np.median(data):.3f}")
        ax.set_xlim(*xlim)
        ax.set_xlabel("Memory (GB)")
        ax.set_ylabel("Fired runs")
        ax.set_title(label)
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("CPU memory distributions by stage", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "06b_cpu_memory_distributions.png"))


# 07 — Timing distributions
def plot_timing_distributions(ev: pd.DataFrame, plots_dir: str, clip_percentile: float = 99.5):
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ct = ev["classifier_time_s"].dropna().values * 1000.0
    ct_disp, ct_hi, ct_nclip = _clip_for_display(ct, clip_percentile)
    bins = max(15, len(ct_disp) // 30)
    ax0.hist(ct_disp, bins=bins, color=COLORS["classifier"], edgecolor="white", lw=0.4, alpha=0.85)
    ax0.axvline(ct.mean(), color=COLORS["mean"], lw=2, ls="--",
                label=f"Mean = {ct.mean():.1f} ms")
    ax0.axvline(np.median(ct), color=COLORS["threshold"], lw=1.5, ls=":",
                label=f"Median = {np.median(ct):.1f} ms")
    ax0.set_xlim(0, ct_hi)
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Events")
    ax0.set_title(
        "Classifier Inference\n"
        f"Time (all events)\n"
        f"{ct_nclip} outlier(s) clipped > {ct_hi:.1f} ms"
    )
    ax0.legend()

    ax1 = fig.add_subplot(gs[1])
    pt = ev.loc[ev["pipeline_fired"] == 1, "wall_time_pipeline_s"].dropna().values
    pt_disp, pt_hi, pt_nclip = _clip_for_display(pt, clip_percentile)
    bins = max(10, len(pt_disp) // 5)
    ax1.hist(pt_disp, bins=bins, color=COLORS["pipeline"], edgecolor="white", lw=0.4, alpha=0.85)
    ax1.axvline(pt.mean(), color=COLORS["mean"], lw=2, ls="--",
                label=f"Mean = {pt.mean():.2f} s")
    ax1.axvline(np.median(pt), color=COLORS["threshold"], lw=1.5, ls=":",
                label=f"Median = {np.median(pt):.2f} s")
    ax1.set_xlim(0, pt_hi)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Fired runs")
    ax1.set_title(
        "Full Direction Pipeline\n"
        "Wall-clock Time\n"
        f"{pt_nclip} outlier(s) clipped > {pt_hi:.2f} s"
    )
    ax1.legend()

    ax2 = fig.add_subplot(gs[2])
    data = [ct, pt * 1000.0]
    labels = ["Classifier\n(ms)", "Direction\nPipeline\n(ms)"]
    bp = ax2.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
    )
    for patch, c in zip(bp["boxes"], [COLORS["classifier"], COLORS["pipeline"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax2.set_yscale("log")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_title("Timing Comparison")
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle("Pipeline Timing Breakdown", fontsize=13)
    _save(fig, os.path.join(plots_dir, "07_timing_distributions.png"))


# 07b — Fired-event total timing (broken-axis boxplot)
def plot_fired_event_total_timing(ev: pd.DataFrame, plots_dir: str):
    fired = ev.loc[ev["pipeline_fired"] == 1].copy()
    if len(fired) == 0:
        print("  [skip] no fired events for total timing comparison plot")
        return

    classifier_fired_ms = fired["classifier_time_s"].dropna().values * 1000.0
    pipeline_fired_ms   = fired["wall_time_pipeline_s"].dropna().values * 1000.0
    total_fired_ms      = classifier_fired_ms + pipeline_fired_ms

    data = [classifier_fired_ms, pipeline_fired_ms, total_fired_ms]
    labels = [
        "ML Michel\nTrigger\n(ms)",
        "Full\nPipeline\n(ms)",
        "Total\nTrigger + Pipeline\n(ms)",
    ]
    box_colors = [COLORS["classifier"], COLORS["pipeline"], COLORS["combined"]]

    def _padded_limits(arr, frac_pad=0.15, min_pad=1.0):
        arr = np.asarray(arr, dtype=float)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        spread = hi - lo
        pad = max(spread * frac_pad, min_pad)
        return lo - pad, hi + pad

    low_lo, low_hi = _padded_limits(classifier_fired_ms, frac_pad=0.20, min_pad=1.0)

    upper_vals = np.concatenate([pipeline_fired_ms, total_fired_ms])
    up_lo, up_hi = _padded_limits(upper_vals, frac_pad=0.12, min_pad=5.0)

    pos_pipeline = pipeline_fired_ms[pipeline_fired_ms > 0]
    if pos_pipeline.size > 0:
        up_lo = max(up_lo, 0.85 * float(np.percentile(pos_pipeline, 1)))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    common_bp_kwargs = dict(
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
    )

    bp_top = ax_top.boxplot(data, **common_bp_kwargs)
    bp_bot = ax_bot.boxplot(data, **common_bp_kwargs)

    for bp in (bp_top, bp_bot):
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)

    ax_bot.set_ylim(low_lo, low_hi)
    ax_top.set_ylim(up_lo, up_hi)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.008
    kwargs = dict(color="k", clip_on=False, lw=1.2)

    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

    ax_top.set_title("Fired-Event Timing Comparison")
    ax_bot.set_ylabel("Time (ms)")
    ax_bot.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "07b_fired_event_total_timing.png"))


# 08 — Cos theta histograms
def plot_costheta_histograms(dirlog: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bins = np.linspace(-1, 1, 40)

    for ax, v in zip(axes, ["U", "V", "Z"]):
        col = f"cos_theta_{v}"
        tp_rows = dirlog[dirlog["outcome"] == "TP"][col].dropna()

        vals = tp_rows.values
        frac_correct = float((vals > 0).mean())
        mean_ct = float(vals.mean())
        median_ct = float(np.median(vals))
        mean_angle = math.degrees(math.acos(np.clip(mean_ct, -1, 1)))
        median_angle = math.degrees(math.acos(np.clip(median_ct, -1, 1)))

        ax.hist(vals, bins=bins, color=VIEW_COLORS[v], edgecolor="white", lw=0.4, alpha=0.85)
        ax.axvline(mean_ct, color=COLORS["mean"], lw=2, ls="-",
                   label=f"Mean = {mean_ct:.3f} ({mean_angle:.1f}°)")
        ax.axvline(median_ct, color=COLORS["threshold"], lw=1.8, ls="--",
                   label=f"Median = {median_ct:.3f} ({median_angle:.1f}°)")
        ax.set_xlabel(r"$\cos\theta$  (pred · GT)")
        ax.set_ylabel("TP events")
        ax.set_xlim(-1, 1)
        ax.set_title(f"{v} view  |  n={len(vals)}  |  {100*frac_correct:.1f}% correct (cosθ > 0)")
        ax.legend(fontsize=8)

    fig.suptitle("Direction Prediction Accuracy: cos θ per View (TP events)", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "08_costheta_histograms.png"))


# 09 — Cos theta vs classifier score
def plot_costheta_vs_score(dirlog: pd.DataFrame, plots_dir: str):
    tp = dirlog[dirlog["outcome"] == "TP"].copy()
    if len(tp) == 0:
        print("  [skip] no TP events for cos_theta vs score plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, v in zip(axes, ["U", "V", "Z"]):
        col = f"cos_theta_{v}"
        sub = tp[[col, "classifier_score"]].dropna()
        sc = ax.scatter(sub["classifier_score"], sub[col],
                        c=sub[col], cmap="RdYlGn", vmin=-1, vmax=1,
                        alpha=0.6, s=18, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="cos θ")
        ax.axhline(0, color="#888888", lw=1, ls="--")
        ax.set_xlabel("Classifier score")
        ax.set_ylabel(r"$\cos\theta$")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{v} view — cos θ vs classifier score")

        r, _ = scipy_stats.pearsonr(sub["classifier_score"], sub[col])
        ax.text(0.05, 0.05, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, color="#333333")

    fig.suptitle("Direction Accuracy vs Classifier Confidence (TP events)", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "09_costheta_vs_score.png"))


# 10 — Event inter-arrival times
def plot_interarrival_times(ev: pd.DataFrame, plots_dir: str, michel_rate: float,
                            nomichel_rate: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, label, lbl_str, rate, color in zip(
        axes,
        [ev["true_label"] == 1, ev["true_label"] == 0],
        ["Michel events", "No-Michel events"],
        [michel_rate, nomichel_rate],
        [COLORS["Michel"], COLORS["noMichel"]],
    ):
        t = ev.loc[label, "sim_time_s"].values
        inter = np.diff(t)
        bins  = np.linspace(0, np.percentile(inter, 99), 40)
        ax.hist(inter, bins=bins, color=color, edgecolor="white", lw=0.4,
                alpha=0.85, density=True, label="Observed")

        x = np.linspace(0, bins[-1], 300)
        ax.plot(x, rate * np.exp(-rate * x), color="#333333", lw=2, ls="--",
                label=f"Exp(λ={rate:.4f})")
        ax.set_xlabel("Inter-arrival time (s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{lbl_str}\nMean = {inter.mean():.1f}s  |  Expected = {1/rate:.1f}s")
        ax.legend()

    fig.suptitle("Event Inter-Arrival Time: Observed vs Theoretical Poisson", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "10_interarrival_times.png"))


# 11 — Summary stats table
def plot_summary_stats(ev: pd.DataFrame, mem: pd.DataFrame,
                       dirlog: pd.DataFrame | None, plots_dir: str):
    tp = int(ev["cumulative_TP"].iloc[-1])
    tn = int(ev["cumulative_TN"].iloc[-1])
    fp = int(ev["cumulative_FP"].iloc[-1])
    fn = int(ev["cumulative_FN"].iloc[-1])
    n  = tp + tn + fp + fn

    sens  = tp / max(tp + fn, 1)
    spec  = tn / max(tn + fp, 1)
    bacc  = 0.5 * (sens + spec)
    prec  = tp / max(tp + fp, 1)
    f1    = 2 * prec * sens / max(prec + sens, 1e-12)

    pt = ev.loc[ev["pipeline_fired"] == 1, "wall_time_pipeline_s"].dropna()
    ct = ev["classifier_time_s"].dropna() * 1000.0

    rows = [
        ["Metric",                             "Value"],
        ["Total events",                       f"{n}"],
        ["Michel (true)",                      f"{tp + fn}"],
        ["No-Michel (true)",                   f"{tn + fp}"],
        ["", ""],
        ["True Positives (TP)",                f"{tp}"],
        ["True Negatives (TN)",                f"{tn}"],
        ["False Positives (FP)",               f"{fp}"],
        ["False Negatives (FN)",               f"{fn}"],
        ["", ""],
        ["Sensitivity (Michel recall)",        f"{sens:.4f}"],
        ["Specificity (no-Michel recall)",     f"{spec:.4f}"],
        ["Balanced accuracy",                  f"{bacc:.4f}"],
        ["Precision",                          f"{prec:.4f}"],
        ["F1 score",                           f"{f1:.4f}"],
        ["", ""],
        ["Pipeline fires (TP + FP)",           f"{tp + fp}"],
        ["Missed Michels (FN)",                f"{fn}"],
        ["", ""],
        ["Classifier time mean ± std",         f"{ct.mean():.1f} ± {ct.std():.1f} ms"],
        ["Pipeline time mean ± std",           f"{pt.mean():.2f} ± {pt.std():.2f} s"],
        ["Pipeline time range",                f"[{pt.min():.2f}, {pt.max():.2f}] s"],
        ["", ""],
        ["Pre-event GPU allocated",            f"{mem['baseline_gpu_gb'].mean():.3f} GB"],
        ["Classifier peak GPU",                f"{mem['classifier_peak_gpu_gb'].mean():.3f} GB"],
        ["Direction pipeline peak GPU",        f"{mem['pipeline_peak_gpu_gb'].mean():.3f} GB"],
        ["Combined event peak GPU",            f"{mem['combined_peak_gpu_gb'].mean():.3f} GB"],
        ["Post-run GPU allocated",             f"{mem['gpu_allocated_gb'].mean():.3f} GB"],
        ["", ""],
        ["Pre-event CPU RSS",                  f"{mem['baseline_cpu_rss_gb'].mean():.3f} GB"],
        ["Classifier peak CPU RSS",            f"{mem['classifier_peak_cpu_rss_gb'].mean():.3f} GB"],
        ["Direction pipeline peak CPU RSS",    f"{mem['pipeline_peak_cpu_rss_gb'].mean():.3f} GB"],
        ["Combined event peak CPU RSS",        f"{mem['combined_peak_cpu_rss_gb'].mean():.3f} GB"],
        ["Post-run CPU RSS",                   f"{mem['cpu_rss_gb'].mean():.3f} GB"],
    ]

    if dirlog is not None and len(dirlog) > 0:
        for v in ["U", "V", "Z"]:
            col = f"cos_theta_{v}"
            tp_ct = dirlog.loc[dirlog["outcome"] == "TP", col].dropna()
            if len(tp_ct) > 0:
                frac = float((tp_ct > 0).mean())
                rows.append([f"Cos θ > 0 ({v} view, TP)",
                             f"{100*frac:.1f}%  (mean cosθ = {tp_ct.mean():.3f})"])

    fig_h = max(6, len(rows) * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_axis_off()

    table_data = [r[1] for r in rows]
    row_labels = [r[0] for r in rows]

    table = ax.table(
        cellText=[[v] for v in table_data],
        rowLabels=row_labels,
        colLabels=["Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif table_data[row - 1] == "":
            cell.set_facecolor("#ecf0f1")
            cell.set_height(0.01)
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    ax.set_title("Simulation Summary", fontsize=14, pad=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "11_summary_stats.png"))


# PDF gallery — fired events
def make_pdf_gallery(dirlog: pd.DataFrame, raw_dir: str, preproc_dir: str,
                     plots_dir: str, n: int = 20):
    tp_rows = dirlog[dirlog["outcome"] == "TP"].copy()
    fp_rows = dirlog[dirlog["outcome"] == "FP"].copy()

    n_tp = min(len(tp_rows), max(0, n - max(0, n - len(tp_rows))))
    n_fp = min(len(fp_rows), n - n_tp)
    n_tp = n - n_fp

    sel_tp = tp_rows.sample(n=min(n_tp, len(tp_rows)), random_state=0) if n_tp > 0 else pd.DataFrame()
    sel_fp = fp_rows.sample(n=min(n_fp, len(fp_rows)), random_state=0) if n_fp > 0 else pd.DataFrame()
    gallery_df = pd.concat([sel_tp, sel_fp]).sort_values("event_count")

    if len(gallery_df) == 0:
        print("  [skip] no events to plot in gallery")
        return

    pdf_path = os.path.join(plots_dir, "gallery_fired_events.pdf")
    with PdfPages(pdf_path) as pdf:
        for _, row in gallery_df.iterrows():
            ev_num = int(row["event_count"])
            raw_path     = os.path.join(raw_dir,     f"event_{ev_num:06d}.npz")
            preproc_path = os.path.join(preproc_dir, f"event_{ev_num:06d}.npz")

            if not os.path.exists(raw_path) or not os.path.exists(preproc_path):
                continue

            raw_npz     = np.load(raw_path, allow_pickle=True)
            preproc_npz = np.load(preproc_path, allow_pickle=True)
            raw_img     = raw_npz["Z_data"].squeeze().astype(np.float32)
            seg_img     = preproc_npz["Z_data"].squeeze().astype(np.float32)

            outcome = str(row["outcome"])
            score   = float(row["classifier_score"])

            ux_pred = float(row["ux_corr_Z"]) if "ux_corr_Z" in row.index else None
            uy_pred = float(row["uy_corr_Z"]) if "uy_corr_Z" in row.index else None

            has_gt = False
            gt_ux = gt_uy = None
            if "gt_ux_Z" in row.index and not pd.isna(row["gt_ux_Z"]):
                gt_ux  = float(row["gt_ux_Z"])
                gt_uy  = float(row["gt_uy_Z"])
                has_gt = True

            cos_z = float(row["cos_theta_Z"]) if ("cos_theta_Z" in row.index
                                                  and not pd.isna(row["cos_theta_Z"])) else None

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            outcome_color = COLORS.get(outcome, "#333333")

            for ax, img, title_suffix in zip(
                axes,
                [raw_img, seg_img],
                ["Raw detector readout", "After seg + Bragg suppression"],
            ):
                H, W = img.shape
                cx, cy = W / 2, H / 2
                L = min(H, W) * 0.20

                if ax == axes[1]:
                    masked = np.ma.masked_where(img == 0, img)
                    cmap   = plt.cm.viridis.copy()
                    cmap.set_bad("white")
                    ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto")
                else:
                    ax.imshow(img, origin="lower", cmap="viridis", aspect="auto")

                if ux_pred is not None and uy_pred is not None:
                    ax.annotate("", xy=(cx + L * ux_pred, cy + L * uy_pred),
                                xytext=(cx, cy),
                                arrowprops=dict(arrowstyle="-|>", color=COLORS["pred"],
                                                lw=2.5, mutation_scale=18))
                if has_gt:
                    ax.annotate("", xy=(cx + L * gt_ux, cy + L * gt_uy),
                                xytext=(cx, cy),
                                arrowprops=dict(arrowstyle="-|>", color=COLORS["GT"],
                                                lw=2.5, mutation_scale=18))

                ax.set_title(f"Z view — {title_suffix}", fontsize=10)
                ax.set_xlabel("Time tick (drift)")
                ax.set_ylabel("Wire number")
                ax.grid(False)

                legend_els = [Patch(facecolor=COLORS["pred"], label="Prediction")]
                if has_gt:
                    legend_els.insert(0, Patch(facecolor=COLORS["GT"], label="GT direction"))
                ax.legend(handles=legend_els, loc="upper right", fontsize=8)

            cos_str = (
                f"cos θ (Z) = {cos_z:.3f}  ({math.degrees(math.acos(np.clip(cos_z, -1, 1))):.1f}°)"
                if cos_z is not None else "no GT direction (FP event)"
            )

            fig.suptitle(
                f"Event {ev_num:06d}   outcome: {outcome}   score: {score:.4f}   {cos_str}",
                fontsize=11, color=outcome_color, fontweight="bold",
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"  saved → gallery_fired_events.pdf  ({len(gallery_df)} events)")


# Main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT / "simulation_output"))
    ap.add_argument("--michel_rate", type=float, default=1.0 / 60.0,
                    help="Used for theoretical Poisson curve (default 1/60 s)")
    ap.add_argument("--nomichel_rate", type=float, default=1.0 / 10.0,
                    help="Used for theoretical Poisson curve (default 1/10 s)")
    ap.add_argument("--gallery_n", type=int, default=20)
    ap.add_argument("--rolling_window", type=int, default=100)
    ap.add_argument("--timing_clip_percentile", type=float, default=99.5)
    ap.add_argument("--no_util_all_events", action="store_true",
                    help="Skip the utilization plot over all events.")
    ap.add_argument("--no_util_fired_runs", action="store_true",
                    help="Skip the utilization plot over fired runs only.")
    return ap.parse_args()


def main():
    args = parse_args()
    out  = args.output_dir

    plots_dir = os.path.join(out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    event_log_path     = os.path.join(out, "event_log.csv")
    memory_log_path    = os.path.join(out, "memory_log.csv")
    direction_log_path = os.path.join(out, "direction_log.csv")

    if not os.path.exists(event_log_path):
        raise FileNotFoundError(f"event_log.csv not found at {event_log_path}")
    if not os.path.exists(memory_log_path):
        raise FileNotFoundError(f"memory_log.csv not found at {memory_log_path}")

    print("Loading CSVs...")
    ev  = _prepare_event_df(pd.read_csv(event_log_path))
    mem = _prepare_memory_df(pd.read_csv(memory_log_path))

    dirlog = None
    if os.path.exists(direction_log_path):
        dirlog = pd.read_csv(direction_log_path)
        print(f"  direction_log: {len(dirlog)} fired events")
    else:
        print("  direction_log.csv not found — skipping direction plots")

    print(f"  event_log: {len(ev)} events")
    print(f"  memory_log: {len(mem)} fired runs")
    print(f"Generating plots → {plots_dir}")

    print("  01 score distributions...")
    plot_score_distributions(ev, plots_dir)

    print("  02 confusion matrix...")
    plot_confusion_matrix(ev, plots_dir)

    print("  03 cumulative outcomes...")
    plot_cumulative_outcomes(ev, plots_dir)

    print("  04 rolling metrics...")
    plot_rolling_metrics(ev, plots_dir, window=args.rolling_window)

    print("  05 memory time series...")
    plot_memory_timeseries(mem, plots_dir)

    if not args.no_util_all_events:
        print("  05b utilization over all events...")
        plot_utilization_all_events(ev, plots_dir)
    else:
        print("  05b [skipped by --no_util_all_events]")

    if not args.no_util_fired_runs:
        print("  05c utilization over fired runs...")
        plot_utilization_fired_runs(ev, plots_dir)
    else:
        print("  05c [skipped by --no_util_fired_runs]")

    print("  06a GPU memory distributions...")
    plot_gpu_memory_distributions(mem, plots_dir)

    print("  06b CPU memory distributions...")
    plot_cpu_memory_distributions(mem, plots_dir)

    print("  07 timing distributions...")
    plot_timing_distributions(ev, plots_dir, clip_percentile=args.timing_clip_percentile)

    print("  07b fired-event total timing...")
    plot_fired_event_total_timing(ev, plots_dir)

    if dirlog is not None:
        print("  08 cos theta histograms...")
        plot_costheta_histograms(dirlog, plots_dir)

        print("  09 cos theta vs score...")
        plot_costheta_vs_score(dirlog, plots_dir)
    else:
        print("  08-09 [skipped — no direction_log.csv]")

    print("  10 inter-arrival times...")
    plot_interarrival_times(ev, plots_dir, args.michel_rate, args.nomichel_rate)

    print("  11 summary stats table...")
    plot_summary_stats(ev, mem, dirlog, plots_dir)

    raw_dir     = os.path.join(out, "raw_events")
    preproc_dir = os.path.join(out, "pipeline_input")
    if dirlog is not None and os.path.isdir(raw_dir) and os.path.isdir(preproc_dir):
        print(f"  PDF gallery ({args.gallery_n} events)...")
        make_pdf_gallery(dirlog, raw_dir, preproc_dir, plots_dir, n=args.gallery_n)
    else:
        print("  PDF gallery [skipped — need direction_log.csv + raw_events/ + pipeline_input/]")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
