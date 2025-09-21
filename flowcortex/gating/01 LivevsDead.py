
"""
Live/Dead gating using FVS viability dyes.
Applies ensemble thresholding: KDE valley, GMM, percentile rule.
Labels cells as live, dead, uncertain, or negative.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# ---------- helpers ----------

def compute_percentiles(signal):
    q05, q10, q25, q50, q75, q90, q99 = np.percentile(signal, [5, 10, 25, 50, 75, 90, 99])
    iqr = q75 - q25
    return q05, q10, q25, q50, q75, q90, q99, iqr

def kde_curve(signal, bandwidth="scott"):
    kde = gaussian_kde(signal, bw_method=bandwidth)
    x_vals = np.linspace(signal.min(), signal.max(), 1000)
    y_vals = kde(x_vals)
    return x_vals, y_vals

def find_kde_valley(x_vals, y_vals, gmm_means, depth_factor=0.15):
    mu1, mu2 = sorted(gmm_means)
    mask = (x_vals >= mu1) & (x_vals <= mu2)
    if mask.sum() < 10:
        return None
    valley_region = y_vals[mask]
    valley_x_range = x_vals[mask]
    min_idx = np.argmin(valley_region)
    valley_x = valley_x_range[min_idx]
    valley_y = valley_region[min_idx]
    left_peak = y_vals[x_vals < mu1].max() if np.any(x_vals < mu1) else 0
    right_peak = y_vals[x_vals > mu2].max() if np.any(x_vals > mu2) else 0
    min_peak = min(left_peak, right_peak)
    return valley_x if valley_y < (1 - depth_factor) * min_peak else None

def ensemble_threshold(signal, raw_signal, depth_factor=0.15):
    signal = np.asarray(signal)
    raw_signal = np.asarray(raw_signal)
    mask = np.isfinite(signal)
    signal, raw_signal = signal[mask], raw_signal[mask]

    if len(signal) < 50 or np.std(signal) < 0.1:
        return None, np.full(len(signal), "unlabeled")

    thresholds = []

    # --- GMM ---
    gmm = GaussianMixture(n_components=2, random_state=42).fit(signal.reshape(-1, 1))
    gmm_means = gmm.means_.flatten()
    if abs(np.diff(gmm_means))[0] > 0.3 and gmm.weights_.min() > 0.1:
        thresholds.append(np.mean(gmm_means))

    # --- KDE ---
    x_vals, y_vals = kde_curve(signal)
    kde_valley = find_kde_valley(x_vals, y_vals, gmm_means, depth_factor)
    if kde_valley is not None:
        thresholds.append(kde_valley)

    # --- Percentile / IQR ---
    _, q10, q25, q50, q75, q90, _, iqr = compute_percentiles(signal)
    if q90 / q10 > 1.3:
        thresholds.append(q50 + 0.5 * iqr)

    # --- Final threshold ---
    if thresholds:
        thr = np.median(thresholds)
        band = 0.1 * thr
        labels = []
        for rv, v in zip(raw_signal, signal):
            if rv <= 0:
                labels.append("negative")
            elif abs(v - thr) <= band:
                labels.append("uncertain")
            elif v > thr:
                labels.append("live")
            else:
                labels.append("dead")
        return thr, np.array(labels)
    else:
        return None, np.full(len(signal), "unlabeled")

# ---------- main function ----------

def process_viability(comp_csv, marker, cofactor=150.0, output_dir="out"):
    """
    Apply arcsinh transform, run ensemble threshold, save labeled CSV + plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(comp_csv)

    if marker not in df.columns:
        raise ValueError(f"Marker {marker} not in file {comp_csv}")

    raw_signal = df[marker].values
    df[marker + "_trans"] = np.arcsinh(raw_signal / cofactor)

    thr, labels = ensemble_threshold(df[marker + "_trans"].values, raw_signal)
    df["LiveDead_Label"] = labels

    # Save CSV
    out_file = os.path.join(output_dir, os.path.basename(comp_csv).replace(".csv", "_LD.csv"))
    df.to_csv(out_file, index=False)

    # Plot
    plt.figure(figsize=(8, 4))
    x_vals, y_vals = kde_curve(df[marker + "_trans"].values)
    plt.plot(x_vals, y_vals, color="blue")
    if thr is not None:
        plt.axvline(thr, color="red", linestyle="--", label="Threshold")
    plt.title("Live/Dead Ensemble Gating")
    plt.xlabel(marker + "_trans")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file.replace(".csv", "_plot.png"))
    plt.close()

    return out_file
