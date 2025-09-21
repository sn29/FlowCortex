import os
import numpy as np
import pandas as pd
from scipy.stats import skew


# === CONFIG ===
MAX_FILES = 8                   # default: subsample files for speed
SUBSAMPLE_PER_FILE = 50_000     # per file per channel
MIN_EVENTS_PER_CH = 1_000       # skip channels with fewer events


# ---------- helpers ----------
def _is_numeric(series):
    """Check if pandas series is numeric (float or int)."""
    return series.dtype.kind in "fc"


def pick_signal_columns(df):
    """
    Select fluorescence channels for cofactor estimation.
    Prefer *_comp columns; if none, fall back to numeric columns
    minus scatter and time channels.
    """
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    comp = [c for c in numeric if c.endswith("_comp")]
    cols = comp if comp else numeric
    drop_keys = ("FSC", "SSC", "Time", "time")
    return [c for c in cols if not any(k in c for k in drop_keys)]


def subsample(values, max_n):
    """Finite-only + random subsample for speed."""
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if x.size > max_n:
        idx = np.random.choice(x.size, max_n, replace=False)
        x = x[idx]
    return x


# ---------- arcsinh ----------
def initial_guess_arcsinh(x):
    """
    flowTrans-like seed:
    cofactor ≈ p99.5 / 5 (robust, compresses tails).
    """
    p995 = np.percentile(np.abs(x), 99.5)
    return max(p995 / 5.0, 1.0)


def stability_objective(x, c):
    """
    Objective to refine cofactor:
    J(c) = |skew(y)| + |corr(mean_bin(y), var_bin(y))|
    where y = arcsinh(x / c).
    """
    y = np.arcsinh(x / c)

    # 1) symmetry
    s = np.abs(skew(y, bias=False))

    # 2) variance stabilizing: mean-variance independence
    qs = np.quantile(y, np.linspace(0, 1, 51))
    bins = np.clip(np.digitize(y, qs, right=True), 1, 50)
    m, v = [], []
    for b in range(1, 51):
        mask = bins == b
        if mask.sum() > 50:
            m.append(y[mask].mean())
            v.append(y[mask].var(ddof=1))
    if len(m) > 2:
        corr = np.corrcoef(m, v)[0, 1]
        corr = 0.0 if np.isnan(corr) else np.abs(corr)
    else:
        corr = 0.0

    return s + corr


def refine_cofactor(x, c0):
    """
    Two-pass log grid search around initial guess.
    Returns refined cofactor c*.
    """
    # coarse search
    grid1 = c0 * np.logspace(-0.6, 0.6, 25)  # ~0.25x .. 4x
    scores1 = [stability_objective(x, c) for c in grid1]
    c = grid1[int(np.argmin(scores1))]

    # fine search
    grid2 = c * np.logspace(-0.3, 0.3, 21)
    scores2 = [stability_objective(x, c2) for c2 in grid2]
    return float(grid2[int(np.argmin(scores2))])


# ---------- logicle (stub) ----------
def estimate_logicle_cofactor(x):
    """
    Placeholder for logicle cofactor estimation.
    Strategy: optimize T, W, M params to balance:
        - dynamic range
        - compression of negative values
        - symmetry around zero
    """
    # TODO: implement full logicle optimizer
    return {
        "transform": "logicle",
        "T": None,
        "W": None,
        "M": None,
        "cofactor": None,
        "note": "Logicle estimation not implemented yet"
    }


# ---------- main ----------
def estimate_cofactors(comp_dir, out_path, max_files=MAX_FILES, method="arcsinh"):
    """
    Estimate cofactors across compensated CSV files.

    Parameters
    ----------
    comp_dir : str
        Directory containing compensated CSVs.
    out_path : str
        Path to save cofactor matrix CSV.
    max_files : int, optional
        Number of files to use (default=MAX_FILES).
    method : str
        "arcsinh" (default) or "logicle".
    """
    files = [os.path.join(comp_dir, f) for f in sorted(os.listdir(comp_dir)) if f.endswith(".csv")]
    files = files[:max_files]
    if not files:
        raise RuntimeError("No compensated CSVs found.")

    per_channel = {}
    for path in files:
        df = pd.read_csv(path)
        cols = pick_signal_columns(df)
        for c in cols:
            x = subsample(df[c].values, SUBSAMPLE_PER_FILE)
            if x.size:
                per_channel.setdefault(c, []).append(x)

    rows = []
    for ch, arrays in per_channel.items():
        x = np.concatenate(arrays)
        x = x[np.isfinite(x)]
        if x.size < MIN_EVENTS_PER_CH:
            continue

        if method == "arcsinh":
            c0 = initial_guess_arcsinh(x)
            c_star = refine_cofactor(x, c0)
            y = np.arcsinh(x / c_star)
            rows.append({
                "channel": ch,
                "transform": "arcsinh",
                "cofactor": round(c_star, 4),
                "n_used": int(x.size),
                "p99_5_abs": float(np.percentile(np.abs(x), 99.5)),
                "neg_frac": float(np.mean(x <= 0)),
                "skew_after": float(skew(y, bias=False)),
            })
        elif method == "logicle":
            rows.append({
                "channel": ch,
                **estimate_logicle_cofactor(x)
            })
        else:
            raise ValueError(f"Unknown method: {method}")

    if not rows:
        raise RuntimeError("No eligible channels for cofactor estimation.")

    df_out = pd.DataFrame(rows).sort_values("channel")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"✅ Cofactor matrix saved: {out_path}")
    print(df_out.head(12))

    return df_out


# Example usage (uncomment in script runs):
# estimate_cofactors("/content/compensated_output", "/content/data/cofactor_matrix.csv", method="arcsinh")
# estimate_cofactors("/content/compensated_output", "/content/data/cofactor_matrix_logicle.csv", method="logicle")
