import os
import pandas as pd
import numpy as np

def load_spill_matrix(meta_csv):
    """
    Extract spillover/compensation matrix from a metadata CSV.
    Looks for columns containing 'SPILL' or 'COMP'.
    """
    meta = pd.read_csv(meta_csv)

    spill_keys = [c for c in meta.columns if "SPILL" in c.upper() or "COMP" in c.upper()]
    if not spill_keys:
        raise ValueError("❌ No spillover/comp keys found in metadata file.")

    # Use first spill key
    spill_str = str(meta[spill_keys[0]].dropna().values[0])
    try:
        values = [float(x) for x in spill_str.replace(";", ",").split(",")]
    except Exception:
        raise ValueError("❌ Could not parse spillover matrix string.")

    # First entry is #channels, then matrix follows
    n = int(values[0])
    mat = np.array(values[1:]).reshape(n + 1, n)  # (channel names + matrix)

    channels = mat[0, :]
    spill_matrix = mat[1:, :].astype(float)

    spill_df = pd.DataFrame(spill_matrix, columns=channels, index=channels)
    return spill_df


def apply_compensation(csv_in, spill_matrix, csv_out):
    """
    Apply spillover compensation to a CSV file containing FCS events.
    Adds CellID column if missing.
    """
    df = pd.read_csv(csv_in)

    # Add CellID if not already present
    if "CellID" not in df.columns:
        df.insert(0, "CellID", [f"C{i+1}" for i in range(len(df))])

    comp_cols = [c for c in df.columns if c in spill_matrix.columns]

    if not comp_cols:
        raise ValueError("❌ No matching fluorescence channels found in CSV.")

    X = df[comp_cols].to_numpy()
    Xc = np.linalg.solve(spill_matrix.values.T, X.T).T  # Apply compensation

    for i, col in enumerate(comp_cols):
        df[f"{col}_comp"] = Xc[:, i]

    df.to_csv(csv_out, index=False)
    print(f"✅ Compensation applied and saved: {csv_out}")
    return df
