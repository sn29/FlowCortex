# singlet_doublet.py
"""
Singlet/Doublet classification module.
- Uses FSC-A vs FSC-H shape ratio + viability density.
- Labels cells as singlet, doublet, or uncertain.
- Saves per-file CSV + scatter plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ---------- Core classifier ----------

def classify_singlet_doublet(df, fsc_a_col="FSC.A", fsc_h_col="FSC.H", fvs_col="FVS450.A"):
    # Work only on live cells
    df = df[df["State_Label"] == "live"].copy()

    # Ratios
    df["Shape_Ratio"] = df[fsc_a_col] / df[fsc_h_col]
    df["Viability_Density"] = df[fvs_col] / df[fsc_a_col]

    # GMM on Shape Ratio
    shape_ratio = df["Shape_Ratio"].dropna().values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(shape_ratio)
    means = np.sort(gmm.means_.flatten())
    shape_thresh = np.mean(means)

    # Label assignment
    def assign_label(row):
        if row["Shape_Ratio"] < shape_thresh and row["Viability_Density"] < 1.0:
            return "singlet"
        elif row["Shape_Ratio"] >= shape_thresh or row["Viability_Density"] >= 1.5:
            return "doublet"
        else:
            return "uncertain"

    df["Doublet_Label"] = df.apply(assign_label, axis=1)
    return df, shape_thresh

# ---------- Batch processor ----------

def process_doublet_batch(input_path, output_path,
                          fsc_a_col="FSC.A", fsc_h_col="FSC.H", fvs_col="FVS450.A"):
    os.makedirs(output_path, exist_ok=True)
    processed_files = []
    thresholds = {}

    for filename in sorted(os.listdir(input_path)):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(input_path, filename)
        try:
            df = pd.read_csv(file_path)
            required_cols = {"State_Label", fsc_a_col, fsc_h_col, fvs_col}
            if not required_cols.issubset(df.columns):
                print(f"⚠️ Skipping {filename}: Missing required columns.")
                continue

            # Classify
            df_out, shape_thresh = classify_singlet_doublet(df, fsc_a_col, fsc_h_col, fvs_col)

            # Save CSV
            out_file = os.path.join(output_path, f"doublet_{filename}")
            df_out.to_csv(out_file, index=False)
            processed_files.append(out_file)
            thresholds[filename] = shape_thresh

            # Save plot
            plt.figure(figsize=(6, 5))
            singlets = df_out[df_out["Doublet_Label"] == "singlet"]
            doublets = df_out[df_out["Doublet_Label"] == "doublet"]

            plt.scatter(singlets[fsc_a_col], singlets[fsc_h_col], s=2, alpha=0.3,
                        color="green", label="Singlet")
            plt.scatter(doublets[fsc_a_col], doublets[fsc_h_col], s=2, alpha=0.3,
                        color="red", label="Doublet")

            plt.xlabel(fsc_a_col)
            plt.ylabel(fsc_h_col)
            plt.title(f"Singlet vs Doublet: {filename}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_file = out_file.replace(".csv", "_plot.png")
            plt.savefig(plot_file)
            plt.close()

            print(f"✔ Processed {filename} | Shape Threshold: {shape_thresh:.2f}")

        except Exception as e:
            print(f"❌ Error in {filename}: {e}")

    return processed_files, thresholds

# Example run
if __name__ == "__main__":
    processed, thr = process_doublet_batch(
        input_path="/content/data/FlowSense_Testing/Step3_Arcsinh_LiveDead",
        output_path="/content/data/FlowSense_Testing/Step4_SingletDoublet"
    )
    print("Processed files:", processed)
    print("Thresholds:", thr)
