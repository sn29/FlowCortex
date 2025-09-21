import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ---------------------------
# Helper: KDE cutoff detector
# ---------------------------
def kde_cutoff(signal, depth_factor=0.1):
    """Find valley cutoff between first two density peaks using KDE."""
    signal = signal[np.isfinite(signal)]
    signal = signal[signal > 0]
    if len(signal) < 50:
        return None

    kde = gaussian_kde(signal)
    x_vals = np.linspace(signal.min(), signal.max(), 1000)
    y_vals = kde(x_vals)

    peaks, props = find_peaks(y_vals, prominence=0.01)
    if len(peaks) < 2:
        return None

    # Pick top 2 peaks by prominence
    top2 = np.argsort(props["prominences"])[-2:]
    left, right = sorted(peaks[top2])

    valley_idx = np.argmin(y_vals[left:right])
    valley_x = x_vals[left + valley_idx]
    valley_y = y_vals[left + valley_idx]

    depth = min(y_vals[left], y_vals[right]) - valley_y
    if depth < depth_factor * min(y_vals[left], y_vals[right]):
        return None

    return valley_x


# ---------------------------
# Main debris filtering
# ---------------------------
def filter_debris(input_path, output_path,
                  fsc_col="FSC.A", ssc_col="SSC.A",
                  debris_percentile=0.01, plot=True):
    """
    Label debris vs cells using hybrid (KDE + percentile) cutoffs.
    Saves labeled CSVs + threshold log.
    """
    os.makedirs(output_path, exist_ok=True)
    log_records, output_files = [], []

    for filename in os.listdir(input_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(input_path, filename)
        try:
            df = pd.read_csv(file_path)

            required = {fsc_col, ssc_col, "State_Label", "Doublet_Label"}
            if not required.issubset(df.columns):
                print(f"⚠ Skipping {filename}: Missing {required - set(df.columns)}")
                continue

            # Lowercase labels
            df["State_Label"] = df["State_Label"].astype(str).str.lower()
            df["Doublet_Label"] = df["Doublet_Label"].astype(str).str.lower()

            # Cutoffs
            fsc_kde = kde_cutoff(df[fsc_col])
            ssc_kde = kde_cutoff(df[ssc_col])
            fsc_pct = df[fsc_col].quantile(debris_percentile)
            ssc_pct = df[ssc_col].quantile(debris_percentile)

            fsc_final = fsc_kde if fsc_kde and fsc_kde < df[fsc_col].quantile(0.1) else fsc_pct
            ssc_final = ssc_kde if ssc_kde and ssc_kde < df[ssc_col].quantile(0.1) else ssc_pct

            # Assign debris label
            df["Debris_Label"] = np.where(
                (df[fsc_col] <= fsc_final) |
                (df[ssc_col] <= ssc_final) |
                (df["State_Label"] == "dead") |
                (df["Doublet_Label"] == "doublet"),
                "debris", "cell"
            )

            # Save
            out_file = os.path.join(output_path, f"debris_{filename}")
            df.to_csv(out_file, index=False)
            output_files.append(out_file)

            # Log
            log_records.append({
                "filename": filename,
                "fsc_cutoff": fsc_final,
                "ssc_cutoff": ssc_final,
                "fsc_method": "kde" if fsc_kde else "percentile",
                "ssc_method": "kde" if ssc_kde else "percentile"
            })

            # Optional plots
            if plot:
                plt.figure(figsize=(12, 5))

                # 1. All cells
                plt.subplot(1, 2, 1)
                colors = df["Debris_Label"].map({"debris": "red", "cell": "gray"})
                plt.scatter(df[fsc_col], df[ssc_col], s=2, alpha=0.3, c=colors)
                plt.title(f"Debris vs Cell: {filename}")
                plt.xlabel(fsc_col)
                plt.ylabel(ssc_col)
                plt.grid(True)

                # 2. Clean gated
                gated = df[
                    (df["Debris_Label"] == "cell") &
                    (df["State_Label"] == "live") &
                    (df["Doublet_Label"] == "singlet")
                ]
                plt.subplot(1, 2, 2)
                plt.scatter(gated[fsc_col], gated[ssc_col], s=2, alpha=0.3, color="green")
                plt.title(f"Clean Gated: {filename}")
                plt.xlabel(fsc_col)
                plt.ylabel(ssc_col)
                plt.grid(True)

                plt.tight_layout()
                plt.show()

            print(f"✅ Processed {filename} | FSC cut: {fsc_final:.2f} | SSC cut: {ssc_final:.2f}")

        except Exception as e:
            print(f"❌ Error in {filename}: {e}")

    # Save threshold log
    if log_records:
        pd.DataFrame(log_records).to_csv(os.path.join(output_path, "Debris_Threshold_Log.csv"), index=False)

    return output_files


# ---------------------------
# CLI example
# ---------------------------
if __name__ == "__main__":
    debris_filtered = filter_debris(
        input_path="/content/data/FlowSense_Testing/Step3_SingletDoublet",
        output_path="/content/data/FlowSense_Testing/Step4_DebrisFiltered",
        plot=True
    )
    print(f"\n📘 Debris filtering complete. Files: {len(debris_filtered)}")
