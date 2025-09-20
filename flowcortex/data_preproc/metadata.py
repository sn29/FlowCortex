"""
Metadata extraction module for FlowCortex.
Extracts core and exploratory metadata from FCS files and saves to CSV.
"""

import os
import pandas as pd
from fcsparser import parse as fcs_parse


def extract_core_metadata(meta, fcs_path):
    """
    Extract stable/core metadata from a single FCS file.
    """
    out = {
        "filename": os.path.basename(fcs_path),
        "filepath": fcs_path,
        "date": meta.get("$DATE", "NA"),
        "total_events": int(meta.get("$TOT", -1)),
        "cytometer": meta.get("$CYT", "NA"),
        "instrument": meta.get("$INST", "NA"),
        "byte_order": meta.get("$BYTEORD", "NA"),
        "n_parameters": int(meta.get("$PAR", 0)),
        "source": meta.get("$SRC", "NA"),
    }
    return out


def extract_full_metadata(meta):
    """
    Extract exploratory metadata (all available keys).
    """
    return {k: v for k, v in meta.items()}


def collect_metadata_from_folder(
    fcs_dir,
    core_out="fcs_metadata_core.csv",
    full_out="fcs_metadata_full.csv"
):
    """
    Parse all FCS files in a folder and save both core + full metadata.
    """
    core_list, full_list = [], []

    for file in sorted(os.listdir(fcs_dir)):
        if not file.endswith(".fcs"):
            continue
        fcs_path = os.path.join(fcs_dir, file)

        try:
            meta, _ = fcs_parse(fcs_path)
            core_list.append(extract_core_metadata(meta, fcs_path))
            full = extract_full_metadata(meta)
            full["filename"] = file
            full_list.append(full)
            print(f"✅ Parsed: {file}")

        except Exception as e:
            print(f"❌ Error parsing {file}: {e}")

    if core_list:
        pd.DataFrame(core_list).to_csv(core_out, index=False)
        print(f"Core metadata saved to: {core_out}")

    if full_list:
        pd.DataFrame(full_list).to_csv(full_out, index=False)
        print(f"Full metadata saved to: {full_out}")
