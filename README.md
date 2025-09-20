# FlowCortex
Automated flow cytometry preprocessing, gating and visualization toolkit
# FlowCortex
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-in%20development-orange)]()

Automated **flow cytometry analysis engine** with reproducible preprocessing, compensation, cofactor estimation, and machine learning–ready gating modules.  
## Features
- **Data Preprocessing**
  - Extract metadata (core + exploratory) from FCS files  
  - Apply compensation using spillover matrices  
  - Estimate per-channel cofactors with skewness & variance-stabilization  
- **Automated Gating**
  - Live/Dead classification with ensemble thresholds (GMM + KDE + IQR)  
  - Singlet/Doublet discrimination via shape ratios  
  - Cell/Debris separation based on FSC/SSC density  
- **Visualization**
  - Density & contour plots (1D, 2D)  
  - Marker expression comparisons (X vs Y plots)  
  - Overlay of cluster labels and uncertainty regions  
- **Modular Architecture**
  - Ready for integration with ML pipelines  
  - Clear logging of QC decisions (JSON/YAML)  
  - Deployment-friendly structure  

## Project Structure
FlowCortex/
│
├── flowcortex/
│ ├── data_preproc/ # Metadata extraction, compensation, cofactor
│ ├── gating/ # Live/Dead, singlet/doublet, debris filtering
│ ├── visualization/ # KDE, contour plots, scatter overlays
│ ├── init.py
│
├── data/ # Example FCS/CSV data (gitignored if large)
├── tests/ # Unit tests for modules
├── notebooks/ # Jupyter/Colab exploration notebooks
├── README.md # Project overview (this file)
├── LICENSE # License file (MIT by default)
└── requirements.txt # Dependencies

from flowcortex.data_preproc import metadata, compensation, cofactor
from flowcortex.gating import live_dead, singlet_doublet
from flowcortex.visualization import plots

# 1. Extract metadata from FCS
metadata.collect_metadata("/path/to/fcs_folder")

# 2. Apply compensation
compensation.apply_spillover("/path/to/metadata.csv")

# 3. Estimate cofactors
cofactor.estimate("/path/to/compensated_csvs")

# 4. Run gating
live_dead_labels = live_dead.classify("/path/to/cofactor_applied.csv")

# 5. Visualize
plots.kde_contour(live_dead_labels)

Roadmap

 Add longitudinal batch effect modeling

 Bayesian posterior scoring for uncertain gates

 Integration with FlowSOM/UMAP embeddings

 Web-based UI for interactive gating

Contributing: Pull requests are welcome! Please open an issue first to discuss what you’d like to change.

License
This project is licensed under the MIT License

Author
Srinidhi