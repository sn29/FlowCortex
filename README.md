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
  - Marker expression comparisons UMAP  
  - Overlay of cluster labels and uncertainty regions  
- **Modular Architecture**
  - Ready for integration with ML pipelines  
  - Clear logging of QC decisions (JSON/YAML)  
  - Deployment-friendly structure  

Roadmap
Add longitudinal batch effect modeling
Bayesian posterior scoring for uncertain gates
Integration with FlowSOM/UMAP embeddings
Web-based UI for interactive gating

Contributing
Pull requests are welcome! Please open an issue first to discuss what you’d like to change.

License
This project is licensed under the MIT License

Author
Srinidhi
