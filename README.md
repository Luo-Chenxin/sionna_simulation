# Sionna Simulation ToolKIT

A toolkit for generating radio map datasets by combining [Sionna](https://github.com/NVlabs/sionna) ray-tracing simulations with real-world transmitter data and OpenStreetMap geographic information.

## Overview

This project provides tools to:
1. Simulate radio propogation in the Massy (91300, France) area using Sionna's ray-tracing capabilities
2. Convert simulation results of a specific into structured datasets suitable for deep learning applications. For detailed methodology and technical specifications, refer to `Internship_Report.pdf`.

## Prerequisites

### Installation Recommendations
Due to some dependency conflicts between GeoLibrary and Sionna, the following installation order is important.
```
conda create -n sionna-clean python=3.10 -y
conda activate sionna-clean
pip install sionna-rt
pip install h5py rasterio
pip install osmnx
pip install mapbox_earcut
pip install ipykernel
...
```

### Map File
Download the Massy area map file:
- [Massy Map File](https://drive.google.com/file/d/1TAxad9_nXrWH8VMxGgW3J0sS1thS7XWr/view?usp=sharing)

#### Blender Export Settings
When exporting the map from Blender, ensure the following options are selected:
-  Export IDs
-  Ignore Default Background
-  Y Forward & Z Up

## Data Sources

### ANFR Transmitter Data
- **Source:** The National Frequency Agency (ANFR) of France
- **Content:** Real-world transmitter locations, frequencies, powers, and antenna characteristics

### OpenStreetMap
- Geographic information extracted for the target area

## References

- [Sionna Documentation](https://nvlabs.github.io/sionna/)
- [ANFR Website](https://www.anfr.fr/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- Internship Report: `Internship_Report.pdf` (included in repository)

## Contact

For questions or collaboration inquiries, please refer to the authors of `Internship_Report.pdf`.