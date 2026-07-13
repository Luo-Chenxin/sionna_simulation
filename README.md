# Sionna Simulation ToolKIT

A toolkit for generating radio map datasets by combining [Sionna](https://github.com/NVlabs/sionna) ray-tracing simulations with real-world transmitter data and OpenStreetMap geographic information.

## Overview

This project provides tools to:
1. Simulate radio propogation in the Massy (91300, France) area using Sionna's ray-tracing capabilities
2. Convert simulation results of a specific into structured datasets suitable for deep learning applications. For detailed methodology and technical specifications, refer to `Internship_Report.pdf`.

## Prerequisites

### Software Dependencies
- Python 3.8+
- Sionna
- TensorFlow
- Jupyter Notebook
- Blender (for map file export)
- Required Python packages:
  - numpy
  - pandas
  - geopandas
  - osmnx
  - matplotlib
  - shapely
  - h5py
  - ...

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