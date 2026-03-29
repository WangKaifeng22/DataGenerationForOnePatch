# DataGenerationForOnePatch

## Description

**DataGenerationForOnePatch** is a project designed for generating high-fidelity ultrasound and acoustic simulation data. It combines the generation of Speed of Sound (SoS) maps based on **Gaussian Random Fields (GRF)** with the open-source acoustic toolbox **k-Wave python** to simulate complex acoustic wave propagation and produce realistic ultrasound datasets.

**Core Workflow:**

- `main.py` serves as the primary entry point that orchestrates the entire data generation pipeline.
- `GenerateSoSMaps.py` is invoked to generate the physical parameter maps (GRF-based SoS maps) that represent the simulated tissue environment.
- `Kwave.py` takes the generated physical parameters and runs the acoustic simulations using the k-Wave python toolbox to produce the final ultrasound data.

## File Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point that orchestrates the full data generation workflow. |
| `Kwave.py` | Core simulation script that runs acoustic simulations using the k-Wave python wrapper. |
| `GenerateSoSMaps.py` | Generates the GRF-based Speed of Sound (SoS) maps representing simulated tissue media. |
| `config.py` | Centralized configuration file for simulation parameters, grid dimensions, and output paths. |
| `grid_coords.py` | Calculates and manages the spatial grid coordinates for the simulation environment. |
| `transducer_mask.py` | Defines the transducer array geometry and creates sensor masks for data recording. |
| `mat_to_npy.py` | Utility script to convert MATLAB `.mat` files into NumPy `.npy` format. |
| `GRF_KL.py` | Implements Gaussian Random Field generation using the Karhunen-Loève (KL) expansion. |

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WangKaifeng22/DataGenerationForOnePatch.git
   cd DataGenerationForOnePatch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install k-Wave Python:**

   > **Note:** The k-Wave python package is required for acoustic simulations and must be installed separately.

   ```bash
   pip install k-wave-python
   ```

## Usage

1. **Configure parameters:** Open `config.py` and adjust the simulation settings (e.g., grid size, simulation duration, output directories) to match your requirements.

2. **Run the simulation:**
   ```bash
   python main.py
   ```

   This will automatically generate the SoS maps, execute the k-Wave simulation, and save the resulting data patches to the configured output directory.

## Data Processing

After generating the raw simulation data, the following utility scripts can be used for visualization and dataset management:

- **`plot_waterfall_kwaveresult.py`**: Visualizes the generated simulation results by producing waterfall plots of the sensor data, allowing you to inspect wave propagation and reflection patterns.
- **`merge_datasets.py`**: Combines multiple individually generated data patches into a single unified dataset, suitable for use in downstream machine learning or analysis workflows.
