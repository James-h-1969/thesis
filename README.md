<div align="center">

# 🌊 SeaSeer

**Neural ODE-based spatiotemporal forecasting for ocean & climate data**

[![CI](https://github.com/James-h-1969/seaseer/actions/workflows/ci.yml/badge.svg)](https://github.com/James-h-1969/seaseer/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

---

## Overview

SeaSeer is a deep learning model for forecasting ocean state a day ahead, built as an honours thesis on predicting marine heat waves in the Great Barrier Reef region.

The idea is to learn the **transport dynamics** of the ocean rather than the state directly: a convolutional network predicts a velocity field from the current state, and that field is integrated forward in time as a Neural ODE to produce the next state.

## How it works

**Inputs.** Daily fields on a 0.25° grid over 10°S–25°S, 142°E–154°E from 1993–2018:

| Source | Variables |
|--------|-----------|
| ERA5 (Copernicus CDS) | Shortwave and longwave radiation, latent and sensible heat flux |
| NOAA OISST | Sea surface temperature |
| CMEMS (Copernicus Marine) | Surface currents, mixed-layer depth, bottom temperature, vertical velocity |

`dataloader.py` aligns every source to the common grid and daily timestep and serves `(state at t, state at t+1)` pairs.

**Model** (`model.py`). A ResNet velocity network predicts the flow field from the input channels. Two optional branches can be switched on:

- an attention velocity branch, blended in with a learned weight;
- an emission head that outputs a mean and standard deviation so predictions carry an uncertainty estimate.

**Training** (`train.py`). AdamW with an MSE loss on the next-day state, saving a checkpoint to `model_checkpoints/`.

Status: the velocity network and data pipeline are in place; ODE integration in the forward pass and the evaluation data loader are still to be implemented.

## Getting Started

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (package manager)

### Installation

```bash
git clone git@github.com:James-h-1969/seaseer.git
cd seaseer
uv sync
pre-commit install
```

### Training

```bash
cd seaseer
make train
```

### Evaluation

```bash
cd seaseer
make eval
```

### Data Generation

Before downloading data, you need to set up credentials for the two data providers:

1. **Copernicus Climate Data Store (ERA5)**: Register at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu), then create `~/.cdsapirc` with:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-api-key>
   ```

2. **Copernicus Marine (CMEMS)**: Register at [https://data.marine.copernicus.eu/register](https://data.marine.copernicus.eu/register), then log in once:
   ```bash
   uv run copernicusmarine login
   ```

Then generate the data:
```bash
make generate_data
```
This downloads data covering the Great Barrier Reef region (10°S–25°S, 142°E–154°E) from 1993–2018.

## Project Structure

```
seaseer/
├── model.py              # SeaSeer model definition
├── train.py              # Training loop
├── eval.py               # Evaluation script
└── README.md             # SeaSeer-specific docs
helpers/
├── models/
│   ├── ResidualBlock.py  # Residual block module
│   └── ResidualNetwork.py# ResNet backbone
└── scripts/
    └── generate_data.py  # Data download & generation
tests/
├── conftest.py           # Shared test fixtures
└── test_models.py        # Test suite
Makefile                  # Train/eval/data shortcuts
```

## Thesis
This repository holds the code for James Hocking's honours thesis, *SeaSeer: A Neural ODE for Predicting Marine Heat Waves*, at the University of Sydney.
