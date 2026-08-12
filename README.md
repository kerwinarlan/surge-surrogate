<div align="center">

# 🌊 SurgeSurrogate

**A fast machine-learning surrogate for storm surge forecasting in Manila Bay**

[![Python](https://img.shields.io/badge/Python%203.9-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-222222?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)

</div>

SurgeSurrogate replaces computationally heavy hydrodynamic simulations (such as
Delft3D, ADCIRC, or FVCOM) with a trained gradient-boosted regression model. A
single forecast that would take minutes to hours inside a hydrodynamic solver
returns in under a millisecond, making it suitable for real-time operational
use, ensemble forecasting, and rapid risk assessment.

The model learns the physical relationship between tropical cyclone
characteristics and the storm surge residual observed at the Manila tide gauge
(GLOSS station 071, UHSLC id 370). It is trained on four historical typhoons
that affected Manila Bay: **Ketsana (2009), Nesat (2011), Rammasun (2014), and
Vamco (2020)**.

---

## Table of Contents

- [Why it exists: surge forecasts are too slow](#why-it-exists-surge-forecasts-are-too-slow)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Local Installation & Setup](#local-installation--setup)
- [Training the Model](#training-the-model)
- [API Usage](#api-usage)
- [Model Performance](#model-performance)
- [Data Sources & Units](#data-sources--units)
- [License](#license)

---

## Why it exists: surge forecasts are too slow

Storm surge is one of the deadliest hazards of tropical cyclones. In 2009,
Typhoon Ketsana (Ondoy) inundated Metro Manila with flooding that displaced
nearly half a million people; in 2013, Typhoon Haiyan demonstrated the
catastrophic potential of surge-driven coastal flooding across the
Philippines. Reliable surge forecasts require simulating the coupled dynamics
of wind, atmospheric pressure, bathymetry, and tides.

Traditional hydrodynamic models such as Delft3D solve the shallow-water
equations on a high-resolution grid. They are physically comprehensive but
computationally expensive. Each storm scenario requires hours of simulation,
which limits how many ensemble members or forecast updates can be produced in
an operational window.

SurgeSurrogate takes a different approach. It is a **physics-informed ML
surrogate**: instead of simulating the ocean, it learns the mapping from a
handful of physically meaningful storm characteristics to the surge signal at
a specific location (Manila Bay). Because the model is a single decision-tree
ensemble, inference cost is negligible. This enables:

- **Real-time nowcasting** during an active typhoon approach.
- **Ensemble surge forecasts** from many track perturbations in seconds.
- **Fast screening** of many hypothetical storms for planning and drill scenarios.

The surrogate does not replace the physics. It replaces the *computation*,
with the physics encoded through carefully engineered features (see
[Key Features](#key-features)).

| Problem | Solution | Result |
|---|---|---|
| Hydrodynamic sims take minutes to hours per storm | Gradient-boosted surrogate trained on four historical typhoons | Surge forecast in under a millisecond |
| Astronomical tides mask the storm-driven signal | 25-hour centered rolling mean isolates the surge residual | The model predicts the physically meaningful quantity |
| Remote data sources can fail mid-pipeline | Resumable IBTrACS downloader + bundled tide fallback | Training never hard-fails on network conditions |

## Key Features

**Spatial feature engineering.** Each storm track point is described relative
to Manila Bay, not in absolute coordinates:

| Feature | Definition | Physical meaning |
| --- | --- | --- |
| `wind_kts` | Maximum sustained wind speed (USA_WIND, knots) | Storm intensity |
| `pressure_deficit_hpa` | `1013.25 - central_pressure` (hPa) | Depth of the pressure anomaly, the primary surge driver |
| `distance_to_manila_km` | Great-circle (Haversine) distance from the storm center to Manila Bay (14.58°N, 120.97°E) | Storm proximity to the target coastline |
| `approach_angle_deg` | Forward azimuth from the storm center toward Manila Bay (0–360°) | Storm approach geometry relative to the bay |

This formulation encodes the physics of surge generation: wind stress and the
inverse-barometer effect drive the water level, while distance and approach
angle control how much of that energy reaches Manila Bay and whether the bay is
on the dangerous right-front quadrant of the storm.

**Surge residual target.** Total water level is dominated by the astronomical
tide, which would mask the storm-driven signal. The pipeline isolates the
meteorological surge by subtracting a **25-hour centered rolling mean** - a
standard estimate of the astronomical tide that removes the diurnal and
semidiurnal tidal constituents without phase lag. The model therefore predicts
the **surge residual** (meters), the physically meaningful quantity for
disaster response.

**Robust multi-source data acquisition.** Storm tracks are pulled from the
NOAA IBTrACS Western Pacific archive with a resumable downloader. Hourly tide
observations for Manila are fetched from the UHSLC open ERDDAP API. If the
remote tide service is unavailable, the pipeline transparently falls back to
the bundled `manila_tide_2014.csv` record with Typhoon Rammasun, so training
never hard-fails on network conditions.

**FastAPI backend.** A small, production-ready HTTP service loads the trained
model once and serves surge predictions through a typed Pydantic schema, with
interactive documentation at `/docs` and measured inference time per request.

## How It Works

```
                 ┌──────────────────────────────────────────┐
                 │           NOAA IBTrACS (WP basin)        │
                 │  Ketsana 2009 · Nesat 2011               │
                 │  Rammasun 2014 · Vamco 2020              │
                 └─────────────────────┬────────────────────┘
                                       │ 3-hourly track points
                                       ▼
                 ┌──────────────────────────────────────────┐
                 │        UHSLC Manila tide gauge (071)     │
                 │  hourly sea level per storm window       │
                 │  (fallback: manila_tide_2014.csv)        │
                 └─────────────────────┬────────────────────┘
                                       │
                                       ▼
        ┌───────────────────────────────────────────────────────────┐
        │  Target:  surge_residual = water_level − 25h rolling mean │
        │  Features: wind_kts, pressure_deficit_hpa,                │
        │            distance_to_manila_km, approach_angle_deg      │
        └──────────────────────────────┬────────────────────────────┘
                                       │
                                       ▼
                 ┌──────────────────────────────────────────┐
                 │  XGBoostRegressor → surge_model.pkl      │
                 └─────────────────────┬────────────────────┘
                                       │ joblib.load()
                                       ▼
                 ┌──────────────────────────────────────────┐
                 │  FastAPI  POST /predict  → surge residual │
                 └──────────────────────────────────────────┘
```

The pipeline (`pipeline.py`) downloads IBTrACS data, fetches tide observations,
interpolates each 3-hourly track onto the tide timestamps, engineers the
features and target, trains an `XGBoostRegressor`, and persists the model. The
API (`api/main.py`) loads the artifact and serves predictions.

## Repository Structure

```
surge-surrogate/
├── api/
│   └── main.py                 # FastAPI application (model serving)
├── data/
│   ├── ibtracs_wp_v04r01.csv   # Cached IBTrACS Western Pacific archive
│   ├── glenda_track_2014.csv   # Extracted Rammasun (Glenda) track
│   └── surge_dataset.csv       # Engineered training set (features + target)
├── models/
│   └── surge_model.pkl         # Trained XGBoost surrogate model
├── notebooks/                  # Scratch / exploration notebooks
├── manila_tide_2014.csv        # Fallback tide record (Rammasun 2014)
├── fetch_glenda.py             # Legacy single-storm track fetcher
├── pipeline.py                 # End-to-end training pipeline
├── requirements.txt            # Python dependencies
└── README.md
```

## Local Installation & Setup

### Prerequisites

- Python 3.9+ (the included virtual environment uses 3.9.6)
- `pip` and `venv` (bundled with Python)
- An internet connection for the first IBTrACS download (~114 MB, resumable)
  and UHSLC tide queries

### macOS: OpenMP runtime for XGBoost

XGBoost requires the OpenMP runtime on macOS. Install it with Homebrew first:

```bash
brew install libomp
```

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

Run the full pipeline from the repository root:

```bash
python pipeline.py
```

The pipeline will:

1. Download the IBTrACS Western Pacific archive (cached after the first run).
2. Extract tracks for Ketsana (2009), Nesat (2011), Rammasun (2014), and
   Vamco (2020).
3. Fetch hourly Manila sea level from UHSLC for each storm window
   (fallback: `manila_tide_2014.csv` + Rammasun).
4. Compute the surge residual and spatial features.
5. Train and evaluate the XGBoost surrogate.
6. Save the artifact to `models/surge_model.pkl` and the dataset to
   `data/surge_dataset.csv`.

Example output:

```
KETSANA (2009): 47 points, 2009-09-25 00:00:00 to 2009-09-30 18:00:00.
...
UHSLC Manila: 409 hourly points, 2014-07-06 06:00:00 to 2014-07-23 06:00:00.
...
Combined dataset: 773 rows across 4 storm(s).
Test MAE: 0.1344 m | Test R2: 0.7581
Model saved to models/surge_model.pkl
```

## API Usage

### Start the server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API is then available at `http://localhost:8000`, with interactive
documentation at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service health / status message |
| `POST` | `/predict` | Predict storm surge residual from storm features |

### Request schema

`POST /predict` accepts a JSON body with four floating-point fields:

| Field | Type | Units | Description |
| --- | --- | --- | --- |
| `wind_kts` | float | knots | Maximum sustained wind speed |
| `pressure_deficit_hpa` | float | hPa | `1013.25 − central_pressure` |
| `distance_to_manila_km` | float | km | Great-circle distance from storm center to Manila Bay |
| `approach_angle_deg` | float | degrees | Forward azimuth from storm to Manila Bay (0–360) |

### Example: surge-peak scenario

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "wind_kts": 85,
    "pressure_deficit_hpa": 53,
    "distance_to_manila_km": 127,
    "approach_angle_deg": 96
  }'
```

Expected JSON response:

```json
{
  "status": "success",
  "predicted_surge_residual_meters": 0.74,
  "inference_time_ms": 0.31
}
```

### Example: calm background conditions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "wind_kts": 25,
    "pressure_deficit_hpa": 5,
    "distance_to_manila_km": 3000,
    "approach_angle_deg": 200
  }'
```

```json
{
  "status": "success",
  "predicted_surge_residual_meters": 0.08,
  "inference_time_ms": 0.28
}
```

> Note: `predicted_surge_residual_meters` is the storm-driven surge above the
> astronomical tide. To obtain total water level, add the astronomical tide
> forecast at the time of interest.

## Model Performance

Evaluated on a 20% held-out split (shuffled, fixed seed 42) of 773
storm-tide-aligned rows:

| Metric | Value |
| --- | --- |
| Test MAE | 0.134 m |
| Test R² | 0.758 |

The model generalizes across four distinct storm events and reproduces the
observed surge-peak behavior: the largest residuals occur just after the
storm's closest approach to Manila Bay, when the bay sits in the storm's
dangerous semicircle.

## Data Sources & Units

| Source | Use | Units converted to |
| --- | --- | --- |
| [NOAA IBTrACS v04r01](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/) | Storm tracks (Ketsana, Nesat, Rammasun, Vamco) | wind in knots, pressure in hPa |
| [UHSLC ERDDAP](https://uhslc.soest.hawaii.edu/erddap/index.html) `global_hourly_rqds` | Manila hourly sea level (station GLOSS 071, id 370) | millimeters → meters |
| `manila_tide_2014.csv` | Local fallback tide record (Rammasun 2014) | meters |

## License

Data from NOAA IBTrACS and UHSLC are redistributed under their respective
terms. See the source repositories for full licensing details.
