"""SurgeSurrogate training pipeline (physical engineering constraints).

1. Downloads IBTrACS Western Pacific track data (resumable).
2. Extracts tracks for Ketsana (2009), Nesat (2011), Rammasun (2014), Vamco (2020).
3. Fetches hourly Manila tide data (UHSLC station GLOSS 071, uhslc_id 370).
   Falls back to manila_tide_2014.csv + Rammasun if UHSLC is unavailable.
4. Isolates the surge residual: observed water level minus a 25-hour
   centered rolling mean (the astronomical tide estimate).
5. Builds spatial features per track point:
   - wind_kts (USA_WIND, already knots)
   - pressure_deficit_hpa (1013.25 - central pressure)
   - distance_to_manila_km (Haversine to Manila Bay 14.58N 120.97E)
   - approach_angle_deg (forward bearing from storm to Manila Bay)
6. Trains an XGBoostRegressor on those features -> surge_residual.
7. Saves the model to models/surge_model.pkl.
"""

import os
import sys
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)
RAW_CSV = os.path.join(ROOT, "data", "ibtracs_wp_v04r01.csv")
TIDE_FALLBACK_CSV = os.path.join(ROOT, "manila_tide_2014.csv")
MODEL_PATH = os.path.join(ROOT, "models", "surge_model.pkl")
DATASET_CSV = os.path.join(ROOT, "data", "surge_dataset.csv")

UHSLC_TABLEDAP = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_rqds.csv"
UHSLC_STATION_ID = 370  # Manila, GLOSS station 071

MANILA_LAT, MANILA_LON = 14.58, 120.97
REFERENCE_PRESSURE_HPA = 1013.25
STORM_MARGIN = pd.Timedelta(days=3)

STORMS = {"KETSANA": 2009, "NESAT": 2011, "RAMMASUN": 2014, "VAMCO": 2020}
TRACK_FEATURES = ["LAT", "LON", "USA_WIND", "USA_PRES"]
MODEL_FEATURES = [
    "wind_kts",
    "pressure_deficit_hpa",
    "distance_to_manila_km",
    "approach_angle_deg",
]

# The full WP basin file is ~114 MB. Treat sizes above this as complete.
MIN_FULL_SIZE = 100 * 1024 * 1024
CHUNK = 64 * 1024


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def download_ibtracs(url: str, dest: str, attempts: int = 5) -> str:
    """Download url to dest with resumable Range requests."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    if existing >= MIN_FULL_SIZE:
        print(f"IBTrACS data already downloaded ({existing} bytes).")
        return dest

    for attempt in range(1, attempts + 1):
        try:
            headers = {"User-Agent": "surge-surrogate-pipeline/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            print(f"Downloading IBTrACS (attempt {attempt}, resuming at {existing} bytes)...")
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
            print(f"Download complete: {dest} ({os.path.getsize(dest)} bytes).")
            return dest
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"Download interrupted: {exc}. Retrying from {existing} bytes...")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download IBTrACS data after {attempts} attempts.")


def load_storm_tracks(raw_csv: str) -> dict[str, pd.DataFrame]:
    """Return one track DataFrame per storm, indexed by ISO_TIME."""
    print("Reading IBTrACS Western Pacific dataset...")
    # Row index 1 in IBTrACS contains unit headers, so skip it.
    df = pd.read_csv(raw_csv, skiprows=[1], low_memory=False)

    tracks: dict[str, pd.DataFrame] = {}
    for name, season in STORMS.items():
        sub = df[(df["NAME"].str.upper() == name) & (df["SEASON"].astype(str) == str(season))]
        if sub.empty:
            raise RuntimeError(f"No IBTrACS points found for {name} ({season}).")
        track = sub[["ISO_TIME"] + TRACK_FEATURES].copy()
        track["ISO_TIME"] = pd.to_datetime(track["ISO_TIME"], errors="coerce")
        # IBTrACS encodes missing values as blank strings; coerce to NaN.
        for col in TRACK_FEATURES:
            track[col] = pd.to_numeric(track[col], errors="coerce")
        track = track.dropna(subset=["ISO_TIME"]).sort_values("ISO_TIME")
        track = track.set_index("ISO_TIME")
        print(
            f"{name} ({season}): {len(track)} points, "
            f"{track.index.min()} to {track.index.max()}."
        )
        tracks[name] = track
    return tracks


def fetch_uhslc_tide(start: pd.Timestamp, end: pd.Timestamp, attempts: int = 3) -> pd.Series:
    """Fetch hourly Manila sea level (m) from the UHSLC ERDDAP API."""
    # Constraint operators (>=, <=) must stay literal; urlencode would escape them.
    url = (
        f"{UHSLC_TABLEDAP}?time,sea_level"
        f"&uhslc_id={UHSLC_STATION_ID}"
        f"&time>={start.isoformat()}Z&time<={end.isoformat()}Z"
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "surge-surrogate-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Row 1 of the ERDDAP CSV is the unit header row; skip it.
                tide = pd.read_csv(resp, skiprows=[1])
            if tide.empty:
                raise RuntimeError(f"UHSLC returned no rows for {start} to {end}.")
            tide["time"] = pd.to_datetime(tide["time"], utc=True).dt.tz_localize(None)
            tide = tide.set_index("time").sort_index()
            sea_level_m = tide["sea_level"].astype(float) / 1000.0  # mm -> m
            print(
                f"UHSLC Manila: {len(sea_level_m)} hourly points, "
                f"{sea_level_m.index.min()} to {sea_level_m.index.max()}."
            )
            return sea_level_m
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last_error = exc
            print(f"UHSLC fetch attempt {attempt} failed: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"UHSLC fetch failed after {attempts} attempts: {last_error}")


def load_fallback_tide() -> pd.Series:
    """Load manila_tide_2014.csv as a fallback sea level series (m)."""
    tide = pd.read_csv(TIDE_FALLBACK_CSV, parse_dates=["time"])
    tide = tide.set_index("time").sort_index()
    print(f"Fallback tide: {len(tide)} rows, {tide.index.min()} to {tide.index.max()}.")
    return tide["water_level"].astype(float)


# ---------------------------------------------------------------------------
# Target and feature engineering
# ---------------------------------------------------------------------------
def surge_residual(sea_level_m: pd.Series) -> pd.Series:
    """Subtract a 25-hour centered rolling mean (astronomical tide estimate)."""
    dt_s = sea_level_m.index.to_series().diff().dt.total_seconds().median()
    window_points = int(round(25 * 3600 / dt_s))
    min_periods = max(12, int(0.7 * window_points))
    astro_tide = sea_level_m.rolling("25h", center=True, min_periods=min_periods).mean()
    return (sea_level_m - astro_tide).rename("surge_residual")


def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance from each (lat, lon) to Manila Bay, in km."""
    r_earth = 6371.0
    phi1 = np.radians(lat)
    phi2 = np.radians(MANILA_LAT)
    dphi = np.radians(MANILA_LAT - lat)
    dlam = np.radians(MANILA_LON - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * r_earth * np.arcsin(np.sqrt(a))


def _bearing_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Forward azimuth from each (lat, lon) toward Manila Bay, in degrees 0-360."""
    phi1 = np.radians(lat)
    phi2 = np.radians(MANILA_LAT)
    dlam = np.radians(MANILA_LON - lon)
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def build_storm_frame(track: pd.DataFrame, sea_level_m: pd.Series, name: str) -> pd.DataFrame:
    """Merge interpolated track with surge residual and spatial features."""
    # Interpolate track features onto the tide timestamps.
    grid = pd.date_range(track.index.min(), track.index.max(), freq="6min")
    storm = track[TRACK_FEATURES].reindex(grid).interpolate(method="time", limit_direction="both")
    storm = storm.reindex(sea_level_m.index)

    residual = surge_residual(sea_level_m)
    merged = pd.concat([storm, residual], axis=1)
    merged = merged.dropna(subset=TRACK_FEATURES + ["surge_residual"])

    merged["wind_kts"] = merged["USA_WIND"]
    merged["pressure_deficit_hpa"] = REFERENCE_PRESSURE_HPA - merged["USA_PRES"]
    merged["distance_to_manila_km"] = _haversine_km(
        merged["LAT"].to_numpy(), merged["LON"].to_numpy()
    )
    merged["approach_angle_deg"] = _bearing_deg(
        merged["LAT"].to_numpy(), merged["LON"].to_numpy()
    )
    frame = merged[MODEL_FEATURES + ["surge_residual"]]
    print(f"{name}: {len(frame)} training rows.")
    return frame


def build_dataset(tracks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the full training set, with UHSLC and a local fallback path."""
    frames: list[pd.DataFrame] = []
    try:
        for name in STORMS:
            track = tracks[name]
            start = track.index.min() - STORM_MARGIN
            end = track.index.max() + STORM_MARGIN
            sea_level = fetch_uhslc_tide(start, end)
            frames.append(build_storm_frame(track, sea_level, name))
    except Exception as exc:  # noqa: BLE001 - fallback on any UHSLC failure
        print(f"UHSLC fetch failed ({exc}). Falling back to manila_tide_2014.csv + Rammasun.")
        sea_level = load_fallback_tide()
        frames = [build_storm_frame(tracks["RAMMASUN"], sea_level, "RAMMASUN")]

    dataset = pd.concat(frames, axis=0)
    dataset = dataset.drop_duplicates()
    print(f"Combined dataset: {len(dataset)} rows across {len(frames)} storm(s).")
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_save(df: pd.DataFrame) -> None:
    """Train XGBoostRegressor on spatial features -> surge residual."""
    X = df[MODEL_FEATURES].to_numpy()
    y = df["surge_residual"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.4f} m | Test R2: {r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    df.to_csv(DATASET_CSV, index=False)
    print(f"Model saved to {MODEL_PATH}")


def main() -> None:
    download_ibtracs(IBTRACS_URL, RAW_CSV)
    tracks = load_storm_tracks(RAW_CSV)
    dataset = build_dataset(tracks)
    if len(dataset) < 50:
        sys.exit("Not enough training rows. Aborting.")
    train_and_save(dataset)


if __name__ == "__main__":
    main()
