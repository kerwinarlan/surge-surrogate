"""Generate assets/surge_animation.gif.

Reads the Rammasun (Glenda) 2014 track and the trained surrogate model,
then animates a 1x2 figure:

  Left  - map-like scatter of the storm position approaching Manila Bay
          (14.58 N, 120.97 E), with a wind-speed-colored trail and a
          dynamically zooming view that frames the closing gap.
  Right - line graph of the model-predicted surge residual (m) growing
          as time progresses, peaking near landfall.

The GIF is written to assets/surge_animation.gif.
"""

import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend
import matplotlib.animation as animation  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACK_CSV = os.path.join(ROOT, "data", "glenda_track_2014.csv")
TIDE_CSV = os.path.join(ROOT, "manila_tide_2014.csv")
MODEL_PATH = os.path.join(ROOT, "models", "surge_model.pkl")
OUT_GIF = os.path.join(ROOT, "assets", "surge_animation.gif")

MANILA_LAT, MANILA_LON = 14.58, 120.97
REFERENCE_PRESSURE_HPA = 1013.25
EARTH_RADIUS_KM = 6371.0

SMOOTH_MINUTES = 30   # display smoothing of the prediction line
STRIDE = 4            # sample every 4th 6-minute point -> ~183 frames
FPS = 12
DPI = 100

MODEL_FEATURES = [
    "wind_kts",
    "pressure_deficit_hpa",
    "distance_to_manila_km",
    "approach_angle_deg",
]


# ---------------------------------------------------------------------------
# Geometry helpers (must match pipeline.py)
# ---------------------------------------------------------------------------
def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance from each (lat, lon) to Manila Bay, in km."""
    phi1 = np.radians(lat)
    phi2 = np.radians(MANILA_LAT)
    dphi = np.radians(MANILA_LAT - lat)
    dlam = np.radians(MANILA_LON - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Forward azimuth from each (lat, lon) toward Manila Bay, in degrees."""
    phi1 = np.radians(lat)
    phi2 = np.radians(MANILA_LAT)
    dlam = np.radians(MANILA_LON - lon)
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_animation_data() -> dict:
    """Return storm track + model predictions aligned to tide timestamps."""
    for path in (TRACK_CSV, TIDE_CSV, MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    track = pd.read_csv(TRACK_CSV)
    track["ISO_TIME"] = pd.to_datetime(track["ISO_TIME"])
    track = track.set_index("ISO_TIME").sort_index()

    tide = pd.read_csv(TIDE_CSV, parse_dates=["time"]).set_index("time").sort_index()

    # Interpolate the 3-hourly track onto the 6-minute tide timestamps.
    grid = pd.date_range(track.index.min(), track.index.max(), freq="6min")
    storm = track[["LAT", "LON", "USA_WIND", "USA_PRES"]]
    storm = storm.reindex(grid).interpolate(method="time", limit_direction="both")
    storm = storm.reindex(tide.index)

    lat = storm["LAT"].to_numpy()
    lon = storm["LON"].to_numpy()
    dist = haversine_km(lat, lon)
    angle = bearing_deg(lat, lon)
    wind = storm["USA_WIND"].to_numpy()
    deficit = REFERENCE_PRESSURE_HPA - storm["USA_PRES"].to_numpy()

    features = np.column_stack([wind, deficit, dist, angle])
    valid = ~np.isnan(features).any(axis=1)

    model = joblib.load(MODEL_PATH)
    pred_raw = model.predict(features[valid])

    ts = storm.index[valid]
    # Light centered smoothing for a clean line; keeps the landfall peak.
    pred = (
        pd.Series(pred_raw, index=ts)
        .rolling(f"{SMOOTH_MINUTES}min", center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    idx = np.arange(0, len(ts), STRIDE)
    return {
        "time": ts[idx],
        "lat": lat[valid][idx],
        "lon": lon[valid][idx],
        "wind": wind[valid][idx],
        "dist": dist[valid][idx],
        "pred": pred[idx],
        "track_lat": track["LAT"].to_numpy(),
        "track_lon": track["LON"].to_numpy(),
        "peak_idx": int(np.argmax(pred[idx])),
    }


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def build_figure(data: dict) -> tuple:
    """Create the 1x2 figure and static decorations."""
    fig, (ax_map, ax_surge) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: map panel --------------------------------------------------
    ax_map.set_facecolor("#cfe6f2")
    ax_map.grid(True, linestyle="--", color="white", alpha=0.7, linewidth=0.8)
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.scatter(
        [MANILA_LON], [MANILA_LAT], marker="*", s=260, c="#d62728", edgecolors="k",
        zorder=6, label="Manila Bay",
    )
    ax_map.annotate(
        "Manila Bay\n(14.58°N, 120.97°E)", xy=(MANILA_LON, MANILA_LAT),
        xytext=(MANILA_LON + 1.2, MANILA_LAT + 1.6), fontsize=9, color="#8c1d18",
        arrowprops=dict(arrowstyle="->", color="#8c1d18", lw=1.0),
    )
    # Full historical track, faint, for context.
    ax_map.plot(
        data["track_lon"], data["track_lat"], color="#444444", alpha=0.45,
        linewidth=1.0, zorder=2,
    )
    ax_map.plot([], [], color="#444444", alpha=0.45, linewidth=1.0,
                label="Full track")
    # Animated trail (line + wind-colored scatter).
    (trail_line,) = ax_map.plot([], [], color="#1f77b4", linewidth=2.0, zorder=4)
    trail_scatter = ax_map.scatter([], [], c=[], cmap="plasma", s=28, zorder=5)
    (storm_marker,) = ax_map.plot([], [], "o", markersize=13, mfc="#d62728",
                                  mec="white", mew=1.6, zorder=7)
    dist_text = ax_map.text(0.03, 0.95, "", transform=ax_map.transAxes,
                            va="top", fontsize=10,
                            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # ---- Right: surge panel ----------------------------------------------
    ax_surge.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax_surge.set_ylabel("Predicted surge residual (m)")
    ax_surge.set_xlabel("Time (UTC)")
    ax_surge.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_surge.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    (surge_line,) = ax_surge.plot([], [], color="#1f77b4", linewidth=2.4, zorder=4)
    (surge_dot,) = ax_surge.plot([], [], "o", color="#d62728", markersize=8, zorder=6)
    (peak_marker,) = ax_surge.plot([], [], "*", color="#d62728", markersize=16, zorder=7)
    peak_text = ax_surge.text(0.03, 0.9, "", transform=ax_surge.transAxes,
                              va="top", fontsize=10,
                              bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax_surge.set_title("Predicted surge residual (model)", fontsize=11)

    fig.suptitle("Typhoon Rammasun (Glenda) - Surge Surrogate Forecast for Manila Bay",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.28)
    return fig, ax_map, ax_surge, trail_line, trail_scatter, storm_marker, \
        dist_text, surge_line, surge_dot, peak_marker, peak_text


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_animation_data()
    n = len(data["time"])
    peak_idx = data["peak_idx"]

    (fig, ax_map, ax_surge, trail_line, trail_scatter, storm_marker, dist_text,
     surge_line, surge_dot, peak_marker, peak_text) = build_figure(data)

    y_min, y_max = float(np.min(data["pred"])), float(np.max(data["pred"]))
    pad = 0.15 * (y_max - y_min)
    ax_surge.set_ylim(y_min - pad, y_max + pad)
    ax_surge.set_xlim(data["time"][0], data["time"][-1])

    def animate(frame: int) -> tuple:
        i = frame
        t = data["time"][i]

        # ---- Left: map ---------------------------------------------------
        trail_line.set_data(data["lon"][: i + 1], data["lat"][: i + 1])
        trail_scatter.set_offsets(np.column_stack([data["lon"][: i + 1],
                                                   data["lat"][: i + 1]]))
        trail_scatter.set_array(data["wind"][: i + 1])
        trail_scatter.set_clim(data["wind"].min(), data["wind"].max())
        storm_marker.set_data([data["lon"][i]], [data["lat"][i]])
        dist_text.set_text(
            f"Distance to Manila Bay: {data['dist'][i]:.0f} km\n"
            f"Wind: {data['wind'][i]:.0f} kts"
        )

        # Dynamic zoom: frame the storm-to-Manila gap as it closes.
        half_lon = float(np.interp(data["dist"][i], [0, 1400], [6.0, 22.0]))
        half_lat = half_lon * 0.55
        cx = 0.5 * data["lon"][i] + 0.5 * MANILA_LON
        cy = 0.5 * data["lat"][i] + 0.5 * MANILA_LAT
        ax_map.set_xlim(cx - half_lon, cx + half_lon)
        ax_map.set_ylim(cy - half_lat, cy + half_lat)

        # ---- Right: surge line ------------------------------------------
        surge_line.set_data(data["time"][: i + 1], data["pred"][: i + 1])
        surge_dot.set_data([t], [data["pred"][i]])
        ax_surge.set_title(
            f"Predicted surge residual (model) - {t.strftime('%m-%d %H:%M')} UTC",
            fontsize=11,
        )
        if i >= peak_idx:
            peak_marker.set_data([data["time"][peak_idx]], [data["pred"][peak_idx]])
            peak_text.set_text(
                f"Peak surge: {data['pred'][peak_idx]:.2f} m\n"
                f"at {data['time'][peak_idx].strftime('%m-%d %H:%M')} UTC"
            )
        return (trail_line, trail_scatter, storm_marker, dist_text,
                surge_line, surge_dot, peak_marker, peak_text)

    os.makedirs(os.path.dirname(OUT_GIF), exist_ok=True)
    anim = animation.FuncAnimation(
        fig, animate, frames=n, interval=1000 // FPS, blit=False
    )
    writer = animation.PillowWriter(fps=FPS)
    print(f"Rendering {n} frames to {OUT_GIF} ...")
    anim.save(OUT_GIF, writer=writer, dpi=DPI,
              progress_callback=lambda i, total: print(f"  frame {i + 1}/{total}", end="\r"))
    print(f"\nSaved GIF: {OUT_GIF} ({os.path.getsize(OUT_GIF) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
