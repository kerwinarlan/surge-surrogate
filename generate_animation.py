"""Generate assets/surge_animation_map.gif.

Reads the Rammasun (Glenda) 2014 track and the trained surrogate model,
then animates a 1x2 figure over a real geographic basemap:

  Left  - the storm track (EPSG:4326 -> EPSG:3857 Web Mercator) animated
          across a CartoDB Voyager basemap zoomed to the Philippines /
          Manila Bay (117-125 deg E, 12-18 deg N). A red star marks the
          exact location of Manila Bay (14.58 N, 120.97 E).
  Right - line graph of the model-predicted surge residual (m) growing
          as time progresses, peaking near landfall.

The GIF is written to assets/surge_animation_map.gif.
"""

import os

import contextily as ctx
import geopandas as gpd
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
OUT_GIF = os.path.join(ROOT, "assets", "surge_animation_map.gif")

MANILA_LAT, MANILA_LON = 14.58, 120.97
REFERENCE_PRESSURE_HPA = 1013.25
EARTH_RADIUS_KM = 6371.0

# Philippines / Manila Bay view in geographic coordinates.
VIEW_LON = (117.0, 125.0)
VIEW_LAT = (12.0, 18.0)
# The storm is inside the view box only from 2014-07-15 ~03:48 to
# 2014-07-16 ~16:48; trim the animation to the crossing period.
TRIM_START = pd.Timestamp("2014-07-15 00:00")
TRIM_END = pd.Timestamp("2014-07-17 00:00")

SMOOTH_MINUTES = 30   # display smoothing of the prediction line
STRIDE = 4            # sample every 4th 6-minute point
FPS = 12
DPI = 100
BASEMAP_ZOOM = 8

CRS_4326 = "EPSG:4326"
CRS_3857 = "EPSG:3857"

MODEL_FEATURES = [
    "wind_kts",
    "pressure_deficit_hpa",
    "distance_to_manila_km",
    "approach_angle_deg",
]


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
def to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert EPSG:4326 (lon, lat) coordinates to EPSG:3857 Web Mercator."""
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=CRS_4326)
    geom = gdf.to_crs(CRS_3857).geometry
    return geom.x.to_numpy(), geom.y.to_numpy()


def view_extent_mercator() -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of the Philippines view in Mercator."""
    x, y = to_mercator(
        np.array([VIEW_LON[0], VIEW_LON[1]]),
        np.array([VIEW_LAT[0], VIEW_LAT[1]]),
    )
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


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
    """Return track (Mercator) + model predictions aligned to tide timestamps."""
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
    )

    # Convert storm positions to Web Mercator and trim to the crossing window.
    x, y = to_mercator(lon[valid], lat[valid])
    trim = (ts >= TRIM_START) & (ts <= TRIM_END)
    ts_t = ts[trim]
    x_t, y_t = x[trim], y[trim]
    wind_t = wind[valid][trim]
    dist_t = dist[valid][trim]
    pred_t = pred.to_numpy()[trim]

    idx = np.arange(0, len(ts_t), STRIDE)
    manila_x, manila_y = to_mercator(
        np.array([MANILA_LON]), np.array([MANILA_LAT])
    )
    track_x, track_y = to_mercator(
        track["LON"].to_numpy(), track["LAT"].to_numpy()
    )

    return {
        "time": ts_t[idx],
        "x": x_t[idx],
        "y": y_t[idx],
        "wind": wind_t[idx],
        "dist": dist_t[idx],
        "pred": pred_t[idx],
        "track_x": track_x,
        "track_y": track_y,
        "manila_x": float(manila_x[0]),
        "manila_y": float(manila_y[0]),
        "peak_idx": int(np.argmax(pred_t[idx])),
    }


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def add_basemap(ax: plt.Axes) -> None:
    """Render a CartoDB Voyager basemap, with fallbacks for robustness."""
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager,
                        crs=CRS_3857, zoom=BASEMAP_ZOOM)
        print("Basemap: CartoDB Voyager.")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"CartoDB Voyager basemap failed ({exc}); trying OpenStreetMap.")
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        crs=CRS_3857, zoom=BASEMAP_ZOOM)
        print("Basemap: OpenStreetMap.")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"OpenStreetMap basemap failed ({exc}); using plain background.")
        ax.set_facecolor("#cfe6f2")


def build_figure(data: dict) -> tuple:
    """Create the 1x2 figure, basemap, and static decorations."""
    fig, (ax_map, ax_surge) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: real-world map panel --------------------------------------
    xmin, xmax, ymin, ymax = view_extent_mercator()
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    add_basemap(ax_map)
    ax_map.set_axis_off()

    # Red star for Manila Bay's exact location.
    ax_map.scatter(
        [data["manila_x"]], [data["manila_y"]], marker="*", s=340,
        c="#d62728", edgecolors="black", linewidths=0.8, zorder=8,
    )
    ax_map.annotate(
        "Manila Bay", xy=(data["manila_x"], data["manila_y"]),
        xytext=(data["manila_x"] + 45000, data["manila_y"] + 70000),
        fontsize=10, fontweight="bold", color="#7a120f",
        arrowprops=dict(arrowstyle="->", color="#7a120f", lw=1.2),
    )

    # Full historical track, faint, for context.
    ax_map.plot(data["track_x"], data["track_y"], color="#333333",
                alpha=0.5, linewidth=1.2, zorder=2)

    # Animated trail (line + wind-colored scatter) and storm marker.
    (trail_line,) = ax_map.plot([], [], color="#1f77b4", linewidth=2.2, zorder=4)
    trail_scatter = ax_map.scatter([], [], c=[], cmap="plasma", s=34, zorder=5)
    (storm_marker,) = ax_map.plot([], [], "o", markersize=14, mfc="#d62728",
                                  mec="white", mew=1.8, zorder=9)
    dist_text = ax_map.text(0.03, 0.95, "", transform=ax_map.transAxes,
                            va="top", fontsize=10,
                            bbox=dict(boxstyle="round", fc="white", alpha=0.9))

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
                              bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax_surge.set_title("Predicted surge residual (model)", fontsize=11)

    fig.suptitle("Typhoon Rammasun (Glenda) - Surge Surrogate Forecast for Manila Bay",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.28)
    return (fig, ax_map, ax_surge, trail_line, trail_scatter, storm_marker,
            dist_text, surge_line, surge_dot, peak_marker, peak_text)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_animation_data()
    n = len(data["time"])
    peak_idx = data["peak_idx"]
    print(f"Animation window: {data['time'][0]} to {data['time'][-1]} "
          f"({n} frames).")

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
        trail_line.set_data(data["x"][: i + 1], data["y"][: i + 1])
        trail_scatter.set_offsets(np.column_stack([data["x"][: i + 1],
                                                   data["y"][: i + 1]]))
        trail_scatter.set_array(data["wind"][: i + 1])
        trail_scatter.set_clim(data["wind"].min(), data["wind"].max())
        storm_marker.set_data([data["x"][i]], [data["y"][i]])
        dist_text.set_text(
            f"Distance to Manila Bay: {data['dist'][i]:.0f} km\n"
            f"Wind: {data['wind'][i]:.0f} kts"
        )

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
