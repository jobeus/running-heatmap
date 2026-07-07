#!/usr/bin/env python3
"""Personal running heatmaps from a Strava data export.

Generates raster heatmaps of your runs on a dark basemap and writes them to a
local output directory:

  outputs/heatmap.html    interactive Folium map with six switchable layers
  outputs/layers/*.png     each layer as a standalone image

Layers:
  Frequency (linear)     Orange   How often you've run each path
  Frequency (log)        Orange   Same but on a log scale
  Pace (average)         Blue     Average pace - brighter = faster
  Heart rate (average)   Red      Average HR - brighter = higher
  Gradient (absolute)    White    How steep - brighter = steeper
  Gradient (change)      Green/purple  Direction - green descend, purple ascend

Usage:
  1. Download your Strava data export and unzip it next to this script.
  2. Point ACTIVITIES_DIR at that folder and set DATE_FROM / DATE_TO below.
  3. python heatmap.py
"""

import math
import gzip
import json
import os
import warnings
import base64
import xml.etree.ElementTree as ET
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import fitparse
import folium
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================

# Activity source
ACTIVITIES_DIR = "export_1406017"
ACTIVITY_TYPES = ["Run", "Walk"]   # e.g. ["Run", "Ride", "Hike", "Walk"]

# Date filter
DATE_FROM = "2026-01-01"   # inclusive; None = no lower limit
DATE_TO   = None           # inclusive; None = today

# Home location
HOME_LAT  = None
HOME_LON  = None
RADIUS_KM = 35.0

# Treadmill exclusion
GPS_SPREAD_MIN_M = 200

# Raster settings
METERS_PER_PIXEL     = 3      # resolution
PADDING_M            = 500    # border padding around all tracks
TRACK_CLIP_RADIUS_KM = 12.0   # set to None to use raw GPS extents

# Glow
BLUR_SIGMA_PX = 10     # Gaussian blur radius in pixels
MAP_OPACITY   = 0.85

# Colour range
SPEED_MIN_MS   = None   # m/s; None = auto
SPEED_MAX_MS   = None
HR_MIN_BPM     = None   # bpm; None = auto
HR_MAX_BPM     = None
AUTO_RANGE_PCT = 5      # percentile clipping for auto range

# Output
ACTIVITIES_CSV = f"{ACTIVITIES_DIR}/activities.csv"
TRACK_CACHE    = "cache/track_cache.json"
OUTPUT_DIR     = "outputs"
OUTPUT_HTML    = f"{OUTPUT_DIR}/heatmap.html"
LAYERS_DIR     = f"{OUTPUT_DIR}/layers"


# ============================================================================
# Step 1 — Filter activities
# ============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _open_maybe_gz(filepath):
    """Open a track file, transparently decompressing .gz."""
    filepath = str(filepath)
    return gzip.open(filepath, "rb") if filepath.endswith(".gz") else open(filepath, "rb")


def _parse_fit(f):
    points = []
    for msg in fitparse.FitFile(f).get_messages("record"):
        d = {x.name: x.value for x in msg}
        # Use `is not None` not truthiness — lat/lon of 0° are valid coordinates
        if d.get("position_lat") is None or d.get("position_long") is None:
            continue
        lat = d["position_lat"]  * (180 / 2**31)
        lon = d["position_long"] * (180 / 2**31)
        # Use `is not None` not `or` — zero speed (stationary) is a valid value
        speed = d.get("enhanced_speed") if d.get("enhanced_speed") is not None else d.get("speed")
        hr    = d.get("heart_rate")
        alt   = d.get("enhanced_altitude") if d.get("enhanced_altitude") is not None else d.get("altitude")
        points.append([lat, lon, speed, hr, alt])
    return points


def _first_float(el, tag):
    """Return the first descendant matching `tag` (any namespace) as float, or None.

    Uses findall, not iter — Element.iter() does exact tag matching and ignores
    the ``{*}`` namespace wildcard, whereas findall's ElementPath honours it.
    """
    for child in el.findall(f".//{{*}}{tag}"):
        if child.text is not None and child.text.strip():
            try:
                return float(child.text)
            except ValueError:
                return None
    return None


def _parse_gpx(f):
    """Parse a GPX track. Strava GPX carries ele + heart rate; speed is usually absent."""
    points = []
    root = ET.parse(f).getroot()
    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.get("lat")
        lon = trkpt.get("lon")
        if lat is None or lon is None:
            continue
        alt   = _first_float(trkpt, "ele")
        speed = _first_float(trkpt, "speed")
        hr_f  = _first_float(trkpt, "hr")   # gpxtpx:hr in TrackPointExtension
        hr    = int(hr_f) if hr_f is not None else None
        points.append([float(lat), float(lon), speed, hr, alt])
    return points


def _parse_tcx(f):
    """Parse a TCX track. Trackpoints carry position, altitude, HR, and (in extensions) speed."""
    points = []
    root = ET.parse(f).getroot()
    for tp in root.findall(".//{*}Trackpoint"):
        pos = tp.find("{*}Position")
        if pos is None:
            continue
        lat = pos.find("{*}LatitudeDegrees")
        lon = pos.find("{*}LongitudeDegrees")
        if lat is None or lon is None or not lat.text or not lon.text:
            continue
        alt   = _first_float(tp, "AltitudeMeters")
        speed = _first_float(tp, "Speed")   # ns3:TPX/Speed in extensions
        hr_el = tp.find("{*}HeartRateBpm")
        hr_f  = _first_float(hr_el, "Value") if hr_el is not None else None
        hr    = int(hr_f) if hr_f is not None else None
        points.append([float(lat.text), float(lon.text), speed, hr, alt])
    return points


def read_track_points(filepath):
    """
    Parse a Strava activity file (.fit, .gpx, .tcx, optionally .gz-compressed) and
    return a list of [lat, lon, speed_ms, hr_bpm, alt_m] per GPS point.
    speed_ms, hr_bpm, and alt_m are None where the sensor had no reading.
    """
    fp   = str(filepath)
    base = fp[:-3] if fp.endswith(".gz") else fp
    try:
        with _open_maybe_gz(fp) as f:
            if base.endswith(".fit"):
                return _parse_fit(f)
            if base.endswith(".gpx"):
                return _parse_gpx(f)
            if base.endswith(".tcx"):
                return _parse_tcx(f)
    except Exception as e:
        print(f"  Warning {filepath}: {e}")
    return []


def get_gps_start(filepath):
    """Return (start_lat, start_lon, spread_m) from any supported track, or (None, None, None)."""
    pts = read_track_points(filepath)
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    if not lats:
        return None, None, None
    mid_lat  = (min(lats) + max(lats)) / 2
    spread_m = max(
        (max(lats) - min(lats)) * 111_000,
        (max(lons) - min(lons)) * 111_000 * math.cos(math.radians(mid_lat)),
    )
    return lats[0], lons[0], spread_m


def detect_home(runs_with_gps):
    """
    Find the most common run-start location as a proxy for home.
    Bins starting points to a ~1 km grid, finds the densest cell,
    then returns the mean of actual coordinates in that cell.
    """
    cell_lats, cell_lons = {}, {}
    for lat, lon in zip(runs_with_gps["start_lat"], runs_with_gps["start_lon"]):
        cell = (round(lat, 2), round(lon, 2))   # ~1 km grid
        cell_lats.setdefault(cell, []).append(lat)
        cell_lons.setdefault(cell, []).append(lon)
    best_cell = max(cell_lats, key=lambda c: len(cell_lats[c]))
    home_lat  = sum(cell_lats[best_cell]) / len(cell_lats[best_cell])
    home_lon  = sum(cell_lons[best_cell]) / len(cell_lons[best_cell])
    return home_lat, home_lon, len(cell_lats[best_cell])


def filter_activities():
    """Load, date-filter and home-radius-filter the activities. Returns the runs DataFrame plus home coords."""
    df = pd.read_csv(ACTIVITIES_CSV)
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], format="mixed", dayfirst=True)
    runs = df[df["Activity Type"].isin(ACTIVITY_TYPES)].copy()
    print(f"Total matching activities in export: {len(runs)}")

    date_from = pd.Timestamp(DATE_FROM) if DATE_FROM else pd.Timestamp.min
    date_to   = pd.Timestamp(DATE_TO)   if DATE_TO   else pd.Timestamp(date.today())
    # A pure date parses to midnight; extend to end-of-day so activities later that
    # day (e.g. today's run) are included in an inclusive filter.
    if date_to == date_to.normalize():
        date_to = date_to + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    runs = runs[runs["Activity Date"].between(date_from, date_to)].copy()
    print(f"After date filter ({date_from.date()} – {date_to.date()}): {len(runs)}")

    # Parse GPS start points (cached per export)
    gps_cache_path = Path(ACTIVITIES_DIR) / "_gps_cache.json"
    gps_cache = json.loads(gps_cache_path.read_text()) if gps_cache_path.exists() else {}

    rows = []
    for _, row in runs.iterrows():
        fn = str(row["Filename"])
        if fn in gps_cache:
            lat, lon, spread = gps_cache[fn]
        else:
            lat, lon, spread = get_gps_start(Path(ACTIVITIES_DIR) / fn)
            gps_cache[fn] = [lat, lon, spread]   # cache even if no GPS (None values)
        rows.append({**row, "start_lat": lat, "start_lon": lon, "gps_spread_m": spread})

    gps_cache_path.write_text(json.dumps(gps_cache))

    runs = pd.DataFrame(rows)
    runs = runs[runs["start_lat"].notna() & (runs["gps_spread_m"] >= GPS_SPREAD_MIN_M)].copy()
    print(f"After removing no-GPS / indoor: {len(runs)}")

    # Auto-detect or use manual home location
    if HOME_LAT is None or HOME_LON is None:
        home_lat, home_lon, n_home_starts = detect_home(runs)
        print(f"Auto-detected home: {home_lat:.4f}, {home_lon:.4f}  "
              f"({n_home_starts} of {len(runs)} activities started there)")
    else:
        home_lat, home_lon = HOME_LAT, HOME_LON
        print(f"Using manual home: {home_lat}, {home_lon}")

    # Filter by home radius
    runs["dist_from_home_km"] = runs.apply(
        lambda r: haversine_km(home_lat, home_lon, r["start_lat"], r["start_lon"]), axis=1
    )
    runs = runs[runs["dist_from_home_km"] <= RADIUS_KM].copy()
    print(f"After home-radius filter (≤{RADIUS_KM} km): {len(runs)} activities")

    return runs, home_lat, home_lon


# ============================================================================
# Step 2 — Parse GPS tracks with speed and heart rate
# ============================================================================

def load_fit_track_full(filepath):
    """Backwards-compatible alias — parse any supported track format."""
    return read_track_points(filepath)


def load_tracks(runs):
    """Parse (cached) GPS tracks for each run. Returns list of (label, points)."""
    track_cache_path = Path(TRACK_CACHE)
    track_cache = json.loads(track_cache_path.read_text()) if track_cache_path.exists() else {}

    # Invalidate cache entries that are missing the altitude field (old 4-field format)
    stale = [k for k, v in track_cache.items() if v and len(v[0]) < 5]
    if stale:
        print(f"  Clearing {len(stale)} stale cache entries (missing altitude field)")
        for k in stale:
            del track_cache[k]

    tracks = []   # list of (label, [[lat, lon, speed, hr, alt], ...])
    for _, row in runs.iterrows():
        fn  = str(row["Filename"])
        fp  = Path(ACTIVITIES_DIR) / fn
        lbl = f"{row['Activity Date'].date()} {row['Activity Name']}"

        if fn in track_cache:
            pts = track_cache[fn]
        else:
            print(f"  Parsing {fn} …")
            pts = load_fit_track_full(fp)
            track_cache[fn] = pts

        if pts:
            tracks.append((lbl, pts))

    track_cache_path.write_text(json.dumps(track_cache))

    total_pts  = sum(len(t) for _, t in tracks)
    all_speeds = [p[2] for _, pts in tracks for p in pts if p[2] is not None]
    all_hrs    = [p[3] for _, pts in tracks for p in pts if p[3] is not None]
    all_alts   = [p[4] for _, pts in tracks for p in pts if p[4] is not None]

    print(f"\nLoaded {len(tracks)} tracks, {total_pts:,} GPS points")
    if all_speeds:
        print(f"Speed:  {np.percentile(all_speeds, 5):.2f}–{np.percentile(all_speeds, 95):.2f} m/s "
              f"(5th–95th pct)  median {np.median(all_speeds):.2f} m/s")
    else:
        print("Speed:  no data")
    if all_hrs:
        print(f"HR:     {np.percentile(all_hrs, 5):.0f}–{np.percentile(all_hrs, 95):.0f} bpm "
              f"(5th–95th pct)  median {np.median(all_hrs):.0f} bpm")
    else:
        print("HR:     no data")
    if all_alts:
        n_alt = sum(1 for _, pts in tracks for p in pts if p[4] is not None)
        print(f"Alt:    {min(all_alts):.0f}–{max(all_alts):.0f} m  ({n_alt:,} points with altitude)")
    else:
        print("Alt:    no data")

    return tracks


# ============================================================================
# Step 3 — Rasterise: paint value grids in one pass
# ============================================================================

def rasterise(tracks, home_lat, home_lon):
    """Rasterise all tracks into value grids. Returns a dict of grids and geo metadata."""
    if not tracks:
        raise ValueError("No tracks loaded — check ACTIVITIES_DIR, ACTIVITY_TYPES, and date filters.")

    # Rasterising in Web Mercator means pixels map directly to basemap tile space —
    # no reprojection needed when placing the image overlay in Leaflet.
    to_wm   = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    from_wm = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # UTM for clip radius and gradient distance — true ground metres
    utm_zone = int((home_lon + 180) / 6) + 1
    utm_base = 32700 if home_lat < 0 else 32600
    utm_crs  = f"EPSG:{utm_base + utm_zone}"
    to_utm   = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    print(f"Rasterising in EPSG:3857; clip check via {utm_crs}")

    home_x_utm, home_y_utm = to_utm.transform(home_lon, home_lat)
    clip_m = TRACK_CLIP_RADIUS_KM * 1000 if TRACK_CLIP_RADIUS_KM is not None else None

    # Grid bounds
    if clip_m is not None:
        clipped_wm_xs, clipped_wm_ys = [], []
        for _, pts in tracks:
            lats_a = np.array([p[0] for p in pts])
            lons_a = np.array([p[1] for p in pts])
            xs_utm, ys_utm = to_utm.transform(lons_a, lats_a)
            mask = ((xs_utm - home_x_utm)**2 + (ys_utm - home_y_utm)**2) <= clip_m**2
            if mask.any():
                xs_wm_c, ys_wm_c = to_wm.transform(lons_a[mask], lats_a[mask])
                clipped_wm_xs.extend(xs_wm_c.tolist())
                clipped_wm_ys.extend(ys_wm_c.tolist())
        x_min_wm = min(clipped_wm_xs) - PADDING_M
        x_max_wm = max(clipped_wm_xs) + PADDING_M
        y_min_wm = min(clipped_wm_ys) - PADDING_M
        y_max_wm = max(clipped_wm_ys) + PADDING_M
        print(f"Grid from clipped GPS extents (clip radius: {TRACK_CLIP_RADIUS_KM} km)")
    else:
        all_lats = np.array([p[0] for _, pts in tracks for p in pts])
        all_lons = np.array([p[1] for _, pts in tracks for p in pts])
        xs_wm_all, ys_wm_all = to_wm.transform(all_lons, all_lats)
        x_min_wm = xs_wm_all.min() - PADDING_M
        x_max_wm = xs_wm_all.max() + PADDING_M
        y_min_wm = ys_wm_all.min() - PADDING_M
        y_max_wm = ys_wm_all.max() + PADDING_M
        print("Grid from GPS extents (no clip radius set)")

    grid_w = int((x_max_wm - x_min_wm) / METERS_PER_PIXEL) + 1
    grid_h = int((y_max_wm - y_min_wm) / METERS_PER_PIXEL) + 1
    print(f"Grid: {grid_w} × {grid_h} px at {METERS_PER_PIXEL} Mercator-m/px")

    count_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    speed_sum  = np.zeros((grid_h, grid_w), dtype=np.float32)
    speed_n    = np.zeros((grid_h, grid_w), dtype=np.float32)
    hr_sum     = np.zeros((grid_h, grid_w), dtype=np.float32)
    hr_n       = np.zeros((grid_h, grid_w), dtype=np.float32)
    grad_sum   = np.zeros((grid_h, grid_w), dtype=np.float32)
    grad_n     = np.zeros((grid_h, grid_w), dtype=np.float32)
    elev_sum   = np.zeros((grid_h, grid_w), dtype=np.float32)
    elev_n     = np.zeros((grid_h, grid_w), dtype=np.float32)

    def paint_segment(x1, y1, x2, y2, speed_val, hr_val, grad_val, elev_val):
        dx, dy  = x2 - x1, y2 - y1
        n_steps = max(int(max(abs(dx), abs(dy))) + 1, 1)
        h, w    = speed_sum.shape
        for i in range(n_steps + 1):
            t  = i / n_steps
            xi = int(round(x1 + t * dx))
            yi = int(round(y1 + t * dy))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            if speed_val is not None:
                speed_sum[yi, xi] += speed_val
                speed_n[yi, xi]   += 1
            if hr_val is not None:
                hr_sum[yi, xi] += hr_val
                hr_n[yi, xi]   += 1
            if grad_val is not None:
                grad_sum[yi, xi] += grad_val
                grad_n[yi, xi]   += 1
            if elev_val is not None:
                elev_sum[yi, xi] += elev_val
                elev_n[yi, xi]   += 1

    for label, pts in tracks:
        lats_a = np.array([p[0] for p in pts])
        lons_a = np.array([p[1] for p in pts])
        xs_utm, ys_utm = to_utm.transform(lons_a, lats_a)
        xs_wm,  ys_wm  = to_wm.transform(lons_a, lats_a)

        if clip_m is not None:
            _mask = ((xs_utm - home_x_utm)**2 + (ys_utm - home_y_utm)**2) <= clip_m**2
            if not _mask.any():
                continue
            pts    = [pts[i] for i in range(len(pts)) if _mask[i]]
            xs_utm = xs_utm[_mask]
            ys_utm = ys_utm[_mask]
            xs_wm  = xs_wm[_mask]
            ys_wm  = ys_wm[_mask]

        px = (xs_wm - x_min_wm) / METERS_PER_PIXEL
        py = (y_max_wm - ys_wm)  / METERS_PER_PIXEL

        for i in range(len(pts)):
            xi = int(round(px[i]))
            yi = int(round(py[i]))
            if 0 <= xi < grid_w and 0 <= yi < grid_h:
                count_grid[yi, xi] += 1

        for i in range(len(pts) - 1):
            s0, s1 = pts[i][2], pts[i+1][2]
            h0, h1 = pts[i][3], pts[i+1][3]
            a0, a1 = pts[i][4], pts[i+1][4]

            seg_speed = (s0 + s1) / 2 if s0 is not None and s1 is not None else (s0 if s0 is not None else s1)
            seg_hr    = (h0 + h1) / 2 if h0 is not None and h1 is not None else (h0 if h0 is not None else h1)

            if a0 is not None and a1 is not None:
                d_dist = math.sqrt((xs_utm[i+1] - xs_utm[i])**2 + (ys_utm[i+1] - ys_utm[i])**2)
                if d_dist >= 0.5:
                    seg_grad = abs(a1 - a0) / d_dist
                    seg_elev = a1 - a0
                else:
                    seg_grad = seg_elev = None
            else:
                seg_grad = seg_elev = None

            paint_segment(px[i], py[i], px[i+1], py[i+1], seg_speed, seg_hr, seg_grad, seg_elev)

    print(f"Count grid — max GPS pts/px: {count_grid.max():.0f}, "
          f"non-zero: {(count_grid > 0).sum():,}")
    print(f"Speed data — {(speed_n > 0).sum():,} pixels")
    print(f"HR data    — {(hr_n > 0).sum():,} pixels")
    print(f"Gradient   — {(grad_n > 0).sum():,} pixels")
    print(f"Elev change — {(elev_n > 0).sum():,} pixels")

    lon_nw, lat_nw = from_wm.transform(x_min_wm, y_max_wm)
    lon_se, lat_se = from_wm.transform(x_max_wm, y_min_wm)

    return {
        "count_grid": count_grid,
        "speed_sum": speed_sum, "speed_n": speed_n,
        "hr_sum": hr_sum, "hr_n": hr_n,
        "grad_sum": grad_sum, "grad_n": grad_n,
        "elev_sum": elev_sum, "elev_n": elev_n,
        "bounds": [[lat_se, lon_nw], [lat_nw, lon_se]],
        "centre": [(lat_nw + lat_se) / 2, (lon_nw + lon_se) / 2],
    }


# ============================================================================
# Step 4 — Blur and normalise
# ============================================================================

def presence_alpha(sample_count_grid, blur_sigma, pct=10):
    binary  = (sample_count_grid > 0).astype(np.float32)
    blurred = gaussian_filter(binary, sigma=blur_sigma)
    positive = blurred[binary > 0]
    if positive.size == 0:
        return blurred
    sat = np.percentile(positive, pct)
    return np.clip(blurred / sat, 0, 1) if sat > 0 else blurred


def blur_normalise(g):
    """Blur and normalise all raster grids into display-ready arrays. Returns a dict."""
    sigma = BLUR_SIGMA_PX
    count_grid = g["count_grid"]

    # Count grid
    b_count        = gaussian_filter(count_grid, sigma=sigma)
    count_norm     = b_count / b_count.max()
    count_log_norm = np.log1p(b_count) / np.log1p(b_count.max())

    # Speed (average) grid
    b_speed_sum = gaussian_filter(g["speed_sum"], sigma=sigma)
    b_speed_n   = gaussian_filter(g["speed_n"],   sigma=sigma)
    mean_speed  = np.where(b_speed_n > 0, b_speed_sum / b_speed_n, 0)
    visited_speeds = mean_speed[b_speed_n > 0.01]
    if len(visited_speeds):
        s_lo = SPEED_MIN_MS if SPEED_MIN_MS is not None else np.percentile(visited_speeds, AUTO_RANGE_PCT)
        s_hi = SPEED_MAX_MS if SPEED_MAX_MS is not None else np.percentile(visited_speeds, 100 - AUTO_RANGE_PCT)
        speed_norm = np.clip((mean_speed - s_lo) / (s_hi - s_lo), 0, 1)
        speed_norm = np.where(b_speed_n > 0, speed_norm, 0)
        _sw = gaussian_filter(speed_norm * (b_speed_n > 0.01).astype(float), sigma=sigma)
        _sn = gaussian_filter((b_speed_n > 0.01).astype(float), sigma=sigma)
        speed_norm = np.where(_sn > 0, _sw / _sn, 0)
        print(f"Pace (average) range: {s_lo:.2f}–{s_hi:.2f} m/s  "
              f"({'auto' if SPEED_MIN_MS is None else 'fixed'})  "
              f"≈ {1000/s_hi:.0f}–{1000/s_lo:.0f} sec/km")
    else:
        s_lo, s_hi = 1.0, 5.0
        speed_norm = np.zeros_like(mean_speed)
        print("Pace:          no speed data")

    # HR (average) grid
    b_hr_sum = gaussian_filter(g["hr_sum"], sigma=sigma)
    b_hr_n   = gaussian_filter(g["hr_n"],   sigma=sigma)
    mean_hr  = np.where(b_hr_n > 0, b_hr_sum / b_hr_n, 0)
    visited_hrs = mean_hr[g["hr_n"] > 0]
    if len(visited_hrs):
        hr_lo = HR_MIN_BPM if HR_MIN_BPM is not None else np.percentile(visited_hrs, AUTO_RANGE_PCT)
        hr_hi = HR_MAX_BPM if HR_MAX_BPM is not None else np.percentile(visited_hrs, 100 - AUTO_RANGE_PCT)
        hr_norm = np.clip((mean_hr - hr_lo) / (hr_hi - hr_lo), 0, 1)
        hr_norm = np.where(b_hr_n > 0, hr_norm, 0)
        _hw = gaussian_filter(hr_norm * (g["hr_n"] > 0).astype(float), sigma=sigma)
        _hn = gaussian_filter((g["hr_n"] > 0).astype(float), sigma=sigma)
        hr_norm = np.where(_hn > 0, _hw / _hn, 0)
        print(f"HR (average) range:   {hr_lo:.0f}–{hr_hi:.0f} bpm  "
              f"({'auto' if HR_MIN_BPM is None else 'fixed'})")
    else:
        hr_lo, hr_hi = 100, 180
        hr_norm = np.zeros_like(mean_hr)
        print("HR:            no heart rate data")

    # Gradient grid
    b_grad_sum = gaussian_filter(g["grad_sum"], sigma=sigma)
    b_grad_n   = gaussian_filter(g["grad_n"],   sigma=sigma)
    mean_grad  = np.where(b_grad_n > 0, b_grad_sum / b_grad_n, 0)
    visited_grads = mean_grad[b_grad_n > 0.01]
    n_grad_px = (g["grad_n"] > 0).sum()
    if n_grad_px and len(visited_grads):
        g_lo = np.percentile(visited_grads, AUTO_RANGE_PCT)
        g_hi = np.percentile(visited_grads, 100 - AUTO_RANGE_PCT)
        grad_norm = np.clip((mean_grad - g_lo) / (g_hi - g_lo), 0, 1)
        grad_norm = np.where(b_grad_n > 0, grad_norm, 0)
        observed_grads = visited_grads * 100
        print(f"Gradient:    {g_lo*100:.1f}%–{g_hi*100:.1f}% (auto range)  "
              f"median {np.median(observed_grads):.1f}%  "
              f"95th pct {np.percentile(observed_grads, 95):.1f}%")
    else:
        grad_norm = np.zeros_like(mean_grad)
        g_lo = g_hi = 0.0
        print("Gradient:    no altitude data")

    # Elevation change grid (signed)
    b_elev_sum = gaussian_filter(g["elev_sum"], sigma=sigma)
    b_elev_n   = gaussian_filter(g["elev_n"],   sigma=sigma)
    mean_elev  = np.where(b_elev_n > 0, b_elev_sum / b_elev_n, 0)
    n_elev_px  = (g["elev_n"] > 0).sum()
    if n_elev_px:
        visited_elevs = mean_elev[b_elev_n > 0.01]
        e_abs_hi = max(abs(np.percentile(visited_elevs, AUTO_RANGE_PCT)),
                       abs(np.percentile(visited_elevs, 100 - AUTO_RANGE_PCT)))
        elev_norm = np.clip(mean_elev / e_abs_hi, -1, 1)
        elev_norm = np.where(b_elev_n > 0, elev_norm, 0)
        _ew = gaussian_filter(elev_norm * (b_elev_n > 0.01).astype(float), sigma=sigma)
        _en = gaussian_filter((b_elev_n > 0.01).astype(float), sigma=sigma)
        elev_norm = np.where(_en > 0, _ew / _en, 0)
        print(f"Elev change: ±{e_abs_hi*100:.1f}% range  "
              f"median {np.median(visited_elevs)*100:.1f}%")
    else:
        elev_norm = np.zeros_like(mean_elev)
        print("Elev change: no altitude data")

    # Alpha masks
    alpha_speed = presence_alpha(g["speed_n"], sigma)
    alpha_hr    = presence_alpha(g["hr_n"],    sigma)
    _presence_grad = presence_alpha(g["grad_n"], sigma) if n_grad_px else np.zeros_like(grad_norm)
    alpha_grad = _presence_grad * (0.15 + 0.85 * grad_norm)
    alpha_elev = presence_alpha(g["elev_n"], sigma) if n_elev_px else np.zeros_like(elev_norm)

    return {
        "count_norm": count_norm, "count_log_norm": count_log_norm,
        "speed_norm": speed_norm, "alpha_speed": alpha_speed,
        "hr_norm": hr_norm, "alpha_hr": alpha_hr,
        "grad_norm": grad_norm, "alpha_grad": alpha_grad,
        "elev_norm": elev_norm, "alpha_elev": alpha_elev,
        "s_lo": s_lo, "s_hi": s_hi,
        "hr_lo": hr_lo, "hr_hi": hr_hi,
        "g_lo": g_lo, "g_hi": g_hi,
    }


# ============================================================================
# Step 5 — Colourmaps
# ============================================================================

def build_cmap(name, nodes):
    """Build a LinearSegmentedColormap from a list of (position, (R,G,B,A)) nodes."""
    pos   = [n[0] for n in nodes]
    cdict = {}
    for ci, ch in enumerate(("red", "green", "blue", "alpha")):
        vals = [n[1][ci] for n in nodes]
        cdict[ch] = [(pos[i], vals[i], vals[i]) for i in range(len(pos))]
    return mcolors.LinearSegmentedColormap(name, cdict, N=512)


def build_colormaps():
    # Orange — frequency: dark orange → amber → yellow → cream
    cmap_count = build_cmap("count", [
        (0.00, (0.00, 0.00, 0.00, 0.00)),
        (0.01, (0.40, 0.10, 0.00, 0.55)),
        (0.20, (0.99, 0.30, 0.01, 0.80)),
        (0.50, (1.00, 0.65, 0.00, 0.92)),
        (0.80, (1.00, 0.92, 0.20, 0.97)),
        (1.00, (1.00, 1.00, 0.80, 1.00)),
    ])

    # Blue — pace: dark navy → blue → royal blue → periwinkle → near-white blue
    cmap_speed_rgb = build_cmap("speed", [
        (0.00, (0.00, 0.10, 0.40, 1.00)),
        (0.35, (0.05, 0.30, 0.80, 1.00)),
        (0.65, (0.20, 0.55, 1.00, 1.00)),
        (0.85, (0.55, 0.75, 1.00, 1.00)),
        (1.00, (0.85, 0.92, 1.00, 1.00)),
    ])

    # Red — heart rate: visible dark red → #ea4747 → rose → near-white pink
    cmap_hr_rgb = build_cmap("hr", [
        (0.00, (0.40, 0.05, 0.05, 1.00)),
        (0.35, (0.70, 0.12, 0.12, 1.00)),
        (0.65, (0.92, 0.28, 0.28, 1.00)),
        (0.85, (1.00, 0.65, 0.65, 1.00)),
        (1.00, (1.00, 0.90, 0.90, 1.00)),
    ])

    # Diverging — gradient (change): green (descending) → dark neutral → purple (ascending)
    cmap_elev_rgb = build_cmap("elev", [
        (0.00, (0.12, 0.80, 0.22, 1.00)),   # strong descent — vivid green
        (0.25, (0.06, 0.52, 0.16, 1.00)),   # moderate descent
        (0.45, (0.06, 0.20, 0.10, 1.00)),   # slight descent — dark green
        (0.50, (0.18, 0.18, 0.18, 1.00)),   # flat — near-black neutral
        (0.55, (0.22, 0.08, 0.30, 1.00)),   # slight ascent — dark purple
        (0.75, (0.52, 0.06, 0.75, 1.00)),   # moderate ascent
        (1.00, (0.82, 0.22, 1.00, 1.00)),   # strong ascent — bright purple
    ])

    print("Colormaps built.")
    return cmap_count, cmap_speed_rgb, cmap_hr_rgb, cmap_elev_rgb


# ============================================================================
# Step 6 — Render layers and build the interactive Folium map
# ============================================================================

def _rgba_bytes(rgba_u8):
    buf = BytesIO()
    Image.fromarray(rgba_u8, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _to_uri(rgba_u8):
    return "data:image/png;base64," + base64.b64encode(_rgba_bytes(rgba_u8)).decode()


def render_layers(norm, cmaps, count_grid):
    """Render each layer to an RGBA uint8 array. Returns list of (name, rgba, visible)."""
    cmap_count, cmap_speed_rgb, cmap_hr_rgb, cmap_elev_rgb = cmaps

    def count_rgba(n):
        return (cmap_count(n) * 255).clip(0, 255).astype(np.uint8)

    def rgba_layer(rgb_norm, alpha_norm, cmap_rgb):
        arr = cmap_rgb(rgb_norm).copy()
        arr[:, :, 3] = alpha_norm
        return (arr * 255).clip(0, 255).astype(np.uint8)

    def white_layer(alpha_norm):
        h, w = alpha_norm.shape
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[:, :, :3] = 255
        arr[:, :, 3] = (alpha_norm * 255).clip(0, 255).astype(np.uint8)
        return arr

    print("Rendering layers…")
    layers = [
        ("Frequency (linear)",   count_rgba(norm["count_norm"]),                                            True),
        ("Frequency (log)",      count_rgba(norm["count_log_norm"]),                                        False),
        ("Pace (average)",       rgba_layer(norm["speed_norm"], norm["alpha_speed"], cmap_speed_rgb),       False),
        ("Heart rate (average)", rgba_layer(norm["hr_norm"], norm["alpha_hr"], cmap_hr_rgb),                False),
        ("Gradient (absolute)",  white_layer(norm["alpha_grad"]),                                           False),
        ("Gradient (change)",    rgba_layer((norm["elev_norm"] + 1) / 2, norm["alpha_elev"], cmap_elev_rgb), False),
    ]
    print(f"Done — {len(layers)} layers rendered.")
    return layers


def save_layer_images(layers, layers_dir):
    """Write each layer as a standalone PNG into layers_dir."""
    Path(layers_dir).mkdir(parents=True, exist_ok=True)
    for name, rgba, _ in layers:
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = Path(layers_dir) / f"{slug}.png"
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
        print(f"  Wrote {path}")


def cmap_to_css(cmap, n=14):
    stops = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, a = cmap(t)
        stops.append(f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})")
    return f"linear-gradient(to right, {', '.join(stops)})"


def pace_str(ms):
    secs = 1000 / ms
    return f"{int(secs // 60)}:{int(secs % 60):02d}/km"


def legend_row(row_id, title, grad_css, label_lo, label_hi, visible=False):
    display = "block" if visible else "none"
    return f"""
    <div id="{row_id}" style="display:{display}">
      <div style="font-weight:600;margin-bottom:3px;color:#eee">{title}</div>
      <div style="height:10px;border-radius:3px;background:{grad_css};
                  border:1px solid rgba(255,255,255,0.08)"></div>
      <div style="display:flex;justify-content:space-between;
                  margin-top:3px;color:#aaa;font-size:11px">
        <span>{label_lo}</span><span>{label_hi}</span>
      </div>
    </div>"""


def build_map(tracks, geo, layers, norm, cmaps, count_grid):
    cmap_count, cmap_speed_rgb, cmap_hr_rgb, cmap_elev_rgb = cmaps
    bounds = geo["bounds"]
    centre = geo["centre"]

    freq_css = cmap_to_css(cmap_count)
    pace_css = cmap_to_css(cmap_speed_rgb)
    hr_css   = cmap_to_css(cmap_hr_rgb)

    legend_html = f"""
<div id="heatmap-legend" style="
    position:fixed; bottom:28px; right:10px; z-index:9999;
    background:rgba(15,15,15,0.88);
    padding:13px 16px 14px; border-radius:9px;
    color:#ddd; font-family:sans-serif; font-size:12px;
    min-width:210px; line-height:1.4;
    border:1px solid rgba(255,255,255,0.10);
    box-shadow:0 2px 8px rgba(0,0,0,0.6);
">
  {legend_row("legend-frequency",      "Frequency (linear)",   freq_css, "1 pass", f"{int(count_grid.max())} passes", visible=True)}
  {legend_row("legend-frequency-log",  "Frequency (log)",      freq_css, "1 pass", f"{int(count_grid.max())} passes (log scale)")}
  {legend_row("legend-pace-avg",       "Pace (average)",       pace_css, pace_str(norm["s_lo"]), pace_str(norm["s_hi"]))}
  {legend_row("legend-heart-rate-avg", "Heart rate (average)", hr_css,   f"{norm['hr_lo']:.0f} bpm", f"{norm['hr_hi']:.0f} bpm")}
  {legend_row("legend-gradient",       "Gradient (absolute)",
      "linear-gradient(to right, rgba(0,0,0,0), rgba(255,255,255,1))",
      f"{norm['g_lo']*100:.1f}%", f"{norm['g_hi']*100:.1f}% grade")}
  {legend_row("legend-elev-change",    "Gradient (change)",
      cmap_to_css(cmap_elev_rgb), "descending", "ascending")}
</div>
"""

    layer_control_css = """
<style>
  .leaflet-control-layers {
    background: rgba(15,15,15,0.88) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 9px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.6) !important;
    color: #ddd !important;
    font-family: sans-serif !important;
    font-size: 12px !important;
  }
  .leaflet-control-layers-expanded { padding: 11px 14px 13px !important; }
  .leaflet-control-layers label {
    color: #eee !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    margin: 4px 0 !important;
  }
  .leaflet-control-layers-separator {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 6px 0 !important;
  }
  .leaflet-control-layers-toggle {
    background-color: rgba(15,15,15,0.88) !important;
    border-radius: 9px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
  }
</style>
"""

    exclusive_js = """
<script>
(function() {
    var exclusiveNames = [
        "Frequency (linear)", "Frequency (log)",
        "Pace (average)", "Heart rate (average)",
        "Gradient (absolute)", "Gradient (change)"
    ];
    var legendIds = {
        "Frequency (linear)":   "legend-frequency",
        "Frequency (log)":      "legend-frequency-log",
        "Pace (average)":       "legend-pace-avg",
        "Heart rate (average)": "legend-heart-rate-avg",
        "Gradient (absolute)":  "legend-gradient",
        "Gradient (change)":    "legend-elev-change"
    };
    function showLegend(activeName) {
        Object.keys(legendIds).forEach(function(name) {
            var el = document.getElementById(legendIds[name]);
            if (el) el.style.display = (name === activeName) ? "block" : "none";
        });
    }
    function setup() {
        var mapObj = null, overlays = null;
        for (var k in window) {
            try {
                if (!mapObj   && window[k] instanceof L.Map) mapObj = window[k];
                if (!overlays && window[k] && window[k].overlays && window[k].base_layers)
                    overlays = window[k].overlays;
            } catch(e) {}
        }
        if (!mapObj || !overlays) { setTimeout(setup, 100); return; }
        mapObj.on('overlayadd', function(e) {
            if (!exclusiveNames.includes(e.name)) return;
            exclusiveNames.forEach(function(name) {
                if (name !== e.name && overlays[name] && mapObj.hasLayer(overlays[name]))
                    mapObj.removeLayer(overlays[name]);
            });
            showLegend(e.name);
        });
    }
    document.addEventListener('DOMContentLoaded', setup);
})();
</script>
"""

    m = folium.Map(location=centre, zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer(
        "CartoDB.DarkMatterNoLabels",
        name="Basemap",
        control=False,
        show=True,
    ).add_to(m)

    track_group = folium.FeatureGroup(name="Raw GPS tracks", show=False)
    for label, pts in tracks:
        folium.PolyLine(
            locations=[(p[0], p[1]) for p in pts],
            color="#fc4c02", weight=1, opacity=0.4, tooltip=label,
        ).add_to(track_group)
    track_group.add_to(m)

    for name, rgba, visible in layers:
        fg = folium.FeatureGroup(name=name, show=visible)
        folium.raster_layers.ImageOverlay(
            image=_to_uri(rgba), bounds=bounds,
            opacity=MAP_OPACITY, interactive=False,
            cross_origin=False, zindex=1,
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(layer_control_css))
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(exclusive_js))
    return m


# ============================================================================
# Main
# ============================================================================

def main():
    Path("cache").mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"Source:  {ACTIVITIES_DIR}/")
    print(f"Types:   {', '.join(ACTIVITY_TYPES)}")
    print(f"Output:  {OUTPUT_HTML}\n")

    runs, home_lat, home_lon = filter_activities()
    tracks = load_tracks(runs)
    grids = rasterise(tracks, home_lat, home_lon)
    norm = blur_normalise(grids)
    cmaps = build_colormaps()
    layers = render_layers(norm, cmaps, grids["count_grid"])

    save_layer_images(layers, LAYERS_DIR)

    m = build_map(tracks, grids, layers, norm, cmaps, grids["count_grid"])
    m.save(OUTPUT_HTML)
    print(f"\nSaved: {OUTPUT_HTML}")
    print(f"Layer images: {LAYERS_DIR}/")
    print(f"Open:  file://{os.path.abspath(OUTPUT_HTML)}")


if __name__ == "__main__":
    main()
