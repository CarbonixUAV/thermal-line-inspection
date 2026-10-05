"""
Thermal hotspot finder for DJI H30T radiometric R-JPEGs (_T.JPG) along transmission lines.

Reads per-pixel temperatures with the DJI Thermal SDK (libdirp), projects the line's
towers and spans (from a GIS KMZ) into each photo using the drone's RTK position and
gimbal angles, finds warm spots inside that corridor, and uses parallax against the
neighbouring photos to drop the ones that are on the ground or moving (cattle, warm grass).

Writes to the output folder:
  images.csv     - per-image stats (temperatures, GPS, time, camera settings)
  hotspots.csv   - one row per hotspot, with priority, parallax class and nearest span/tower
  run_args.json  - the settings used (the review app reads these)

Review the results with:  python review_server.py <output folder> --photos <imagery folder>

Usage:
  python thermal_hotspots.py <folder of _T.JPG> [more folders] --network <towers.kmz> --out <folder>
"""
import argparse
import csv
import ctypes as C
import datetime as dt
import json
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

# DJI Thermal SDK binaries (not redistributable, download from DJI - see README)
SDK_BIN = Path(os.environ.get("DJI_TSDK_BIN",
                              Path(__file__).resolve().parent / "dji_thermal_sdk" / "utility" / "bin" / "windows" / "release_x64"))
FOCAL_PX = 2000.0  # H30T thermal: 24 mm lens / 12 um pixel pitch


# ---------------------------------------------------------------- SDK access
class _Res(C.Structure):
    _fields_ = [("width", C.c_int32), ("height", C.c_int32)]


class _MeasParams(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("distance", "humidity", "emissivity", "reflection", "ambient_temp")]


_lib = None
_NET = None


def _init_worker(net):
    global _NET
    _NET = net
    _init_sdk()


def _init_sdk():
    """Load libdirp.dll once per worker process."""
    global _lib
    os.add_dll_directory(str(SDK_BIN))
    os.chdir(SDK_BIN)  # SDK looks for libv_list.ini / plugin DLLs next to itself
    _lib = C.CDLL(str(SDK_BIN / "libdirp.dll"))
    _lib.dirp_set_verbose_level(0)


def read_temperatures(path):
    """Return (temps float32 HxW in degC, measurement params dict)."""
    data = Path(path).read_bytes()
    h = C.c_void_p()
    rc = _lib.dirp_create_from_rjpeg(data, len(data), C.byref(h))
    if rc != 0:
        raise RuntimeError(f"dirp_create_from_rjpeg failed ({rc})")
    try:
        res = _Res()
        _lib.dirp_get_rjpeg_resolution(h, C.byref(res))
        mp = _MeasParams()
        _lib.dirp_get_measurement_params(h, C.byref(mp))
        temps = np.empty((res.height, res.width), np.float32)
        rc = _lib.dirp_measure_ex(h, temps.ctypes.data_as(C.POINTER(C.c_float)), temps.nbytes)
        if rc != 0:
            raise RuntimeError(f"dirp_measure_ex failed ({rc})")
    finally:
        _lib.dirp_destroy(h)
    return temps, {f: round(getattr(mp, f), 2) for f, _ in _MeasParams._fields_}


# ------------------------------------------------------------------ metadata
_XMP_KEYS = ["GpsLatitude", "GpsLongitude", "RelativeAltitude", "AbsoluteAltitude",
             "LRFTargetLat", "LRFTargetLon", "LRFTargetDistance", "GimbalPitchDegree",
             "GimbalYawDegree", "GimbalRollDegree", "UTCAtExposure"]


def read_xmp(path):
    head = Path(path).open("rb").read(200_000)
    out = {}
    for k in _XMP_KEYS:
        m = re.search(rb'drone-dji:' + k.encode() + rb'="([^"]*)"', head)
        out[k] = m.group(1).decode() if m else ""
    return out


# ------------------------------------------------------------- GIS network
def load_network(kmz, bbox, max_span_m=1000):
    """
    Towers inside bbox (lat_min, lat_max, lon_min, lon_max) from a tower-structure
    KMZ export (fields FL_TOWER_NO, SUPPORTING_CIRCUIT_ID, HEIGHT = ground elevation in m). Spans join
    consecutive tower numbers on the same circuit.
    """
    kml = zipfile.ZipFile(kmz).read("doc.kml").decode("utf-8", "ignore")
    towers = []
    for pm in re.finditer(r"<Placemark.*?</Placemark>", kml, re.S):
        pm = pm.group(0)
        c = re.search(r"<coordinates>\s*([-\d.]+),([-\d.]+)", pm)
        if not c:
            continue
        lon, lat = float(c[1]), float(c[2])
        if not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
            continue
        f = dict(re.findall(r"<td>([A-Z_0-9]+)</td>\s*<td>([^<]*)</td>", pm))
        num = re.search(r"\d+", f.get("FL_TOWER_NO", ""))
        try:
            ground = float(f.get("HEIGHT", ""))
        except ValueError:
            continue
        towers.append({"name": f.get("FL_TOWER_NO", "?"), "num": int(num[0]) if num else -1,
                       "circuit": f.get("SUPPORTING_CIRCUIT_ID", ""), "lat": lat, "lon": lon, "ground": ground})
    spans = []
    by_circuit = {}
    for i, t in enumerate(towers):
        by_circuit.setdefault(t["circuit"], []).append(i)
    for idx in by_circuit.values():
        idx.sort(key=lambda i: towers[i]["num"])
        for a, b in zip(idx, idx[1:]):
            A, B = towers[a], towers[b]
            d = np.hypot((A["lat"] - B["lat"]) * 111320, (A["lon"] - B["lon"]) * 111320 * np.cos(np.radians(A["lat"])))
            if abs(A["num"] - B["num"]) <= 2 and d <= max_span_m:
                spans.append((a, b))
    return {"towers": towers, "spans": spans}


def _enu(lat, lon, alt, lat0, lon0, alt0):
    R = 6378137.0
    e = np.radians(np.asarray(lon) - lon0) * R * np.cos(np.radians(lat0))
    n = np.radians(np.asarray(lat) - lat0) * R
    return np.stack([e, n, np.asarray(alt, float) - alt0], -1)


def _camera(yaw, pitch, roll):
    """Unit vectors (forward, right, up) in ENU for DJI gimbal angles (deg)."""
    y, p, r = np.radians([yaw, pitch, roll])
    fwd = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(y), -np.sin(y), 0.0])
    up = np.cross(right, fwd)
    return fwd, right * np.cos(r) + up * np.sin(r), up * np.cos(r) - right * np.sin(r)


def _project(P, cam, shape):
    fwd, right, up = cam
    z = P @ fwd
    zs = np.maximum(z, 1e-3)
    u = shape[1] / 2 + FOCAL_PX * (P @ right) / zs
    v = shape[0] / 2 - FOCAL_PX * (P @ up) / zs
    return np.stack([u, v], -1), z


def gis_corridor(shape, meta, net, args, n=40):
    """
    Project spans and towers into the photo. Returns (conductor corridor mask,
    tower corridor mask, list of (label, polyline Nx2) for naming hotspots).
    The corridor is widened by args.corridor_deg to absorb pose/datum error.
    """
    lat0, lon0 = float(meta["GpsLatitude"]), float(meta["GpsLongitude"])
    alt0 = float(meta["AbsoluteAltitude"])
    cam = _camera(float(meta["GimbalYawDegree"]), float(meta["GimbalPitchDegree"]),
                  float(meta.get("GimbalRollDegree") or 0))
    towers = net["towers"]
    base = _enu([t["lat"] for t in towers], [t["lon"] for t in towers], [t["ground"] for t in towers],
                lat0, lon0, alt0)
    dist = np.hypot(base[:, 0], base[:, 1])
    max_dist = args.max_range
    margin = FOCAL_PX * np.tan(np.radians(args.corridor_deg))
    H, W = shape
    cond = np.zeros(shape, np.uint8)
    tower = np.zeros(shape, np.uint8)
    depth = np.zeros(shape, np.float32)   # distance to the asset (m), for sizing hotspots
    labels = []

    def visible(uv, z):
        return (z > 5) & (uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H)

    attach_h = args.tower_height * 0.7   # middle of the crossarm stack
    for a, b in net["spans"]:
        if min(dist[a], dist[b]) > max_dist + 600:   # no part of this span can be in range
            continue
        s = np.linspace(0, 1, n)[:, None]
        # Conductor sag: parabolic approximation of the catenary, lowest at mid-span.
        span_len = np.hypot(*(base[b] - base[a])[:2])
        sag = args.sag_ratio * span_len * 4 * s * (1 - s)
        pts = base[a] + s * (base[b] - base[a])
        pts[:, 2:] += attach_h - sag
        uv, z = _project(pts, cam, shape)
        ok = visible(uv, z) & (np.linalg.norm(pts, axis=1) <= max_dist)
        if ok.sum() < 2:
            continue
        for k in range(n - 1):
            if ok[k] and ok[k + 1]:
                extent = FOCAL_PX * 15 / max(min(z[k], z[k + 1]), 5)   # +/-15 m covers phases, earth wire, sag
                thick = max(int(min(2 * (margin + extent), 800)), 3)
                p, q = tuple(uv[k].astype(int)), tuple(uv[k + 1].astype(int))
                cv2.line(cond, p, q, 1, thick)
                cv2.line(depth, p, q, float((z[k] + z[k + 1]) / 2), thick)
        labels.append((f"{towers[a]['name']}-{towers[b]['name']}", uv[ok]))
    for i, t in enumerate(towers):
        if dist[i] > max_dist:
            continue
        pts = np.stack([base[i], base[i] + [0, 0, args.tower_height]])
        uv, z = _project(pts, cam, shape)
        if not visible(uv, z).all():
            continue
        half = FOCAL_PX * 8 / max(z.min(), 5)   # lattice + crossarms ~ +/-8 m
        thick = int(min(2 * (margin + half), 800))
        cv2.line(tower, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), 1, thick)
        cv2.line(depth, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), float(z.mean()), thick)
        labels.append((t["name"], uv))
    return cond.astype(bool), tower.astype(bool), labels, depth


def nearest_asset(labels, x, y):
    best, name = 1e18, ""
    for lab, poly in labels:
        if len(poly) == 2 and "-" not in lab:   # tower: distance to vertical axis segment
            p, q = poly
            t = np.clip(np.dot([x, y] - p, q - p) / max(np.dot(q - p, q - p), 1e-6), 0, 1)
            d = np.hypot(*([x, y] - (p + t * (q - p))))
            d *= 0.5  # prefer the tower when close to both
        else:
            d = np.min(np.hypot(poly[:, 0] - x, poly[:, 1] - y))
        if d < best:
            best, name = d, lab
    return name


# ------------------------------------------------------- line / tower finding
def _ridges(temps, sigma):
    """Hessian ridge strength at one scale: (dark thin lines, bright thin lines)."""
    g = cv2.GaussianBlur(temps, (0, 0), sigma)
    xx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    root = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
    return (xx + yy + root) / 2 * sigma ** 2, -(xx + yy - root) / 2 * sigma ** 2


# Tuned against a hand-annotated reference photo from the first survey.
# max_angle: away from towers a conductor must run within this many degrees of the projected span
# (rejects treelines / roads crossing the corridor); within free_near_tower_px of a GIS tower's projected
# position the conductors fan out steeply to the crossarms, so a looser max_angle_near_tower applies there.
CONDUCTOR = {"length": 45, "lo": 2.3, "hi": 4.0, "min_len": 100, "max_thick": 4.0, "max_angle": 30,
             "free_near_tower_px": 120, "max_angle_near_tower": 60}


_KERNELS = {}


def _line_kernels(length, n):
    """n straight-line averaging kernels, one per orientation (0..180 deg)."""
    if (length, n) not in _KERNELS:
        ks, c = [], length // 2
        for i in range(n):
            a = np.pi * i / n
            k = np.zeros((length, length), np.float32)
            dx, dy = np.cos(a) * c, np.sin(a) * c
            cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
            ks.append((np.degrees(a), k / k.sum()))
        _KERNELS[(length, n)] = ks
    return _KERNELS[(length, n)]


def _thin_ridges(temps):
    """(dark, bright) thin-line strength at 1-2 px scale."""
    d1, b1 = _ridges(temps, 1.0)
    d2, b2 = _ridges(temps, 1.6)
    return np.maximum(d1, d2).clip(0), np.maximum(b1, b2).clip(0)


def _normalise(r, win=61):
    """Divide by local texture strength so a line over busy grass still stands out."""
    return r / (cv2.blur(r, (win, win)) + 0.05)


def line_response(r, length=45, n=18):
    """Oriented line filter: best average ridge strength along a straight run -> (response, angle deg)."""
    z = _normalise(r)
    best = np.zeros_like(z)
    ang = np.zeros_like(z)
    for a, k in _line_kernels(length, n):
        f = cv2.filter2D(z, -1, k)
        m = f > best
        best[m], ang[m] = f[m], a
    return best, ang


def conductor_mask(temps, cond_corr, spans=(), sky=None, P=CONDUCTOR, near_tower=None):
    """
    Conductors as curves, so sag is followed naturally. A sagging conductor is
    locally straight, so an oriented line filter builds up a strong response
    along it while grass texture averages out; hysteresis thresholding then
    follows the curve. Conductors are cold (dark) against the ground and warm
    (bright) against the sky. Each curve must be long, thin and run roughly
    along the projected span.
    """
    dark, bright = _thin_ridges(temps)
    r = np.where(sky, bright, dark) if sky is not None else dark
    resp, ang = line_response(r, P["length"])
    # hysteresis: keep weak-response regions that contain a strong response ...
    weak = ((resp >= P["lo"]) & cond_corr).astype(np.uint8)
    n0, lab0 = cv2.connectedComponents(weak, connectivity=8)
    keep = np.unique(lab0[(resp >= P["hi"]) & (weak > 0)])
    m = np.isin(lab0, keep[keep > 0])
    # ... then thin to the ridge centre and re-link small gaps
    m &= r > cv2.blur(r, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    has_strong = np.ones(n, bool)
    # projected span segments (for the direction check): midpoints and angles
    seg_mid, seg_ang = [], []
    for poly in (spans if P["max_angle"] is not None else ()):
        for a, b in zip(poly[:-1], poly[1:]):
            seg_mid.append((a + b) / 2)
            seg_ang.append(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180)
    seg_mid, seg_ang = np.array(seg_mid).reshape(-1, 2), np.array(seg_ang)
    out = np.zeros(temps.shape, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        extent = np.hypot(w, h)
        if not has_strong[i] or extent < P["min_len"] or area / extent > P["max_thick"]:
            continue
        comp = lab == i
        near = near_tower is not None and near_tower[comp].mean() > 0.3
        limit = P["max_angle_near_tower"] if near else P["max_angle"]
        if len(seg_ang):
            # compare with the direction of the nearest projected span segment, wherever it is
            cy, cx = np.argwhere(comp).mean(0)
            span_dir = seg_ang[np.argmin(np.hypot(seg_mid[:, 0] - cx, seg_mid[:, 1] - cy))]
            d = np.abs(ang[comp] - span_dir)
            if np.median(np.minimum(d, 180 - d)) > limit:
                continue
        out |= comp
    return out


def find_assets_in_corridor(temps, cond_corr, tower_corr, spans=(), towers=(), sky=None):
    """
    Conductor mask inside the GIS corridor (towers are not detected).
    towers: projected GIS tower axes [(base_uv, top_uv), ...], used only to relax the
    span-direction check where conductors fan out to the crossarms.
    Returns (conductor mask, empty tower mask).
    """
    near = np.zeros(temps.shape, np.uint8)
    r = CONDUCTOR["free_near_tower_px"]
    for base, top in towers:
        cv2.line(near, tuple(np.clip(base, -9999, 9999).astype(int)), tuple(np.clip(top, -9999, 9999).astype(int)), 1, 2 * r)
    cond = conductor_mask(temps, cond_corr, spans, sky, near_tower=near > 0)
    return cond, np.zeros(temps.shape, bool)


def sky_mask(temps, pitch_deg=None):
    """
    Pixels that are sky (clear or cloud), so they can be left out of colour
    ranges. Sky is colder than the ground at the bottom of the frame and is
    connected to the top edge. If the gimbal pitch is known, the horizon row is
    projected: above it, anything colder than the ground counts (so clouds are
    caught); well below it, nothing can be sky.
    """
    H, W = temps.shape
    small = cv2.resize(temps, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    ground = float(np.median(small[int(h * 0.6):]))   # bottom 40% of a down-looking shot is ground
    rows = np.arange(h, dtype=np.float32)[:, None]
    threshold = np.full((h, 1), ground - 6, np.float32)
    if pitch_deg is not None:
        horizon = (H / 2 - FOCAL_PX * np.tan(np.radians(-pitch_deg))) / 4   # row of the horizon (small image)
        threshold = np.where(rows < horizon - h * 0.02, ground - 2, threshold)
        threshold = np.where(rows > horizon + h * 0.08, -1e9, threshold)  # can't be sky well below the horizon
    cold = (small < threshold).astype(np.uint8)
    cold = cv2.morphologyEx(cold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cold)
    top = np.unique(lab[0][lab[0] > 0])
    sky = np.isin(lab, top) if len(top) else np.zeros_like(cold, bool)
    return cv2.resize(sky.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def display_range(temps, sky=None):
    """Automatic colour range from the non-sky pixels: (1st percentile, 99.9th percentile)."""
    v = temps[~sky] if sky is not None and (~sky).sum() > temps.size * 0.03 else temps.ravel()
    return float(np.percentile(v, 1)), float(np.percentile(v, 99.9))


# ----------------------------------------------------------------- detection
def find_hotspots(temps, zone, delta, min_temp, min_area, min_above_scene):
    """
    Warm blobs inside `zone`: hotter than the local background by >= delta degC (or above
    min_temp). Local background = median over a ~64 px neighbourhood (computed on a 4x
    downsampled image for speed), so small hot components (joints, clamps) stand out.
    """
    small = temps[::4, ::4]
    bg_small = ndimage.median_filter(small, size=17)
    bg = np.kron(bg_small, np.ones((4, 4), np.float32))[: temps.shape[0], : temps.shape[1]]
    rise = temps - bg
    mask = rise >= delta
    if min_temp is not None:
        mask |= temps >= min_temp
    mask &= (temps >= np.median(temps) + min_above_scene) & zone
    mask = ndimage.binary_opening(mask, iterations=1)  # drop single-pixel noise
    labels, _ = ndimage.label(mask)
    spots = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == i
        area = int(region.sum())
        if area < min_area:
            continue
        t = temps[sl][region]
        iy, ix = np.unravel_index(np.argmax(np.where(region, temps[sl], -np.inf)), region.shape)
        spots.append({
            "max_temp": float(t.max()), "mean_temp": float(t.mean()),
            "background": float(bg[sl][region].mean()), "delta_t": float(rise[sl][region].max()),
            "area_px": area, "x": int(sl[1].start + ix), "y": int(sl[0].start + iy),
            "bbox": (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), "note": "",
        })
    spots.sort(key=lambda s: s["delta_t"], reverse=True)
    return spots


# Priority bands used in the client's thermal reports (delta T, degC). PT 180 starts at the
# --delta detection threshold (default 3 degC).
SEVERITIES = [(80, "PT 1", (255, 0, 255)), (30, "PT 30", (255, 0, 0)),
              (10, "PT 90", (255, 140, 0)), (3, "PT 180", (255, 255, 0))]


def severity(delta_t):
    for lo, name, _ in SEVERITIES:
        if delta_t >= lo:
            return name
    return "Below PT 180"


# ------------------------------------------------------------------- worker
# ------------------------------------------------------------------ parallax
# Consecutive photos are ~0.75 s / 7-8 m apart. A neighbouring photo is warped so the ground lines up
# (one homography); dense optical flow then gives the remaining motion of the LOCAL ground around each
# hotspot (uneven terrain). A hotspot's shift relative to its local ground decides:
#   elevated  - shifts >= elevated_px along the parallax direction (conductors, towers, trees)
#   ground    - shifts <= ground_px (on the ground: cattle lying/standing, warm grass)
#   moving    - shifts clearly but across the parallax direction (a walking animal / vehicle)
#   far       - the ground behind it is more than far_m away (near the horizon): too far to test
#   unknown   - tested, but the shift is ambiguous, or too small for something off the ground at that range
#               (it must move >= k_elevated_min x the shift an object min_height_m up would show there)
#   untested  - parallax can't judge: no confident match (e.g. isolated close-ups), the drone was hovering,
#               or a mis-match
#   tiny      - untested and smaller than min_ground_area_px: too small to track
# If the ring around a hotspot itself moves after the warp it is not ground (e.g. a close-up where the
# structure fills the frame), so the raw shift is used instead.
# Tuned on reviewed runs (300 cows, 35 trees, 33 real hotspots): 0 real lost, ~2/3 of cows filtered.
PARALLAX = {"offsets": (-2, -1, 1, 2), "max_gap_s": 120, "min_inlier_frac": 0.5, "search_px": 70,
            "min_score": 0.8, "max_match_px": 50, "ground_ring_px": 1.0,
            "elevated_px": 2.0, "ground_px": 1.5, "align": 0.8, "moving_align": 0.6, "max_move_px": 25,
            "min_ground_area_px": 9, "far_m": 1200.0, "min_height_m": 8.0, "k_elevated_min": 0.4}
_pcache = {}      # path -> (temps, meta, contrast image, keypoint xy, descriptors)
_folders = {}
_sift = None
_dis = None


def _shot_time(path):
    m = re.search(r"DJI_(\d{14})_", Path(path).name)
    return dt.datetime.strptime(m[1], "%Y%m%d%H%M%S") if m else None


def _neighbours(path, P=PARALLAX):
    folder = str(Path(path).parent)
    if folder not in _folders:
        _folders[folder] = sorted(f for f in os.listdir(folder) if f.upper().endswith("_T.JPG"))
    files = _folders[folder]
    i = files.index(Path(path).name)
    t0 = _shot_time(path)
    out = []
    for o in P["offsets"]:
        if 0 <= i + o < len(files):
            q = str(Path(folder) / files[i + o])
            tq = _shot_time(q)
            if t0 and tq and abs((tq - t0).total_seconds()) <= P["max_gap_s"]:
                out.append((o, q))
    return out


def _pframe(path, temps=None, meta=None):
    """Temperatures, metadata, contrast-normalised image and SIFT features of a photo (cached per worker)."""
    global _sift
    if path not in _pcache:
        if temps is None:
            temps, _ = read_temperatures(path)
        meta = meta or read_xmp(path)
        try:
            sky = sky_mask(temps, float(meta["GimbalPitchDegree"]))
        except (KeyError, ValueError):
            sky = sky_mask(temps)
        lo, hi = display_range(temps, sky)
        img8 = (np.clip((temps - lo) / max(hi - lo, 1e-3), 0, 1) * 255).astype(np.uint8)
        img8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img8)
        if _sift is None:
            _sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
        kp, desc = _sift.detectAndCompute(img8, (~sky).astype(np.uint8) * 255)
        _pcache[path] = (temps, meta, img8, np.float32([k.pt for k in kp]).reshape(-1, 2), desc)
        if len(_pcache) > 12:
            _pcache.pop(next(iter(_pcache)))
    return _pcache[path]


def _ground_homography(pa, pb):
    """Homography B -> A fitted to the dominant (ground) motion, and its inlier fraction."""
    ka, da = _pframe(pa)[3:]
    kb, db = _pframe(pb)[3:]
    if da is None or db is None or len(da) < 12 or len(db) < 12:
        return None, 0.0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 12:
        return None, 0.0
    H, inl = cv2.findHomography(kb[[g.queryIdx for g in good]], ka[[g.trainIdx for g in good]], cv2.RANSAC, 2.0)
    return H, (float(inl.sum()) / len(good)) if inl is not None else 0.0


def _pose(meta):
    lat0, lon0, alt0 = float(meta["GpsLatitude"]), float(meta["GpsLongitude"]), float(meta["AbsoluteAltitude"])
    cam = _camera(float(meta["GimbalYawDegree"]), float(meta["GimbalPitchDegree"]), float(meta.get("GimbalRollDegree") or 0))
    try:   # ground height from the laser rangefinder, in the drone's own altitude datum
        ground = float(meta["LRFTargetAbsAlt"]) if meta.get("LRFTargetAbsAlt") else alt0 - float(meta["RelativeAltitude"])
    except (KeyError, ValueError):
        ground = None
    return lat0, lon0, alt0, cam, ground


def _parallax_dir(ma, mb, u, v, shape):
    """
    Unit image direction in which a nearer-than-ground point at pixel (u, v) shifts between A and B
    (from B's camera position relative to A). Works for forward, sideways and hovering motion.
    """
    lat0, lon0, alt0, (fwd, right, up), _ = _pose(ma)
    b = _enu([float(mb["GpsLatitude"])], [float(mb["GpsLongitude"])], [float(mb["AbsoluteAltitude"])], lat0, lon0, alt0)[0]
    bf, br, bu = b @ fwd, b @ right, b @ up
    d = np.array([(u - shape[1] / 2) * bf - FOCAL_PX * br, (v - shape[0] / 2) * bf + FOCAL_PX * bu])
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else None


def _ground_distance(meta, u, v, shape):
    """Distance (m) along the ray through (u, v) to the ground; inf at/above the horizon; None if unknown."""
    lat0, lon0, alt0, (fwd, right, up), ground = _pose(meta)
    if ground is None:
        return None
    d = fwd + (u - shape[1] / 2) / FOCAL_PX * right - (v - shape[0] / 2) / FOCAL_PX * up
    d /= np.linalg.norm(d)
    return (alt0 - ground) / -d[2] if -d[2] > 0.003 else float("inf")


def expected_shift(ma, mb, u, v, shape, height):
    """
    Pixels an object `height` m above the ground at pixel (u, v) would shift against the ground behind it
    between A and B. Reliable when the camera looks down at close range; small near the horizon.
    """
    lat0, lon0, alt0, (fwd, right, up), ground = _pose(ma)
    if ground is None:
        return None
    d = fwd + (u - shape[1] / 2) / FOCAL_PX * right - (v - shape[0] / 2) / FOCAL_PX * up
    d /= np.linalg.norm(d)
    down = -d[2]
    if down <= 0.003:
        return 0.0
    z_ground = (alt0 - ground) / down
    z_obj = max(z_ground - height / down, 5.0)
    b = _enu([float(mb["GpsLatitude"])], [float(mb["GpsLongitude"])], [float(mb["AbsoluteAltitude"])], lat0, lon0, alt0)[0]
    return float(FOCAL_PX * np.linalg.norm(b - np.dot(b, d) * d) * max(1 / z_obj - 1 / z_ground, 0))


def parallax_tests(path, temps, meta, spots, P=PARALLAX):
    """
    For each hotspot, one entry per usable neighbouring photo: raw shift and shift relative to the
    local ground (px), the local ground motion, match score, and alignment with the parallax direction.
    """
    global _dis
    a8 = _pframe(path, temps, meta)[2]
    h, w = temps.shape
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    results = [[] for _ in spots]
    for o, q in _neighbours(path, P):
        H, frac = _ground_homography(path, q)
        if H is None or frac < P["min_inlier_frac"]:
            continue
        tb, mb, b8 = _pframe(q)[:3]
        tbw = cv2.warpPerspective(tb, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=-999)
        flow = _dis.calc(a8, cv2.warpPerspective(b8, H, (w, h)), None)   # remaining scene motion after the warp
        for i, s in enumerate(spots):
            bx0, by0, bx1, by1 = s["bbox"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            x0, y0, x1, y1 = max(bx0 - 4, 0), max(by0 - 4, 0), min(bx1 + 4, w), min(by1 + 4, h)
            tmpl = np.clip(temps[y0:y1, x0:x1] - s["background"], 0, None).astype(np.float32)   # excess heat only
            if tmpl.max() <= 0 or tmpl.size < 9:
                continue
            r = P["search_px"]
            sx0, sy0, sx1, sy1 = max(x0 - r, 0), max(y0 - r, 0), min(x1 + r, w), min(y1 + r, h)
            win = np.clip(tbw[sy0:sy1, sx0:sx1] - s["background"], 0, None).astype(np.float32)
            if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
                continue
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(win, tmpl, cv2.TM_CCORR_NORMED))
            dx, dy = sx0 + loc[0] - x0, sy0 + loc[1] - y0
            # local ground motion: median flow in a ring around the hotspot
            gx0, gy0, gx1, gy1 = max(bx0 - 30, 0), max(by0 - 30, 0), min(bx1 + 30, w), min(by1 + 30, h)
            ring = np.ones((gy1 - gy0, gx1 - gx0), bool)
            ring[max(by0 - 8 - gy0, 0):max(by1 + 8 - gy0, 0), max(bx0 - 8 - gx0, 0):max(bx1 + 8 - gx0, 0)] = False
            f = flow[gy0:gy1, gx0:gx1][ring]
            gdx, gdy = (float(np.median(f[:, 0])), float(np.median(f[:, 1]))) if len(f) else (0.0, 0.0)
            rx, ry = dx - gdx, dy - gdy
            rel = float(np.hypot(rx, ry))
            pd = _parallax_dir(meta, mb, cx, cy, temps.shape)
            align = None
            if pd is not None and rel > 0.5:
                align = abs(rx * pd[0] + ry * pd[1]) / rel
            results[i].append({"offset": o, "shift": float(np.hypot(dx, dy)), "rel": rel, "ground_dx": gdx,
                               "ground_dy": gdy, "score": float(score), "align": align,
                               "expected": expected_shift(meta, mb, cx, cy, temps.shape, P["min_height_m"])})
    return results


def parallax_rel(t, P=PARALLAX):
    """Shift relative to local ground, unless the surrounding ring is itself moving (then it isn't ground)."""
    return t["rel"] if np.hypot(t["ground_dx"], t["ground_dy"]) <= P["ground_ring_px"] else t["shift"]


def parallax_class(tests, area_px, ground_dist=None, P=PARALLAX):
    """'elevated', 'ground', 'moving', 'far', 'unknown', 'untested' or 'tiny' (see the notes above PARALLAX)."""
    T = [t for t in tests if t["score"] >= P["min_score"] and t["shift"] <= P["max_match_px"]]
    small = area_px < P["min_ground_area_px"]
    if not T:
        return "tiny" if small else "untested"
    if ground_dist is not None and ground_dist > P["far_m"]:
        return "far"
    rel = float(np.median([parallax_rel(t, P) for t in T]))
    raw = float(np.median([t["shift"] for t in T]))
    al = [t["align"] for t in T if t["align"] is not None]
    align = float(np.median(al)) if al else None
    ex = [t["expected"] for t in T if t.get("expected") is not None]
    expected = float(np.median(ex)) if ex else 0.0
    if rel >= P["elevated_px"] and align is not None and align >= P["align"]:
        # something off the ground this close would move much more (catches slightly shifting cattle)
        return "elevated" if rel >= P["k_elevated_min"] * expected else ("tiny" if small else "unknown")
    if small:
        return "tiny"                      # too small to track reliably
    if rel >= P["elevated_px"] and align is None:
        return "untested"                  # drone didn't move (hover): no parallax direction to judge by
    if rel >= P["elevated_px"] and raw > P["max_move_px"]:
        return "untested"                  # big jump across the parallax direction: a mis-match
    if rel >= P["elevated_px"] and align < P["moving_align"]:
        return "moving"
    if rel <= P["ground_px"]:
        return "ground"
    return "unknown"                       # tested, but the shift is ambiguous


def process(job):
    path, group, args = job
    try:
        temps, mp = read_temperatures(path)
        meta = read_xmp(path)
        try:
            pitch = float(meta["GimbalPitchDegree"])
        except (KeyError, ValueError):
            pitch = None
        # every warm blob in the GIS line corridor (or whole image) outside the sky;
        # parallax then decides what is on the ground
        zone = ~sky_mask(temps, pitch)
        corridor, labels, depth = None, [], None
        if _NET is not None:   # optional: corridor, asset names and hotspot size
            cond_corr, tower_corr, labels, depth = gis_corridor(temps.shape, meta, _NET, args)
            corridor = cond_corr | tower_corr
            if args.zone == "corridor":
                zone &= corridor
            labels = gis_corridor(temps.shape, meta, _NET, argparse.Namespace(**{**vars(args), "max_range": 1500}))[2]
        spots = find_hotspots(temps, zone, args.delta, args.min_temp, args.min_area, args.min_above_scene) \
            if zone.any() else []
        for s, tests in zip(spots, parallax_tests(path, temps, meta, spots) if spots else []):
            cx, cy = (s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2
            s["parallax"] = parallax_class(tests, s["area_px"], _ground_distance(meta, cx, cy, temps.shape))
            confident = [parallax_rel(t) for t in tests if t["score"] >= PARALLAX["min_score"]
                         and t["shift"] <= PARALLAX["max_match_px"]]
            s["parallax_shift"] = round(float(np.median(confident)), 1) if confident else ""
            if s["parallax"] == "ground":
                s["note"] = "on the ground (no parallax against neighbouring photos)"
            elif s["parallax"] == "moving":
                s["note"] = "moving object (shifted against the parallax direction, likely animal/vehicle)"
        for s in spots:
            s["range_m"] = s["size_m2"] = None
            x0, y0, x1, y1 = s["bbox"]
            d = depth[y0:y1, x0:x1] if depth is not None else np.zeros(0)
            if (d > 0).any():
                s["range_m"] = float(np.median(d[d > 0]))
                s["size_m2"] = s["area_px"] * (s["range_m"] / FOCAL_PX) ** 2
                if not s["note"] and s["size_m2"] > args.max_hot_size:
                    s["note"] = f"large warm object ({s['size_m2']:.1f} m2 at ~{s['range_m']:.0f} m, likely animal/vehicle)"
        real = [s for s in spots if not s["note"]]
        img_row = {
            "group": group, "file": Path(path).name, "path": path,
            "min_temp": round(float(temps.min()), 2),
            "median_temp": round(float(np.median(temps)), 2),
            "max_temp": round(float(temps.max()), 2),
            "line_in_view": bool(corridor.any()) if corridor is not None else "",
            "n_hotspots": len(real), "n_excluded": len(spots) - len(real),
            "worst_delta_t": round(real[0]["delta_t"], 2) if real else "",
            "worst_severity": severity(real[0]["delta_t"]) if real else "",
            **meta, **mp, "error": "",
        }
        spot_rows = [{
            "group": group, "file": Path(path).name, "rank": n,
            "severity": "Excluded" if s["note"] else severity(s["delta_t"]), "note": s["note"],
            "asset": nearest_asset(labels, s["x"], s["y"]) if labels else "",
            "max_temp": round(s["max_temp"], 2), "mean_temp": round(s["mean_temp"], 2),
            "background": round(s["background"], 2), "delta_t": round(s["delta_t"], 2),
            "area_px": s["area_px"], "x": s["x"], "y": s["y"],
            "x0": s["bbox"][0], "y0": s["bbox"][1], "x1": s["bbox"][2], "y1": s["bbox"][3],
            "range_m": round(s["range_m"], 1) if s["range_m"] else "",
            "size_m2": round(s["size_m2"], 3) if s["size_m2"] else "",
            "parallax": s.get("parallax", ""), "parallax_shift_px": s.get("parallax_shift", ""),
            "drone_lat": meta["GpsLatitude"], "drone_lon": meta["GpsLongitude"],
            "target_lat": meta["LRFTargetLat"], "target_lon": meta["LRFTargetLon"],
            "utc": meta["UTCAtExposure"],
        } for n, s in enumerate(spots, start=1)]
        return img_row, spot_rows
    except Exception as e:  # keep going on bad files
        return {"group": group, "file": Path(path).name, "path": path, "error": str(e)}, []


# ------------------------------------------------------------------- report
def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="folders of _T.JPG images")
    ap.add_argument("--network", help="optional tower-structure KMZ: limits the search to the line corridor, names the "
                                      "nearest span/tower and sizes hotspots. Without it the whole image is searched")
    ap.add_argument("--out", default=None, help="output folder (default results/<timestamp>)")
    ap.add_argument("--zone", choices=("corridor", "image"), default="corridor",
                    help="with --network: look for hotspots in the line corridor (default) or the whole image")
    ap.add_argument("--delta", type=float, default=3.0, help="degC above local background to flag (default 3)")
    ap.add_argument("--min-temp", type=float, default=None, help="also flag anything above this absolute degC")
    ap.add_argument("--min-above-scene", type=float, default=0.0,
                    help="hotspot must be this many degC above the image median (default 0)")
    ap.add_argument("--min-area", type=int, default=4, help="minimum hotspot size in pixels (default 4)")
    ap.add_argument("--tower-height", type=float, default=45.0, help="assumed tower height above ground (default 45 m)")
    ap.add_argument("--sag-ratio", type=float, default=0.035,
                    help="mid-span conductor sag as a fraction of span length (default 0.035 = 3.5%%)")
    ap.add_argument("--corridor-deg", type=float, default=3.0,
                    help="extra corridor width either side, in degrees, for GPS/gimbal/datum error (default 3)")
    ap.add_argument("--max-range", type=float, default=220.0,
                    help="only inspect the line within this distance of the drone (default 220 m)")
    ap.add_argument("--max-hot-size", type=float, default=0.6,
                    help="hotspots larger than this (m2 at the line's range) are treated as animals/vehicles (default 0.6)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images per folder (for testing)")
    ap.add_argument("--every", type=int, default=1, help="only process every Nth image (for testing)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    args.out = str(Path(args.out or Path("results") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")).resolve())
    args.network = str(Path(args.network).resolve()) if args.network else None
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "run_args.json").write_text(json.dumps(vars(args), indent=1))  # review app reuses these
    jobs = []
    for folder in args.folders:
        folder = Path(folder).resolve()
        files = sorted(p for p in folder.glob("*.JPG") if p.stem.upper().endswith("_T"))[:: args.every]
        jobs += [(str(p), folder.name, args) for p in files[: args.limit]]
    print(f"{len(jobs)} images -> {args.out} using {args.workers} workers")

    net = None
    if args.network:
        try:
            pos = [read_xmp(p) for p, _, _ in jobs[:: max(1, len(jobs) // 500)]]
            lats = [float(m["GpsLatitude"]) for m in pos if m["GpsLatitude"]]
            lons = [float(m["GpsLongitude"]) for m in pos if m["GpsLongitude"]]
            pad = 0.03  # ~3 km
            net = load_network(args.network, (min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad))
            if not net["spans"]:
                raise ValueError("no towers/spans in a recognised format near these flights")
            circuits = sorted({t["circuit"] for t in net["towers"]})
            print(f"GIS: {len(net['towers'])} structures, {len(net['spans'])} spans near the flights ({', '.join(circuits)})")
        except Exception as e:
            net = None
            print(f"WARNING: not using {args.network} ({e}); searching the whole image")
    if net is None:
        args.zone = "image"
        print("No GIS: whole image searched, no asset names or hotspot sizes")
    (Path(args.out) / "run_args.json").write_text(json.dumps(vars(args), indent=1))

    t0 = time.time()
    img_rows, spot_rows = [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(net,)) as pool:
        # chunks of consecutive photos, so each worker reuses its neighbours' features
        for i, (img, spots) in enumerate(pool.imap_unordered(process, jobs, chunksize=16), start=1):
            img_rows.append(img)
            spot_rows += spots
            if i % 100 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(jobs)}  {rate:.1f} img/s  ETA {(len(jobs) - i) / rate:.0f}s", flush=True)

    img_rows.sort(key=lambda r: (r["group"], r["file"]))
    spot_rows.sort(key=lambda r: -r["delta_t"])
    write_csv(Path(args.out) / "images.csv", img_rows)
    write_csv(Path(args.out) / "hotspots.csv", spot_rows)
    print(f"Done in {time.time() - t0:.0f}s: {sum(1 for r in img_rows if r.get('n_hotspots'))}/{len(img_rows)} images "
          f"with hotspots, {sum(1 for r in img_rows if r.get('error'))} errors -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
