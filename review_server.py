"""
Interactive review of thermal_hotspots.py results.

  python review_server.py results/<run> --photos <imagery folder>     (opens http://localhost:8765)

--photos is searched (recursively) for the zoom-camera _Z.JPG photos shown beside each thermal photo.

1. Review   - mark each auto-detected hotspot as real or a false positive
              (cows in the paddock, trees, vehicles...), or a whole photo at once.
2. Analyse  - on confirmed photos, draw spot / box / line measurements
              (max / mean / min) like DJI Thermal Analysis Tool, and fill in
              defect details (tower, component, phase, side, action, notes).
3. Report   - fill in the cover page and generate a PDF report.

Everything you do is saved to <run>/review.json as you go, so you can stop
and come back. Temperatures are read straight from the original R-JPEGs
with the DJI Thermal SDK, so measurements are true radiometric values.
"""
import argparse
import base64
import csv
import datetime as dt
import html
import io
import json
import os
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags

import thermal_hotspots as th

ROOT = Path(__file__).resolve().parent
APP_HTML = ROOT / "review_app.html"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]

# Priority bands (same as the example report / thermal_hotspots severities)
PRIORITIES = [(80, "PT 1", "Catastrophic", "Thermal defects >80°C"),
              (30, "PT 30", "Severe", "Thermal defects 30°C to 80°C"),
              (10, "PT 90", "Major", "Thermal defects 10°C to 30°C"),
              (0, "PT 180", "Minor", "Thermal defects < 10°C")]


# parallax results hidden behind the "show undecided" checkbox (tested-but-ambiguous, too far, too small)
UNDECIDED = ("unknown", "far", "tiny")

# corridor settings for the line overlay if the run didn't save run_args.json
LINE_DEFAULTS = {"corridor_deg": 3.0, "tower_height": 45.0, "sag_ratio": 0.035, "max_range": 220.0}


def priority(delta_t):
    for lo, pt, name, _ in PRIORITIES:
        if delta_t is not None and delta_t >= lo:
            return pt, name
    return "PT 180", "PT 180"


# ------------------------------------------------------------------ palettes
def _iron():
    stops = [(0, (0, 0, 0)), (.15, (32, 0, 110)), (.35, (140, 0, 160)), (.5, (200, 40, 110)),
             (.65, (235, 100, 30)), (.8, (250, 170, 0)), (.93, (255, 235, 90)), (1, (255, 255, 255))]
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, [s for s, _ in stops], [c[i] for _, c in stops]) for i in range(3)], 1)


PALETTES = {
    "iron": _iron().astype(np.uint8),
    "inferno": cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1],
    "rainbow": cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0, ::-1],
    "white hot": np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1),
    "black hot": np.repeat(np.arange(255, -1, -1, dtype=np.uint8)[:, None], 3, 1),
}


def colourise(temps, lo, hi, palette="iron"):
    idx = np.clip((temps - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    return PALETTES.get(palette, PALETTES["iron"])[idx]


# ------------------------------------------------------------------- stats
def region_values(temps, m):
    """Temperatures inside a measurement (point / box / line) in image pixel coords."""
    H, W = temps.shape
    p = m["pts"]
    if m["type"] == "point":
        x, y = int(round(p[0][0])), int(round(p[0][1]))
        return temps[min(max(y, 0), H - 1), min(max(x, 0), W - 1)][None]
    if m["type"] == "box":
        x0, x1 = sorted([int(round(p[0][0])), int(round(p[1][0]))])
        y0, y1 = sorted([int(round(p[0][1])), int(round(p[1][1]))])
        x0, y0 = max(x0, 0), max(y0, 0)
        return temps[y0:min(y1, H - 1) + 1, x0:min(x1, W - 1) + 1].ravel()
    (x0, y0), (x1, y1) = p[0], p[1]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.clip(np.round(np.linspace(x0, x1, n)).astype(int), 0, W - 1)
    ys = np.clip(np.round(np.linspace(y0, y1, n)).astype(int), 0, H - 1)
    return temps[ys, xs]


def stats(temps, m):
    v = region_values(temps, m)
    if not v.size:
        return None
    return {"max": float(v.max()), "mean": float(v.mean()), "min": float(v.min())}


# --------------------------------------------------------------------- data
class Run:
    def __init__(self, run_dir, photos=None):
        self.photos = Path(photos).resolve() if photos else None
        self.dir = Path(run_dir).resolve()
        self.cache = self.dir / "_review_cache"
        self.cache.mkdir(exist_ok=True)
        self.review_path = self.dir / "review.json"
        self.lock = threading.Lock()
        self.sdk_lock = threading.Lock()
        self.temps_cache, self.view_cache, self.lines_cache, self.asset_png = {}, {}, {}, {}
        ap = self.dir / "run_args.json"
        self.args = json.loads(ap.read_text()) if ap.exists() else {}
        self.images = self._load_images()
        self.net, self.net_status = None, "loading GIS network…"
        threading.Thread(target=self._load_network, daemon=True).start()
        self.zoom_index = self._index_zoom_photos()
        self.review = json.loads(self.review_path.read_text("utf-8")) if self.review_path.exists() else {}
        self.review.setdefault("images", {})
        self.review.setdefault("report", {})

    def _load_images(self):
        rows = {r["file"]: r for r in csv.DictReader(open(self.dir / "images.csv", encoding="utf-8"))}
        spots = {}
        hp = self.dir / "hotspots.csv"
        if hp.exists():
            for s in csv.DictReader(open(hp, encoding="utf-8")):
                spots.setdefault(s["file"], []).append(s)
        out = {}
        for f, r in rows.items():
            if f not in spots:
                continue
            hs = []
            for s in sorted(spots[f], key=lambda s: int(s["rank"])):
                def num(k, d=None):
                    try:
                        return float(s[k])
                    except (KeyError, ValueError, TypeError):
                        return d
                x, y = num("x", 0), num("y", 0)
                hs.append({
                    "id": s["rank"], "severity": s["severity"], "note": s.get("note", ""),
                    "max_temp": num("max_temp"), "background": num("background"), "delta_t": num("delta_t"),
                    "area_px": num("area_px"), "size_m2": num("size_m2"), "range_m": num("range_m"),
                    "asset": s.get("asset", ""), "x": x, "y": y,
                    "parallax": s.get("parallax", ""), "parallax_shift": num("parallax_shift_px"),
                    "bbox": [num("x0", x - 3), num("y0", y - 3), num("x1", x + 3), num("y1", y + 3)],
                })
            out[f] = {"file": f, "group": r["group"], "path": r["path"], "hotspots": hs,
                      "lat": r.get("GpsLatitude", ""), "lon": r.get("GpsLongitude", ""),
                      "target_lat": r.get("LRFTargetLat", ""), "target_lon": r.get("LRFTargetLon", ""),
                      "utc": r.get("UTCAtExposure", ""), "median_temp": r.get("median_temp"),
                      "emissivity": r.get("emissivity"), "distance": r.get("distance"),
                      "humidity": r.get("humidity"), "ambient": r.get("ambient_temp"),
                      "reflection": r.get("reflection"), "pitch": r.get("GimbalPitchDegree")}
        return out

    # ---- what the app is sent (big runs have hundreds of thousands of hotspots)
    def filtered(self, excluded=False, undecided=False):
        """
        Photos and hotspots to send to the app. Auto-excluded (ground / moving / animals) and
        undecided (parallax 'unknown' or 'far') hotspots are left out unless asked for;
        anything already reviewed is always sent. Returns (images, hidden counts).
        """
        hidden = {"excluded": 0, "undecided": 0}
        out = []
        for f, im in self.images.items():
            rev = self.review["images"].get(f, {}).get("hotspots", {})
            keep = []
            for h in im["hotspots"]:
                if rev.get(h["id"], {}).get("status"):
                    keep.append(h)
                elif h["note"] and not excluded:
                    hidden["excluded"] += 1
                elif not h["note"] and h.get("parallax") in UNDECIDED and not undecided:
                    hidden["undecided"] += 1
                else:
                    keep.append(h)
            if keep:
                out.append({**im, "hotspots": keep})
        return out, hidden

    # ---- sky / automatic colour range
    def view(self, f):
        """(temps, sky mask, auto_lo, auto_hi) with sky left out of the auto range."""
        with self.lock:
            if f in self.view_cache:
                return self.view_cache[f]
        t = self.temps(f)
        try:
            pitch = float(self.images[f].get("pitch") or th.read_xmp(self.images[f]["path"])["GimbalPitchDegree"])
        except (TypeError, ValueError):
            pitch = None
        sky = th.sky_mask(t, pitch)
        lo, hi = th.display_range(t, sky)
        with self.lock:
            if len(self.view_cache) > 40:
                self.view_cache.pop(next(iter(self.view_cache)))
            self.view_cache[f] = (t, sky, lo, hi)
        return t, sky, lo, hi

    # ---- GIS corridor / detected lines, for the optional overlay
    def _load_network(self):
        try:
            kmz = Path(self.args.get("network") or "")
            if not kmz.is_file():
                self.net_status = "no GIS network file"
                return
            lats = [float(i["lat"]) for i in self.images.values() if i["lat"]]
            lons = [float(i["lon"]) for i in self.images.values() if i["lon"]]
            pad = 0.03
            net = th.load_network(kmz, (min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad))
            self.net, self.net_status = net, "ok"
        except Exception as e:
            self.net_status = f"GIS load failed: {e}"

    def lines(self, f):
        with self.lock:
            if f in self.lines_cache:
                return self.lines_cache[f]
        if self.net is None:
            return {"ok": False, "status": self.net_status}
        import cv2
        from types import SimpleNamespace
        a = SimpleNamespace(**{**LINE_DEFAULTS, **{k: v for k, v in self.args.items() if k in LINE_DEFAULTS}})
        t, sky, _, _ = self.view(f)
        meta = th.read_xmp(self.images[f]["path"])
        cond_corr, tower_corr, labels, _ = th.gis_corridor(t.shape, meta, self.net, a)
        spans = [poly for lab, poly in labels if "-" in lab]
        towers = [poly for lab, poly in labels if "-" not in lab]
        cond, _ = th.find_assets_in_corridor(t, cond_corr, tower_corr, spans, towers, sky)
        # detected conductors (green) as an overlay image
        rgba = np.zeros(t.shape + (4,), np.uint8)
        rgba[cv2.dilate(cond.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 255, 110, 230)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG")
        contours, _ = cv2.findContours((cond_corr | tower_corr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clip = lambda p: np.clip(np.asarray(p, float), -5000, 5000).round(1).tolist()
        out = {"ok": True,
               "corridor": [cv2.approxPolyDP(c, 2, True).reshape(-1, 2).tolist() for c in contours],
               "gis": [{"label": lab, "span": "-" in lab, "pts": clip(poly)} for lab, poly in labels]}
        with self.lock:
            self.lines_cache[f] = out
            self.asset_png[f] = buf.getvalue()
            if len(self.asset_png) > 40:
                self.asset_png.pop(next(iter(self.asset_png)))
        return out

    def _index_zoom_photos(self):
        idx = {}
        if self.photos is None:
            return idx
        for dirpath, _, files in os.walk(self.photos):
            for f in files:
                m = re.match(r"DJI_(\d{14})_(\d+)_Z\.JPG$", f.upper())
                if m:
                    t = dt.datetime.strptime(m[1], "%Y%m%d%H%M%S")
                    idx.setdefault(m[2], []).append((t, str(Path(dirpath) / f)))
        return idx

    def zoom_path(self, thermal_file):
        """
        Zoom photo taken with this thermal photo. They share the sequence number,
        but the zoom file's timestamp is sometimes a second later, so match on
        sequence number + nearest time (sequence numbers restart each flight).
        """
        m = re.match(r"DJI_(\d{14})_(\d+)_T\.JPG$", thermal_file.upper())
        if not m or m[2] not in self.zoom_index:
            return None
        t = dt.datetime.strptime(m[1], "%Y%m%d%H%M%S")
        gap, path = min((abs((zt - t).total_seconds()), p) for zt, p in self.zoom_index[m[2]])
        return path if gap <= 5 else None

    def temps(self, f):
        with self.lock:
            if f in self.temps_cache:
                return self.temps_cache[f]
        with self.sdk_lock:
            t, _ = th.read_temperatures(self.images[f]["path"])
        with self.lock:
            if len(self.temps_cache) > 40:
                self.temps_cache.pop(next(iter(self.temps_cache)))
            self.temps_cache[f] = t
        return t

    def save_review(self, data):
        with self.lock:
            self.review = data
            tmp = self.review_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), "utf-8")
            os.replace(tmp, self.review_path)


# ------------------------------------------------------------ image helpers
def zoom_focal_px(path, width, height):
    """Focal length in pixels from the 35 mm-equivalent focal length in EXIF."""
    ex = Image.open(path).getexif().get_ifd(0x8769)
    tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
    f35 = float(tags.get("FocalLengthIn35mmFilm") or 48)
    return f35 * np.hypot(width, height) / 43.27


THERMAL_F35 = 52.0  # H30T thermal, from EXIF


def thermal_to_zoom(pt, zpath, zsize, tshape=(1024, 1280)):
    """Map a thermal pixel onto the zoom photo (cameras are co-aligned; parallax ignored)."""
    ft = THERMAL_F35 * np.hypot(tshape[1], tshape[0]) / 43.27
    fz = zoom_focal_px(zpath, *zsize)
    s = fz / ft
    return (zsize[0] / 2 + (pt[0] - tshape[1] / 2) * s, zsize[1] / 2 + (pt[1] - tshape[0] / 2) * s), s


def _font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_thermal(temps, measurements, lo, hi, palette, scale=1, with_bar=True):
    """Palette image with measurement annotations and a colour bar (for the PDF)."""
    rgb = colourise(temps, lo, hi, palette)
    img = Image.fromarray(rgb).resize((temps.shape[1] * scale, temps.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    # sized for the final page: the image is shown ~90 mm wide
    fs, lw = max(14, img.width // 34), max(2, img.width // 300)
    f = _font(fs)
    for m in measurements:
        p = [(x * scale, y * scale) for x, y in m["pts"]]
        col = (0, 200, 255) if m["type"] == "point" else (0, 255, 120) if m["type"] == "line" else (80, 200, 255)
        if m["type"] == "point":
            x, y = p[0]
            r = img.width // 14
            d.line([(x - r, y), (x + r, y)], fill=col, width=lw)
            d.line([(x, y - r), (x, y + r)], fill=col, width=lw)
            d.text((x + lw * 3, y - fs - lw * 2), m["label"], fill="white", font=f, stroke_width=2, stroke_fill="black")
        elif m["type"] == "box":
            (x0, y0), (x1, y1) = p
            pad = lw * 2
            d.rectangle([min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad], outline=col, width=lw)
            d.text((min(x0, x1) - pad, min(y0, y1) - pad - fs - 4), m["label"], fill="white", font=f, stroke_width=2, stroke_fill="black")
        else:
            d.line(p, fill=col, width=lw)
            d.text((p[0][0] + 4, p[0][1] - fs - 6), m["label"], fill="white", font=f, stroke_width=2, stroke_fill="black")
    if not with_bar:
        return img
    bw, pad = img.width // 22, img.width // 7
    out = Image.new("RGB", (img.width + bw + pad + 10 * scale, img.height), "white")
    out.paste(img, (0, 0))
    top = fs + 16
    bar = PALETTES.get(palette, PALETTES["iron"])[np.linspace(255, 0, img.height - 2 * top).astype(int)]
    bar = Image.fromarray(np.repeat(bar[:, None, :], bw, 1).astype(np.uint8))
    x0 = img.width + 10
    out.paste(bar, (x0, top))
    d = ImageDraw.Draw(out)
    d.rectangle([x0, top, x0 + bw, top + bar.height], outline="black", width=2)
    d.text((x0, 4), f"{hi:.1f} °C", fill="black", font=f)
    d.text((x0, top + bar.height + 6), f"{lo:.1f}", fill="black", font=f)
    for tval in np.linspace(lo, hi, 6)[1:-1]:
        y = top + bar.height * (1 - (tval - lo) / max(hi - lo, 1e-3))
        d.line([(x0 + bw, y), (x0 + bw + 10, y)], fill="black", width=2)
        d.text((x0 + bw + 14, y - fs / 2), f"{tval:.0f}", fill="black", font=f)
    return out


def to_data_uri(img, fmt="JPEG", quality=88):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- PDF report
def taken_at(file):
    m = re.search(r"DJI_(\d{14})_", file)   # DJI file names carry local time
    return dt.datetime.strptime(m[1], "%Y%m%d%H%M%S") if m else None


def build_report(run, out_pdf):
    rep = run.review.get("report", {})
    defects = []
    for f, st in run.review["images"].items():
        if f not in run.images:
            continue
        a = st.get("analysis") or {}
        confirmed = [h for h in run.images[f]["hotspots"] if (st.get("hotspots", {}).get(h["id"], {}).get("status") == "real")]
        if not confirmed or a.get("include") is False:
            continue
        defects.append((f, st, a, confirmed))
    defects.sort(key=lambda d: taken_at(d[0]) or dt.datetime.min)

    circuit = rep.get("inspection_of") or "Transmission line"
    esc = lambda s: html.escape(str(s if s is not None else ""))
    logo = ""
    for name in ("report_logo.png", "report_logo.jpg"):
        p = ROOT / name
        if p.exists():
            logo = f'<img class="logo" src="{to_data_uri(Image.open(p).convert("RGB"), "PNG")}">'
            break
    header = f'<div class="hdr"><div class="ttl">{esc(rep.get("header_title") or "Transmission Line")}<small>{esc(rep.get("header_subtitle") or "Non Invasive Inspection Report")}</small></div>{logo}</div>'
    footer = f'<div class="ftr">{esc(rep.get("footer") or "")}</div>'

    pages = []
    # ---- cover
    wx = rep.get("weather", [{}])
    wrows = "".join(
        f"<tr><td>{esc(w.get('date'))}</td><td>{esc(w.get('time'))}</td><td>{esc(w.get('load'))}</td><td>{esc(w.get('wind'))}</td>"
        f"<td>{esc(w.get('conditions'))}</td><td>{esc(w.get('temp'))}</td><td>{esc(w.get('dew'))}</td><td>{esc(w.get('humidity'))}</td>"
        f"<td>{esc(w.get('notes'))}</td></tr>" for w in wx) or "<tr>" + "<td>&nbsp;</td>" * 9 + "</tr>"
    pages.append(f"""<section class="page cover">{logo and f'<div class="coverlogo">{logo}</div>'}
<h3>Transmission Inspection Details</h3>
<table class="kv narrow">
<tr><td>Inspection of</td><td>{esc(circuit)}</td></tr>
<tr><td>Noti No.</td><td>{esc(rep.get('noti_no'))}</td></tr>
<tr><td>Order No.</td><td>{esc(rep.get('order_no'))}</td></tr>
<tr><td>Inspection completed by</td><td>{esc(rep.get('inspector'))}</td></tr>
<tr><td>Report Version</td><td>{esc(rep.get('report_version'))}</td></tr></table>
<h3>Survey Details</h3>
<table class="grid narrow"><tr><th>Inspection Type</th><th>Completed</th><th>No. of Faults</th></tr>
<tr><td>Thermal</td><td>Yes</td><td>{len(defects) or 'Nil Detected'}</td></tr>
<tr><td>Visual</td><td>{esc(rep.get('visual_completed') or 'Yes')}</td><td>{esc(rep.get('visual_faults') or 'Nil Detected')}</td></tr></table>
<h3>Weather Conditions</h3>
<table class="grid"><tr><th>Date</th><th>Start Time</th><th>Load</th><th>Wind Speed</th><th>Weather Conditions</th><th>Temp.</th><th>Dew point</th><th>Humidity</th><th>Notes</th></tr>{wrows}</table>
<table class="kv narrow notes"><tr><td>Notes:<div class="notesbox">{esc(rep.get('notes')).replace(chr(10), '<br>')}</div></td></tr></table>
</section>""")

    # ---- one page per defect
    for n, (f, st, a, confirmed) in enumerate(defects, start=1):
        info = run.images[f]
        temps = run.temps(f)
        meas = a.get("measurements") or [
            {"type": "box", "label": f"Ar{i}", "pts": [[h["bbox"][0] - 2, h["bbox"][1] - 2], [h["bbox"][2] + 2, h["bbox"][3] + 2]]}
            for i, h in enumerate(confirmed, 1)]
        for m in meas:
            m["stats"] = stats(temps, m)
        lo = a.get("range_lo"); hi = a.get("range_hi")
        hottest = max([m["stats"]["max"] for m in meas if m["stats"]] + [h["max_temp"] for h in confirmed])
        _, _, auto_lo, auto_hi = run.view(f)   # sky is left out of the automatic range
        if lo is None:
            lo = auto_lo
        if hi is None:
            hi = max(auto_hi + 2, hottest + 1, lo + 5)
        pal = a.get("palette") or "iron"
        thermal = render_thermal(temps, meas, float(lo), float(hi), pal, scale=1)

        # reference for delta T: first spot if placed, else the detector's local background
        areas = [m for m in meas if m["type"] in ("box", "line") and m["stats"]]
        spotm = [m for m in meas if m["type"] == "point" and m["stats"]]
        hot = max((m["stats"]["max"] for m in areas), default=max(h["max_temp"] for h in confirmed))
        if spotm:
            ref, ref_txt = spotm[0]["stats"]["max"], f"{spotm[0]['label']} spot"
        else:
            ref, ref_txt = min(h["background"] for h in confirmed if h["background"] is not None), "local background"
        delta = a.get("delta_override")
        delta = float(delta) if delta not in (None, "") else hot - ref
        pt, sev = priority(delta)
        pt = a.get("priority_override") or pt
        when = taken_at(f)
        report_no = f"{pt.replace('PT ', '')}_{when:%y%m%d}_{circuit.replace(' ', '')}_{n}" if when else f"{n}"

        zp = run.zoom_path(f)
        visual_html = area_html = '<div class="missing">No visual photo</div>'
        if zp:
            z = Image.open(zp).convert("RGB")
            centre_meas = areas[0] if areas else (meas[0] if meas else None)
            if centre_meas:
                cx = np.mean([p[0] for p in centre_meas["pts"]]); cy = np.mean([p[1] for p in centre_meas["pts"]])
            else:
                cx, cy = temps.shape[1] / 2, temps.shape[0] / 2
            (zx, zy), s = thermal_to_zoom((cx, cy), zp, z.size, temps.shape)
            area = z.copy()
            da = ImageDraw.Draw(area)
            if 0 <= zx < z.width and 0 <= zy < z.height:
                half = max(z.width / 8, 60 * s)
                box = (int(max(zx - half, 0)), int(max(zy - half * 0.75, 0)),
                       int(min(zx + half, z.width)), int(min(zy + half * 0.75, z.height)))
                visual_html = f'<img src="{to_data_uri(z.crop(box).resize((900, int(900 * (box[3] - box[1]) / max(box[2] - box[0], 1)))))}">'
                da.rectangle(box, outline=(0, 255, 120), width=max(6, z.width // 400))
            else:
                visual_html = '<div class="missing">Hotspot is outside the zoom camera frame</div>'
            area.thumbnail((1100, 1100))
            area_html = f'<img src="{to_data_uri(area)}">'

        rows = []
        for m in meas:
            if not m["stats"]:
                continue
            if m["type"] == "point":
                rows.append(f"<tr><td>{esc(m['label'])} Spotmeter Temperature</td><td>{m['stats']['max']:.1f} °C</td></tr>")
            else:
                kind = "Area" if m["type"] == "box" else "Line"
                rows.append(f"<tr><td>{esc(m['label'])} {kind} Max. / Mean / Min.</td><td>{m['stats']['max']:.1f} / "
                            f"{m['stats']['mean']:.1f} / {m['stats']['min']:.1f} °C</td></tr>")
        rows.append(f"<tr><td>&Delta;T (hottest vs {esc(ref_txt)})</td><td>{delta:.1f} °C &nbsp;({esc(pt)})</td></tr>")
        lat, lon = info["target_lat"] or info["lat"], info["target_lon"] or info["lon"]
        det = a.get("details", {})
        tower = det.get("tower") or confirmed[0]["asset"]
        pages.append(f"""<section class="page">{header}
<div class="band">Thermal Defect</div>
<table class="kv4"><tr><td class="lab">Report No.</td><td>{esc(report_no)}</td><td class="lab">Priority</td><td>{esc(pt)}</td></tr></table>
<table class="pics"><tr><td class="pic">{visual_html}</td><td class="pic thermal"><img src="{to_data_uri(thermal)}"></td></tr>
<tr><td class="cap">Visual Image</td><td class="cap">Thermal Image</td></tr>
<tr><td class="pic label">{esc(det.get('equipment_label') or tower)}</td><td class="pic">{area_html}</td></tr>
<tr><td class="cap">Equipment Label</td><td class="cap">Area Photo</td></tr></table>
<table class="kv"><tr><td>Date</td><td>{when:%d/%m/%Y}</td></tr><tr><td>Image Time</td><td>{when:%I:%M:%S %p}</td></tr>{''.join(rows)}
<tr><td>Image File</td><td>{esc(f)}</td></tr>
<tr><td>Emissivity / Distance / Humidity</td><td>{esc(info['emissivity'])} / {esc(info['distance'])} m / {esc(info['humidity'])} %</td></tr></table>
<table class="kv">
<tr><td>Inspection of</td><td>{esc(circuit)}</td></tr>
<tr><td>Inspector</td><td>{esc(rep.get('inspector'))}</td></tr>
<tr><td>Location</td><td>{esc(det.get('location') or (f'{float(lat):.6f}, {float(lon):.6f}' if lat else ''))}</td></tr>
<tr><td>Area</td><td>{esc(det.get('area'))}</td></tr>
<tr><td>Tower No.</td><td>{esc(tower)}</td></tr>
<tr><td>Equipment</td><td>{esc(det.get('equipment'))}</td></tr>
<tr><td>Component</td><td>{esc(det.get('component'))}</td></tr>
<tr><td>Phase</td><td>{esc(det.get('phase'))}</td></tr>
<tr><td>Side</td><td>{esc(det.get('side'))}</td></tr>
<tr><td>Suggested Action</td><td>{esc(det.get('action'))}</td></tr>
<tr><td>Notes</td><td>{esc(det.get('notes'))}</td></tr></table>
{footer}</section>""")

    # ---- priority table / disclaimer
    prow = "".join(f"<tr><td>{pt}</td><td>{desc} at maximum demand</td></tr>" for _, pt, _, desc in PRIORITIES)
    disclaimer = esc(rep.get("disclaimer") or "").replace("\n", "<br>")
    pages.append(f"""<section class="page">{header}
<h3 class="left">Priority Rating of Detected Temperatures</h3>
<table class="kv prio">{prow}</table>
<p class="it">Priorities may be upgraded due to low load, visible damage, high winds, safety issues or at the discretion of the thermographer.
&Delta;T is the hottest temperature in the marked area minus the reference (spot measurement where placed, otherwise the surrounding background).</p>
<p>The temperature of a resistive load increases approximately 4 times as the load doubles; inspections are most sensitive at high load.</p>
<table class="grid"><tr><th>Tests Undertaken</th><th>Test Equipment</th></tr>
<tr><td>Thermal Inspection</td><td>{esc(rep.get('equipment') or 'DJI Zenmuse H30T radiometric infrared camera (M400)')}</td></tr>
<tr><td>Visual Inspection</td><td>Zenmuse H30T zoom camera</td></tr></table>
<p class="it">{disclaimer}</p>{footer}</section>""")

    css = """
@page { size: A4; margin: 10mm 12mm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 9.5pt; color: #111; margin: 0; }
.page { page-break-after: always; display: flex; flex-direction: column; height: 276mm; overflow: hidden; }
.page:last-child { page-break-after: auto; }
.hdr { display: flex; justify-content: space-between; align-items: center; background: #0b2545; color: white; padding: 8px 16px; height: 54px; flex: none; }
.hdr .ttl { font-size: 22pt; } .hdr small { display: block; font-size: 10pt; opacity: .8; }
.hdr .logo { max-height: 56px; max-width: 180px; background: white; padding: 3px; }
.coverlogo { text-align: right; margin-bottom: 20px; } .coverlogo .logo { max-height: 90px; }
.ftr { margin-top: auto; background: #0b2545; color: white; padding: 6px 16px; font-size: 10pt; min-height: 14px; }
h3 { text-align: center; margin: 22px 0 4px; font-size: 11pt; } h3.left { text-align: left; }
table { border-collapse: collapse; width: 100%; } td, th { border: 1px solid #333; padding: 1px 5px; vertical-align: top; }
th { text-align: left; } .narrow { width: 70%; margin: 0 auto; }
.kv td:first-child { width: 35%; } .kv, .kv4 { margin-top: 6px; } .kv4 .lab { background: #eee; width: 15%; }
.band { text-align: center; margin-top: 8px; }
.pics td.pic { height: 58mm; text-align: center; vertical-align: middle; width: 50%; padding: 1px; }
.pics td.pic img { max-width: 100%; max-height: 57mm; }
.pics td.label { font-size: 12pt; }
.pics td.cap { background: #c8c8c8; padding: 1px 5px; }
.missing { color: #777; font-style: italic; }
.notes td { height: 70px; } .notesbox { margin-top: 4px; }
.prio td { font-size: 13pt; } .it { font-style: italic; }
"""
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Thermal Report</title><style>{css}</style></head><body>{''.join(pages)}</body></html>"
    out_html = Path(out_pdf).with_suffix(".html")
    out_html.write_text(doc, "utf-8")
    browser = next((p for p in EDGE if p.exists()), None)
    if not browser:
        return out_html, None, len(defects)
    if Path(out_pdf).exists():
        Path(out_pdf).unlink()
    # separate profile so it doesn't attach to (and hang on) an Edge/Chrome window that is already open
    profile = Path(out_pdf).parent / "_review_cache" / "pdf_browser_profile"
    subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                    f"--user-data-dir={profile}", f"--print-to-pdf={out_pdf}", out_html.as_uri()], timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_html, (Path(out_pdf) if Path(out_pdf).exists() else None), len(defects)


# ------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    run: Run = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", status=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        R = self.run
        try:
            if path in ("/", "/index.html"):
                return self._send(APP_HTML.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/data":
                images, hidden = R.filtered(q.get("excluded") == "1", q.get("undecided") == "1")
                return self._send({"run": str(R.dir), "images": images, "review": R.review, "hidden": hidden,
                                   "palettes": {k: v.tolist() for k, v in PALETTES.items()},
                                   "priorities": PRIORITIES})
            if path.startswith("/api/temps/"):
                f = path.rsplit("/", 1)[1]
                t, sky, lo, hi = R.view(f)
                body = np.round(t * 100).astype("<i2").tobytes()
                return self._send(body, "application/octet-stream",
                                  headers={"X-Width": str(t.shape[1]), "X-Height": str(t.shape[0]),
                                           "X-Auto-Lo": f"{lo:.2f}", "X-Auto-Hi": f"{hi:.2f}",
                                           "X-Sky-Frac": f"{sky.mean():.3f}"})
            if path.startswith("/api/sky/"):
                f = path.rsplit("/", 1)[1]
                _, sky, _, _ = R.view(f)
                rgba = np.zeros(sky.shape + (4,), np.uint8)
                rgba[sky] = (40, 120, 255, 110)
                buf = io.BytesIO()
                Image.fromarray(rgba, "RGBA").save(buf, "PNG")
                return self._send(buf.getvalue(), "image/png")
            if path.startswith("/api/lines/"):
                return self._send(R.lines(path.rsplit("/", 1)[1]))
            if path.startswith("/api/assets/"):
                f = path.rsplit("/", 1)[1]
                if f not in R.asset_png:
                    R.lines_cache.pop(f, None)
                    R.lines(f)
                return self._send(R.asset_png.get(f, b""), "image/png")
            if path.startswith("/api/thumb/"):
                f = path.rsplit("/", 1)[1]
                cp = R.cache / (f + ".thumb2.jpg")
                if not cp.exists():
                    t, _, lo, hi = R.view(f)
                    im = Image.fromarray(colourise(t, lo, max(hi + 2, lo + 5), "iron"))
                    im.thumbnail((200, 160))
                    im.save(cp, quality=80)
                return self._send(cp.read_bytes(), "image/jpeg")
            if path.startswith("/api/visual/"):
                f = path.rsplit("/", 1)[1]
                zp = R.zoom_path(f)
                if not zp:
                    return self._send({"error": "no zoom photo"}, status=404)
                w = int(q.get("w", 1400))
                cp = R.cache / f"{f}.z{w}.jpg"
                if not cp.exists():
                    im = Image.open(zp).convert("RGB")
                    full = im.size
                    im.thumbnail((w, w))
                    im.save(cp, quality=85)
                    (R.cache / f"{f}.z.json").write_text(json.dumps({"full": full}))
                return self._send(cp.read_bytes(), "image/jpeg")
            if path.startswith("/api/visualmap/"):
                f = path.rsplit("/", 1)[1]
                zp = R.zoom_path(f)
                if not zp:
                    return self._send({"ok": False})
                size = Image.open(zp).size
                ft = THERMAL_F35 * np.hypot(1280, 1024) / 43.27
                return self._send({"ok": True, "w": size[0], "h": size[1], "scale": zoom_focal_px(zp, *size) / ft})
            if path == "/report.pdf" and (R.dir / "thermal_report.pdf").exists():
                return self._send((R.dir / "thermal_report.pdf").read_bytes(), "application/pdf")
            if path == "/report.html" and (R.dir / "thermal_report.html").exists():
                return self._send((R.dir / "thermal_report.html").read_bytes(), "text/html; charset=utf-8")
            return self._send({"error": "not found"}, status=404)
        except Exception as e:
            return self._send({"error": str(e)}, status=500)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/review":
                self.run.save_review(data)
                return self._send({"ok": True})
            if self.path == "/api/report":
                if data:
                    self.run.save_review(data)
                html_path, pdf, n_def = build_report(self.run, self.run.dir / "thermal_report.pdf")
                return self._send({"ok": True, "pdf": "/report.pdf" if pdf else None, "html": "/report.html",
                                   "defects": n_def})
            return self._send({"error": "not found"}, status=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send({"error": str(e)}, status=500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="results folder from thermal_hotspots.py")
    ap.add_argument("--photos", help="folder searched recursively for zoom-camera _Z.JPG photos")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run).resolve()
    th._init_sdk()  # note: changes cwd to the SDK folder; all paths above are absolute
    Handler.run = Run(run_dir, args.photos)
    n = len(Handler.run.images)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"{n} flagged photos from {run_dir}\nReview app: {url}   (Ctrl+C to stop; progress saves to review.json)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
