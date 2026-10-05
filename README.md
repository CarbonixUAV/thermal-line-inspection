# thermal-line-inspection

Finds and reviews thermal hotspots on overhead transmission lines from drone
surveys flown with a DJI Zenmuse H30T (radiometric `_T.JPG` + zoom `_Z.JPG`).

1. **`thermal_hotspots.py`** reads true temperatures from each R-JPEG, projects
   the line's towers and spans (from a GIS export) into the photo, finds warm
   spots in that corridor, and uses parallax against the neighbouring photos to
   drop what is on the ground or moving (cattle, sun-warmed grass).
2. **`review_server.py`** opens a local review app: mark detections real or
   false positive, measure spots/boxes/lines like DJI Thermal Analysis Tool,
   fill in defect details, and generate a PDF report.

No imagery, GIS data or results are stored in this repo.

## Setup (Windows)

```
pip install -r requirements.txt
```

Download the **DJI Thermal SDK** (v1.8 or later, it must support your camera) from
<https://www.dji.com/downloads/softwares/dji-thermal-sdk> and either unzip it into
`dji_thermal_sdk/` here, or point `DJI_TSDK_BIN` at its
`utility/bin/windows/release_x64` folder. The SDK is under DJI's licence and is
not included.

The PDF report is printed with Microsoft Edge or Chrome (headless).

## Use

```
python thermal_hotspots.py <folder of _T.JPG> [more folders] --network towers.kmz --out results/run1
python review_server.py results/run1 --photos <folder containing the _Z.JPG photos>
```

`--network` is a KMZ of tower point placemarks with `FL_TOWER_NO`,
`SUPPORTING_CIRCUIT_ID` and `HEIGHT` (ground elevation) fields; consecutive
tower numbers on a circuit become spans. Run `python thermal_hotspots.py -h`
for the tuning options (ΔT threshold, corridor width, sag, range).

Outputs: `images.csv`, `hotspots.csv` (one row per hotspot with priority,
parallax class and nearest span/tower) and `run_args.json`. The review app
saves your work to `review.json` in the same folder.

## Parallax classes

Each hotspot is matched in the photos up to two either side, after warping
them so the ground lines up, and compared with the motion of the ground right
around it:

| class | meaning | review app |
|---|---|---|
| elevated | moves against the ground along the parallax direction (conductor, tower, tree) | shown, first |
| untested | no usable neighbour (isolated close-up, drone hovering, mis-match) | shown |
| unknown / far / tiny | ambiguous, ground behind is > 1.2 km away, or < 9 px | hidden behind a checkbox |
| ground / moving | stays put, or walked across the parallax direction | auto-excluded |

Priority bands (ΔT above background): PT 180 < 10 °C, PT 90 10–30 °C,
PT 30 30–80 °C, PT 1 > 80 °C.

## Check

```
python test_core.py
```
