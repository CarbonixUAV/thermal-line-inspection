"""Self-check of the decision logic (no imagery or DJI SDK needed):  python test_core.py"""
import numpy as np

import thermal_hotspots as th


def t(shift, rel=None, align=1.0, score=0.95, expected=4.0, ground=(0.0, 0.0)):
    """One parallax test against a neighbouring photo."""
    return {"shift": shift, "rel": shift if rel is None else rel, "align": align, "score": score,
            "expected": expected, "ground_dx": ground[0], "ground_dy": ground[1]}


def main():
    # priority bands (delta T, degC)
    assert [th.severity(d) for d in (2, 3, 9.9, 10, 29.9, 30, 80)] == \
        ["Below PT 180", "PT 180", "PT 180", "PT 90", "PT 90", "PT 30", "PT 1"]

    c = th.parallax_class
    assert c([t(4), t(5)], area_px=50) == "elevated"                      # shifts along the parallax direction
    assert c([t(4, expected=20), t(5, expected=20)], 50) == "unknown"     # too little for something 8 m up this close
    assert c([t(0.3), t(0.5)], 50) == "ground"                            # stays put
    assert c([t(0.3), t(0.5)], 5) == "tiny"                               # too small to track
    assert c([t(6, align=0.2), t(5, align=0.3)], 50) == "moving"          # walked across the parallax direction
    assert c([t(6, align=None)], 50) == "untested"                        # drone hovering: no direction to judge
    assert c([t(1, score=0.3)], 50) == "untested"                         # no confident match
    assert c([t(0.3)], 50, ground_dist=5000) == "far"                     # near the horizon
    # ring around the hotspot moving -> it isn't ground, use the raw shift
    assert c([t(14, rel=0.3, ground=(14, 0), align=None)], 200) == "untested"

    # sky: cold rows above the horizon are sky, the warm ground below is not
    temps = np.full((1024, 1280), 15.0, np.float32)
    temps[:300] = -20
    sky = th.sky_mask(temps, pitch_deg=-5)
    assert sky[:250].all() and not sky[400:].any()
    print("ok")


if __name__ == "__main__":
    main()
