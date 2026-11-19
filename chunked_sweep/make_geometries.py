"""Write a geometry list (CSV) for a multi-geometry sweep: every combination of the lists below.

    python make_geometries.py geometries.csv

Edit the lists, run, then submit with supercloud/submit_geometry_sweep.sh geometries.csv.
Units: µm and degrees. t_clad: inf = SiO2 everywhere above the film, 0 = air,
0 < t_clad < inf = planar SiO2 whose top is t_clad above the ridge top.
"""
import csv
import itertools
import math
import numpy as np
import sys

import sweep_config as C

T_LN = np.arange(0.390, 0.4101, 0.002)  # LN film thickness
ETCH_FRACTION = [1.0]                   # etch depth / t_ln (1.0 = fully etched, no slab)
W_TOP = np.arange(1.500, 1.9001, 0.050) # ridge top width
SIDEWALL_DEG = [70.0]                   # sidewall angle from horizontal (90 = vertical)
T_CLAD = [1.350]                        # cladding

# The film interfaces sit at the same sub-pixel position in every chunk only if every film / slab
# thickness is a whole number of pixels (README §2.2). With FIT_DY_TO_FILM (sweep_config.py) the
# pitch is adjusted per geometry so t_ln always is; a slab (partial etch) then still has to be a
# whole number of those pixels. Without it, both must be multiples of DY.
DY = 0.0125


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "geometries.csv"
    rows = []
    for t, ef, w, sw, tc in itertools.product(T_LN, ETCH_FRACTION, W_TOP, SIDEWALL_DEG, T_CLAD):
        # round to 1 pm / 1e-6 deg: lists built by repeated addition carry float noise
        # (0.40800000000000003), and an unrounded t_ln next to a rounded etch leaves a
        # ~1e-17 µm "slab" that the geometry treats as a real layer
        t, w, sw, tc = (round(v, 6) for v in (t, w, sw, tc))
        etch = round(t * ef, 6)
        dy = t / max(1, round(t / DY)) if C.FIT_DY_TO_FILM else DY     # = run_geometry.film_dy
        for name, th in (("t_ln", t), ("slab", t - etch)):
            if th > 0 and abs(th / dy - round(th / dy)) > 1e-6:
                print(f"note: {name} = {th:.4f} µm is not a multiple of dy = {1e3 * dy:.3f} nm "
                      f"(t_ln={t}, etch fraction={ef})")
        rows.append(dict(name="", t_ln=t, etch_depth=etch, w_top=w, sidewall_deg=sw, t_clad=tc))
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("# made by make_geometries.py; one row per geometry (µm, degrees); t_clad inf = fully clad, 0 = air\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} geometries -> {out}")


if __name__ == "__main__":
    main()
