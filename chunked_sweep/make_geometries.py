"""Write a geometry list (CSV) for a multi-geometry sweep: every combination of the lists below.

    python make_geometries.py geometries.csv

Edit the lists, run, then submit with supercloud/submit_geometry_sweep.sh geometries.csv.
Units: µm and degrees. t_clad: inf = SiO2 everywhere above the film, 0 = air,
0 < t_clad < inf = planar SiO2 whose top is t_clad above the ridge top.
t_undercut picks the rib type: 0 = regular rib on a flat SiO2 substrate; > 0 = undercut rib (the
SiO2 is etched this deep below the LN at the sidewall angle, leaving a pedestal), which has no
cladding and no slab, so it is only made with t_clad = 0 and etch fraction 1.0.
"""
import csv
import itertools
import math
import numpy as np
import sys

import sweep_config as C

T_LN = np.arange(0.400, 0.650, 0.100)   # LN film thickness
ETCH_FRACTION = [1.0]                   # etch depth / t_ln (1.0 = fully etched, no slab)
W_TOP = np.arange(0.3, 2.05, 0.100)     # ridge top width
SIDEWALL_DEG = [70.0]                   # sidewall angle from horizontal (90 = vertical)
T_CLAD = [1.350]                        # cladding
T_UNDERCUT = [0.0]                      # RIB TYPE: 0 = regular rib; > 0 = undercut rib, SiO2 etched this deep
                                        # [µm] below the LN (only combined with t_clad 0 and etch fraction
                                        # 1.0; other combinations are skipped, with a count printed).
                                        # e.g. undercut sweep: T_CLAD = [0.0]; T_UNDERCUT = [0.5, 1.0]

# The film interfaces sit at the same sub-pixel position in every chunk only if every film / slab
# thickness is a whole number of pixels (README §2.2). With FIT_DY_TO_FILM (sweep_config.py) the
# pitch is adjusted per geometry so t_ln always is; a slab (partial etch) then still has to be a
# whole number of those pixels. Without it, both must be multiples of DY.
DY = 0.0125


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "geometries.csv"
    rows, skipped = [], 0
    for t, ef, w, sw, tc, uc in itertools.product(T_LN, ETCH_FRACTION, W_TOP, SIDEWALL_DEG, T_CLAD, T_UNDERCUT):
        # round to 1 pm / 1e-6 deg: lists built by repeated addition carry float noise
        # (0.40800000000000003), and an unrounded t_ln next to a rounded etch leaves a
        # ~1e-17 µm "slab" that the geometry treats as a real layer
        t, w, sw, tc, uc = (round(float(v), 6) for v in (t, w, sw, tc, uc))
        etch = round(t * ef, 6)
        if uc < 0:
            raise SystemExit(f"T_UNDERCUT: {uc} < 0 (0 = regular rib, > 0 = undercut depth in µm)")
        if uc > 0 and (tc != 0 or etch != t):     # an undercut rib has no cladding and no slab
            skipped += 1
            continue
        dy = t / max(1, round(t / DY)) if C.FIT_DY_TO_FILM else DY     # = run_geometry.film_dy
        for name, th in (("t_ln", t), ("slab", t - etch), ("t_undercut", uc)):
            if th > 0 and abs(th / dy - round(th / dy)) > 1e-6:
                print(f"note: {name} = {th:.4f} µm is not a multiple of dy = {1e3 * dy:.3f} nm "
                      f"(t_ln={t}, etch fraction={ef})")
        rows.append(dict(name="", t_ln=t, etch_depth=etch, w_top=w, sidewall_deg=sw, t_clad=tc, t_undercut=uc))
    if skipped:
        print(f"skipped {skipped} combination(s) with t_undercut > 0 and t_clad != 0 or etch fraction != 1 "
              f"(an undercut rib has no cladding and no slab)")
    if not rows:
        raise SystemExit("no geometries (check T_UNDERCUT against T_CLAD / ETCH_FRACTION)")
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("# made by make_geometries.py; one row per geometry (µm, degrees); t_clad inf = fully clad, 0 = air;"
                " t_undercut 0 = regular rib, > 0 = undercut rib\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_uc = sum(r["t_undercut"] > 0 for r in rows)
    print(f"{len(rows)} geometries ({len(rows) - n_uc} regular rib, {n_uc} undercut rib) -> {out}")


if __name__ == "__main__":
    main()
