"""Settings for the chunked TFLN mode-solve sweep. Edit THIS file; run_geometry.py reads it.

Every number here is explained in README.md (section 2: why these chunks; section 3: costs).
Units: µm.
"""
import dataclasses

ALL_TARGETS = ("TE00", "TM00", "TE10", "TM10", "TE20", "TM20", "TE30", "TM30")
LOW_TARGETS = ("TE00", "TM00", "TE10", "TM10")


@dataclasses.dataclass(frozen=True)
class Chunk:
    name: str          # used in file names
    lam_min: float     # µm, inclusive
    lam_max: float     # µm, inclusive. A wavelength shared by two chunks is solved in both:
                       # the lower chunk's result goes into the tables, the other one into the
                       # seam check (how much the grid change moves n_eff / n_g there).
    Lx: float          # window [µm]. Rounded to an FFT-friendly pixel count (at most 2 % smaller),
    Ly: float          # and ENLARGED automatically if the geometry needs it (see MIN_MARGIN).
    dx: float          # lateral pitch [µm]
    dy: float          # vertical pitch [µm]
    nev: int           # modes PER x-PARITY SECTOR (2·nev modes per wavelength in total)
    targets: tuple = ALL_TARGETS   # modes that must be found (else nev is raised, see NEV_EXTRA)


# ---------------------------------------------------------------------------------------
# The chunks. nev here = your proposal's total / 2 (the parity solve splits the modes into two
# equal sectors; README §2.3). dy = 12.5 nm in EVERY chunk (README §2.2): the n_eff / n_g error is
# set mostly by dy, so a coarser dy at long λ would add up to ~5e-4 (n_eff) / ~8e-4 (n_g) and a
# visible step at every chunk boundary; coarsening only dx costs ~25 % more in total.
CHUNKS = (
    Chunk("A_0400-0600nm", 0.40, 0.60, Lx=4.0,  Ly=3.0, dx=0.0125, dy=0.0125, nev=6),
    Chunk("B_0600-1000nm", 0.60, 1.00, Lx=5.0,  Ly=4.0, dx=0.0125, dy=0.0125, nev=5),
    Chunk("C_1000-1300nm", 1.00, 1.30, Lx=7.0,  Ly=7.0, dx=0.020,  dy=0.0125, nev=4, targets=LOW_TARGETS),
    Chunk("D_1300-1700nm", 1.30, 1.70, Lx=9.0,  Ly=8.0, dx=0.025,  dy=0.0125, nev=3, targets=LOW_TARGETS),
    Chunk("E_1700-2000nm", 1.70, 2.00, Lx=10.0, Ly=9.0, dx=0.025,  dy=0.0125, nev=3, targets=LOW_TARGETS),
)
# Your original proposal (dy = dx in every chunk) — ~20 % cheaper, ~2-4x larger error above 1 µm.
# To use it, replace the last three lines above with:
#   Chunk("C_1000-1300nm", 1.00, 1.30, Lx=6.0, Ly=6.0, dx=0.020, dy=0.020, nev=4, targets=LOW_TARGETS),
#   Chunk("D_1300-1700nm", 1.30, 1.70, Lx=7.0, Ly=7.0, dx=0.025, dy=0.025, nev=3, targets=LOW_TARGETS),
#   Chunk("E_1700-2000nm", 1.70, 2.00, Lx=8.0, Ly=8.0, dx=0.025, dy=0.025, nev=3, targets=LOW_TARGETS),

STEP = 0.010            # wavelength step [µm]; must divide every chunk's range

# ---------------------------------------------------------------------------------------
# Solver
USE_PARITY = True       # solve the two x-mirror sectors separately (README §2.3). False: one solve
                        # with 2·nev modes (≈2x slower, and TE/TM modes of opposite parity can mix
                        # where they cross).
ALIGN_FRAC = 1 / 3      # LN/substrate interface 1/3 pixel above a pixel centre (README §2.2)
FIT_DY_TO_FILM = True   # per geometry, stretch / shrink each chunk's dy to the nearest pitch that
                        # makes t_ln a whole number of pixels (0.394 µm at dy = 12.5 nm -> 32 px of
                        # 12.31 nm). Both film interfaces then sit at ALIGN_FRAC in every geometry,
                        # so the error varies smoothly across a t_ln sweep instead of wiggling with
                        # a 12.5 nm period (README §2.2). No effect when t_ln is already a multiple
                        # of dy. Changing it changes those geometries' grids, so they refuse to
                        # resume a run started with the other setting. False: use the chunk's dy.
NEV_EXTRA = 2           # if a target mode is missing at some λ while the deepest solved mode of
                        # its sector is still guided, re-solve that λ with nev + NEV_EXTRA per
                        # sector, and keep that for the rest of the work unit if it found the
                        # target. Skipped when the sector shows an anticrossing (duplicate or
                        # hybrid label: the target is there, mislabelled). 0 = never.
MIN_MARGIN = 1.4        # the window must leave >= MIN_MARGIN·λ_max of cladding on every side of
                        # the ridge base / film (else it is enlarged). 1.4 matches the window
                        # study: it keeps the window error of n_g below ~1e-5 except for modes
                        # near cutoff (TM10 above ~1.8 µm).
K_TOL = 1e-8            # Newton tolerance on k [1/µm]; n_eff is converged to ~K_TOL·λ
EIG_TOL = 1e-8          # LOBPCG relative residual tolerance

# ---------------------------------------------------------------------------------------
# Output
FIELD_DTYPE = "complex64"   # stored E fields (8 bytes per component per pixel)
# E fields go into these files (<geometry>_fields_<label>.h5), one HDF5 group per chunk (the chunks
# have different grids). gzip + shuffle shrinks the fields to ~60 %; estimated sizes for the default
# geometry and chunks: 0400-1000nm ~1.4 GB, 1000-2000nm ~1.5 GB. A chunk not listed gets its own file.
FIELD_FILES = (
    ("0400-1000nm", ("A_0400-0600nm", "B_0600-1000nm")),
    ("1000-2000nm", ("C_1000-1300nm", "D_1300-1700nm", "E_1700-2000nm")),
)
FIELD_COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)
EDGE_FLAG = 1e-5            # flag a mode whose intensity within 0.25 µm of the window edge is above this
NEAR_GAP = 0.008            # flag a mode whose same-sector neighbour is closer than this in n_eff
                            # (an anticrossing: n_g of both branches changes fast there; TE30 at
                            # 0.4 µm, 0.0075 above TM01, has an n_g error of -7e-4, README §2.6)

# ---------------------------------------------------------------------------------------
# --test: tiny, coarse configuration (a few minutes on a laptop) to check the whole pipeline.
TEST_CHUNKS = (
    Chunk("A_test", 0.50, 0.60, Lx=3.8, Ly=2.5, dx=0.05, dy=0.025, nev=4),
    Chunk("B_test", 0.60, 0.70, Lx=4.0, Ly=3.0, dx=0.05, dy=0.025, nev=3, targets=LOW_TARGETS),
)
TEST_STEP = 0.050
