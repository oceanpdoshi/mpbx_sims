#%%
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from mpbjax import grid_2d, kottke as K, modes as Mo, shapes as Shp, analysis as An
from mpbjax.helmholtz import sliceinv_3x3
from mpbjax.materials import Si3N4, SiO2, mat_vals

# TODO - :
# 1. first order of business, add the Matsuda model for MgO:LiNbO3 to MPBX's material library - follow its Gayer impelementaiton.
# 2. Second order of business, setup and visualize an LN waveguide
# 3. Run a mode solve for the waveguide, and visualize the mode profile
# 4. Run a mode solve sweep, and compare that to Tidy3D's local mode solver results w/dielectric smoothing. They should match!
# 5. Benchmark the runtime (currently just using the CPU version). 
# 6. Benchmark the runtime again, but this time using a single-GPU (fairest comparison to Tidy3D 30 mins.)