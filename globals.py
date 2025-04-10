
from matplotlib import colors
import numpy as np

VMIN=-11000
VMAX=11000
NORM_ELEVATION = colors.Normalize(vmin=VMIN, vmax=VMAX) # Fixed range for colorbar
PLATES = 15 #earth rate
SUBDIVISIONS: int = 3 #3 is balanced for testing, but 5 is needed for reasonable accuracy
MAX_ANGULAR_VELOCITY_RAD_PER_YR = np.radians(1.0) # Corresponds to ~11 cm/yr at equator for Earth radius. Adjust as needed.
ELEVATION_MOUNTAIN_BASE = 10000.0 # meters
ELEVATION_TRENCH_BASE = -11000.0 # meters
ELEVATION_DIFFUSION_FACTOR = 0.05 # How much elevation spreads per pass
ELEVATION_DIFFUSION_PASSES = 5 # Number of smoothing passes
RADIUS = 6371000

CONTINENTAL_PLATE_PROB = 0.3  # Probability of a plate being continental
CONTINENTAL_BASE_ELEVATION = 500.0 # meters
OCEANIC_BASE_ELEVATION = -4000.0 # meters

PHI = (1.0 + np.sqrt(5.0)) / 2.0