# constants.py
PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
AXIAL_TILT_DEGREES = 23.5
MIN_PLATE_SPEED_CM_YR = 1.0  # ~1 cm/year (slow moving plates)
MAX_PLATE_SPEED_CM_YR = 10.0  # ~10 cm/year (fast moving plates)
SEA_LEVEL_PRESSURE_HPA = 1013.25  # Standard atmospheric pressure at sea level
GAS_CONSTANT = 287.05  # Specific gas constant for dry air (J/kg·K)
GRAVITY = 9.81  # m/s²
cbar_obj = None  # Global variable to store the colorbar object
# Atmospheric layers (altitude in km, temperature gradient in °C/km)
ATMOSPHERIC_LAYERS = [
    {"name": "Troposphere", "altitude_range": (0, 12), "temp_gradient": -6.5},
    {"name": "Stratosphere", "altitude_range": (12, 50), "temp_gradient": 0.1},
    {"name": "Mesosphere", "altitude_range": (50, 80), "temp_gradient": -2.8},
    {"name": "Thermosphere", "altitude_range": (80, 700), "temp_gradient": 0.0}
]