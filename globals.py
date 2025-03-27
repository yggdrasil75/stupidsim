
import json
import math

config = {}
def loadConfig() -> json:
	global config
	with open('planet_config.json') as f:
		config = json.load(f)
	return config

	
def saveConfig(config: json =config):
	with open('planet_config.json', 'w') as f:
		json.dump(config, f)

PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
PLANET_MASS_EARTH = 1.0  # Earth masses
PLANET_RADIUS_EARTH = 1.0  # Earth radii
GAS_CONSTANT = 287.05  # Specific gas constant for dry air (J/kg·K)
# Universal gravitational constant (m^3 kg^-1 s^-2)
GRAVITATIONAL_CONSTANT = 6.67430e-11

# Earth's mass (kg)
EARTH_MASS = 5.9722e24

# Earth's radius (m)
EARTH_RADIUS = 6371000.0

# Calculate actual planet properties
PLANET_MASS = PLANET_MASS_EARTH * EARTH_MASS
PLANET_RADIUS = PLANET_RADIUS_EARTH * EARTH_RADIUS
PLANET_RADIUS_KM = PLANET_RADIUS / 1000.0

# Calculate surface gravity (m/s²)
GRAVITY = (GRAVITATIONAL_CONSTANT * PLANET_MASS) / (PLANET_RADIUS ** 2)

# Calculate escape velocity (m/s)
ESCAPE_VELOCITY = math.sqrt(2 * GRAVITATIONAL_CONSTANT * PLANET_MASS / PLANET_RADIUS)

# Calculate atmospheric scale height (m) - approximate
MEAN_MOLECULAR_WEIGHT = 0.029  # kg/mol (approximate for Earth's air)
ATMOSPHERIC_SCALE_HEIGHT = (GAS_CONSTANT * 288.15) / (MEAN_MOLECULAR_WEIGHT * GRAVITY)  # Using 288K as reference temp
AXIAL_TILT_DEGREES = 23.5
MIN_PLATE_SPEED_CM_YR = 1.0 * (GRAVITY/9.81)  # Scale with gravity
MAX_PLATE_SPEED_CM_YR = 10.0 * (GRAVITY/9.81)  # Scale with gravity
SEA_LEVEL_PRESSURE_HPA = 1013.25  # Standard atmospheric pressure at sea level
GRAVITY = 9.81  # m/s²
cbar_obj = None  # Global variable to store the colorbar object
HADLEY_CELL_WIDTH = 30  # Degrees latitude
FERREL_CELL_WIDTH = 30  # Degrees latitude
POLAR_CELL_WIDTH = 30   # Degrees latitude
CORIOLIS_FACTOR = 0.0001  # Simplified Coriolis effect factor
# Atmospheric layers (altitude in km, temperature gradient in °C/km)
ATMOSPHERIC_LAYERS = [
    {
        "name": "troposphere",
        "altitude_range": (0, 12),  # Default mid-latitude range
        "temp_gradient": -6.5,  # °C per km
        "lat_variation": {
            "equator": (0, 18),  # Higher at equator (0-18km)
            "midlat": (0, 12),   # Mid-latitude range (0-12km)
            "polar": (0, 8)      # Lower at poles (0-8km)
        }
    },
    {
        "name": "tropopause",
        "altitude_range": (12, 15),  # Default mid-latitude range
        "temp_gradient": 0,  # Isothermal
        "lat_variation": {
            "equator": (18, 20),  # Higher at equator (18-20km)
            "midlat": (12, 15),   # Mid-latitude range (12-15km)
            "polar": (8, 10)       # Lower at poles (8-10km)
        }
    },
    {
        "name": "stratosphere",
        "altitude_range": (15, 50),
        "temp_gradient": 1.0,  # Temperature inversion
        "lat_variation": None  # No latitudinal variation
    },
    {
        "name": "stratopause",
        "altitude_range": (50, 55),
        "temp_gradient": 0,  # Isothermal
        "lat_variation": None
    },
    {
        "name": "mesosphere",
        "altitude_range": (55, 85),
        "temp_gradient": -2.0,  # Temperature decreases with altitude
        "lat_variation": None
    },
    {
        "name": "mesopause",
        "altitude_range": (85, 90),
        "temp_gradient": 0,  # Isothermal
        "lat_variation": None
    },
    {
        "name": "thermosphere",
        "altitude_range": (90, 500),
        "temp_gradient": 2.5,  # Temperature increases with altitude
        "lat_variation": None
    },
    {
        "name": "exosphere",
        "altitude_range": (500, 10000),
        "temp_gradient": 0,  # Effectively isothermal
        "lat_variation": None
    }
]
# Atmospheric composition constants (by volume)
ATMOSPHERIC_COMPOSITION = {
    'N2': 0.7808,  # Nitrogen
    'O2': 0.2095,  # Oxygen
    'Ar': 0.0093,  # Argon
    'CO2': 0.0004, # Carbon dioxide
    'H2O': 0.01,   # Water vapor (variable)
    'CH4': 1.8e-6, # Methane
    'O3': 7.0e-6   # Ozone (variable)
}
# Specific gas constants (J/kg·K)
GAS_CONSTANTS = {
    'N2': 296.80,
    'O2': 259.84,
    'Ar': 208.13,
    'CO2': 188.92,
    'H2O': 461.50,
    'CH4': 518.28,
    'O3': 173.21
}
# Molecular weights (kg/mol)
MOLECULAR_WEIGHTS = {
    'N2': 0.0280134,
    'O2': 0.0319988,
    'Ar': 0.039948,
    'CO2': 0.0440095,
    'H2O': 0.01801528,
    'CH4': 0.0160425,
    'O3': 0.0479982
}
# Greenhouse gas absorption coefficients (W/m² per kg/m²)
GREENHOUSE_ABSORPTION = {
    'CO2': 0.05,
    'H2O': 0.1,
    'CH4': 0.03,
    'O3': 0.15
}
ALBEDO_VALUES = {
    'water': 0.06,       # Ocean water
    'ice': 0.6,         # Fresh ice/snow
    'forest': 0.15,     # Dense forest
    'grassland': 0.25,  # Grassland
    'desert': 0.35,     # Sandy desert
    'rock': 0.2,        # Bare rock
    'urban': 0.15       # Cities (if implemented)
}