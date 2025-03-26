
import json

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
AXIAL_TILT_DEGREES = 23.5
MIN_PLATE_SPEED_CM_YR = 1.0  # ~1 cm/year (slow moving plates)
MAX_PLATE_SPEED_CM_YR = 10.0  # ~10 cm/year (fast moving plates)
SEA_LEVEL_PRESSURE_HPA = 1013.25  # Standard atmospheric pressure at sea level
GAS_CONSTANT = 287.05  # Specific gas constant for dry air (J/kg·K)
GRAVITY = 9.81  # m/s²
cbar_obj = None  # Global variable to store the colorbar object
HADLEY_CELL_WIDTH = 30  # Degrees latitude
FERREL_CELL_WIDTH = 30  # Degrees latitude
POLAR_CELL_WIDTH = 30   # Degrees latitude
CORIOLIS_FACTOR = 0.0001  # Simplified Coriolis effect factor
# Atmospheric layers (altitude in km, temperature gradient in °C/km)
ATMOSPHERIC_LAYERS = [
    {"name": "Troposphere", "altitude_range": (0, 12), "temp_gradient": -6.5},
    {"name": "Stratosphere", "altitude_range": (12, 50), "temp_gradient": 0.1},
    {"name": "Mesosphere", "altitude_range": (50, 80), "temp_gradient": -2.8},
    {"name": "Thermosphere", "altitude_range": (80, 700), "temp_gradient": 0.0}
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
