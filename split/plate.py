# plate.py
import numpy as np
import random
from constants import MIN_PLATE_SPEED_CM_YR, MAX_PLATE_SPEED_CM_YR, PLANET_RADIUS_KM, SEA_LEVEL_PRESSURE_HPA
from utils import cartesian_to_lat_lon, calculate_temperature

class Plate:
    def __init__(self, plate_id, center_vertex, movement_direction):
        self.plate_id = plate_id
        self.center_vertex = center_vertex
        # Initialize with a direction but no magnitude
        self.movement_direction = movement_direction / np.linalg.norm(movement_direction)
        self.speed_cm_yr = random.uniform(MIN_PLATE_SPEED_CM_YR, MAX_PLATE_SPEED_CM_YR)
        self.vertices = set()
        self.center_point = None
        self.temperature = 15  # Average temperature in °C
        self.pressure = SEA_LEVEL_PRESSURE_HPA  # Average pressure in hPa
        self.elevation = 0
        self.water_fraction = 0

    def move(self, days_elapsed):
        """Move the plate based on real-world time scaling."""
        # Convert speed from cm/year to km/day
        speed_km_day = (self.speed_cm_yr / 100000) / 365.25
        movement_distance = speed_km_day * days_elapsed

        # Apply movement
        movement_vector = self.movement_direction * movement_distance
        self.center_point += movement_vector

        # Project back to sphere surface
        self.center_point = self.center_point / np.linalg.norm(self.center_point) * PLANET_RADIUS_KM

        # Update plate temperature based on movement (with pressure)
        lat, lon = cartesian_to_lat_lon(*self.center_point)
        elevation = 0  # Using 0 as plate center elevation for simplicity
        self.temperature = calculate_temperature(lat, lon, 0, 12, elevation, 0, self.pressure)

        # Update plate pressure based on temperature (ideal gas law)
        # Higher temperatures generally lead to lower pressure systems
        self.pressure = SEA_LEVEL_PRESSURE_HPA * (1 - 0.01 * (self.temperature - 15))