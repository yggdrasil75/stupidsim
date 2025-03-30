import random

import numpy as np

from globals import ALBEDO_VALUES, MAX_PLATE_SPEED_CM_YR, MIN_PLATE_SPEED_CM_YR, PLANET_RADIUS_KM, SEA_LEVEL_PRESSURE_HPA
from pressure import calculate_diurnal_pressure_variation, calculate_pressure_with_layers, calculate_seasonal_pressure_variation
from temperature import calculate_temperature
from utils import calculate_slope, cartesian_to_lat_lon, find_spherical_neighbors



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
        self.pressure_systems = {
            'base': SEA_LEVEL_PRESSURE_HPA,
            'seasonal': 0,
            'diurnal': 0,
            'storm': 0,
            'boundary': 0
        }
        self.humidity = 50  # Average humidity in %
        self.humidity_history = []  # Track humidity changes over time
        self.evaporation_rate = 0
        self.precipitation_rate = 0
        self.last_update_day = 0
        self.albedo = 0.3  # Default mid-range albedo
        self.albedo_history = []  # Track albedo changes over time
        self.surface_composition = {
            'water': 0,
            'ice': 0,
            'forest': 0,
            'grassland': 0,
            'desert': 0,
            'rock': 0
        }

    def update_pressure_systems(self, day_of_year, hour_of_day, elevation, temperature, active_storms):
        """Update all pressure systems for this plate."""
        lat, lon = cartesian_to_lat_lon(*self.center_point)
        
        # Base pressure from elevation and temperature
        self.pressure_systems['base'] = calculate_pressure_with_layers(elevation, temperature, lat)
        
        # Seasonal variation
        self.pressure_systems['seasonal'] = calculate_seasonal_pressure_variation(lat, day_of_year)
        
        # Diurnal variation
        self.pressure_systems['diurnal'] = calculate_diurnal_pressure_variation(
            lat, lon, elevation, day_of_year, hour_of_day, temperature
        )
        
        # Storm effects
        storm_effect = 0
        for storm in active_storms:
            storm_effect += storm.calculate_pressure_effect(lat, lon)
        self.pressure_systems['storm'] = storm_effect
        
        # Combine all effects with different weights
        self.pressure = (
            0.6 * self.pressure_systems['base'] +
            0.15 * self.pressure_systems['seasonal'] +
            0.05 * self.pressure_systems['diurnal'] +
            0.2 * self.pressure_systems['storm'] +
            self.pressure_systems['boundary']
        )
        
        # Update last update time
        self.last_update_day = day_of_year
        
    def update_albedo(self, elevations, water_fraction, temperatures, day_of_year):
        """Calculate plate albedo based on surface composition."""
        if not self.vertices:
            return
            
        # Calculate surface composition percentages
        surface_counts = {
            'water': 0,
            'ice': 0,
            'forest': 0,
            'grassland': 0,
            'desert': 0,
            'rock': 0
        }
        
        for vertex_idx in self.vertices:
            elev = elevations[vertex_idx]
            water = water_fraction[vertex_idx]
            temp = temperatures[vertex_idx]
            
            # Determine surface type for this vertex
            if water > 0.9:
                surface_type = 'water'
            elif temp < -5 and water > 0.1:
                surface_type = 'ice'
            elif elev < 0:
                surface_type = 'water'
            elif water > 0.3:
                surface_type = 'forest' if temp > 10 else 'grassland'
            elif elev > 4000:
                surface_type = 'ice' if temp < 0 else 'rock'
            elif temp > 30 and water < 0.1:
                surface_type = 'desert'
            else:
                surface_type = 'grassland'
                
            surface_counts[surface_type] += 1
        
        # Calculate percentages
        total = len(self.vertices)
        for key in surface_counts:
            self.surface_composition[key] = surface_counts[key] / total
        
        # Calculate weighted average albedo
        weighted_albedo = 0
        for surface_type, percentage in self.surface_composition.items():
            weighted_albedo += percentage * ALBEDO_VALUES[surface_type]
        
        # Apply seasonal adjustments
        lat, lon = cartesian_to_lat_lon(*self.center_point)
        if 'ice' in self.surface_composition and self.surface_composition['ice'] > 0.1:
            season_factor = np.sin(np.radians(day_of_year/365 * 360))
            if lat > 0:  # Northern hemisphere
                season_factor *= -1
            weighted_albedo *= (1 + 0.3 * season_factor)
        
        self.albedo = weighted_albedo
        self.albedo_history.append(weighted_albedo)
        if len(self.albedo_history) > 100:
            self.albedo_history.pop(0)

    def update_humidity_systems(self, day_of_year, elevations, water_fraction, vertices):
        """Update humidity based on plate conditions and neighboring water"""
        if not self.vertices:
            return
            
        # Calculate average conditions for the plate
        avg_elevation = np.mean([elevations[i] for i in self.vertices])
        avg_water = np.mean([water_fraction[i] for i in self.vertices])
        
        # Get plate center coordinates
        lat, lon = cartesian_to_lat_lon(*self.center_point)
        
        # Seasonal humidity variation
        seasonal_factor = 1 + 0.3 * np.sin(np.radians(day_of_year/365 * 360))
        
        # Evaporation based on temperature and water availability
        self.evaporation_rate = (0.5 * avg_water * (1 + 0.02 * (self.temperature - 15)) * seasonal_factor)
        
        # Precipitation based on humidity and elevation
        self.precipitation_rate = (0.3 * self.humidity * (1 + 0.01 * avg_elevation)) * seasonal_factor
        
        # Net humidity change
        humidity_change = self.evaporation_rate - self.precipitation_rate
        
        # Update plate humidity with damping
        self.humidity += humidity_change * 0.1
        self.humidity = np.clip(self.humidity, 10, 90)  # Keep within reasonable bounds
        
        # Record history
        self.humidity_history.append(self.humidity)
        if len(self.humidity_history) > 100:
            self.humidity_history.pop(0)

    def move(self, days_elapsed, day_of_year, hour_of_day, active_storms, elevations, water_fraction, vertices):
        """Move the plate based on real-world time scaling."""
        # Convert speed from cm/year to km/day
        speed_km_day = (self.speed_cm_yr / 100000) / 365.25
        movement_distance = speed_km_day * days_elapsed

        # Apply movement
        movement_vector = self.movement_direction * movement_distance
        self.center_point += movement_vector

        # Project back to sphere surface
        self.center_point = self.center_point / np.linalg.norm(self.center_point) * PLANET_RADIUS_KM

        # Calculate average elevation for the plate
        if self.vertices:
            avg_elevation = np.mean([elevations[i] for i in self.vertices])
        else:
            avg_elevation = 0
            
        # Update plate climate systems
        self.update_pressure_systems(day_of_year, hour_of_day, avg_elevation, self.temperature, active_storms)
        self.update_humidity_systems(day_of_year, elevations, water_fraction, vertices)
        
        # Update plate temperature based on movement (with pressure)
        lat, lon = cartesian_to_lat_lon(*self.center_point)
        self.temperature = calculate_temperature(lat, lon, 0, 12, avg_elevation, 0, self.pressure, self.albedo)
        
def simulate_plate_tectonics_spherical(vertices, faces, plates, plate_assignment, elevations, 
                                       days_elapsed=30, max_neighbor_distance_km=1000, step_size=0.03, 
                                       active_storms=[], water_fraction=0, temperatures=None, rainfall=None, humidity=None):
    """Simulate plate tectonics on spherical mesh with pressure and temperature effects."""
    # Get current date and time
    day_of_year = (plates[0].last_update_day + days_elapsed) % 365
    hour_of_day = 12  # Noon for simplicity
    
    # Move plates and update their pressure systems
    for plate in plates:
        plate.move(days_elapsed, day_of_year, hour_of_day, active_storms, elevations, water_fraction, vertices)
    if water_fraction is None:
        water_fraction = np.where(elevations < 0, 1.0, 0.0)

    # Calculate boundary effects
    boundary_effects = np.zeros_like(elevations)
    pressure_changes = np.zeros(len(vertices))
    temperature_changes = np.zeros(len(vertices))
    humidity_changes = np.zeros(len(vertices))

    for i, vertex in enumerate(vertices):
        current_plate_id = plate_assignment[i]
        if current_plate_id == 0:
            continue

        neighbors = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)
        current_plate = plates[current_plate_id-1]

        for neighbor_idx in neighbors:
            neighbor_plate_id = plate_assignment[neighbor_idx]

            if neighbor_plate_id != current_plate_id and neighbor_plate_id != 0:
                # Simplified boundary interaction
                neighbor_plate = plates[neighbor_plate_id-1]

                # Elevation adjustment based on plate movement
                boundary_effects[i] += 0.00001 * step_size
                boundary_effects[neighbor_idx] -= 0.001 * step_size

                # Pressure changes at plate boundaries
                pressure_diff = current_plate.pressure - neighbor_plate.pressure
                pressure_changes[i] += pressure_diff * 0.01
                pressure_changes[neighbor_idx] -= pressure_diff * 0.01
                
                # Humidity transfer between plates
                humidity_diff = current_plate.humidity - neighbor_plate.humidity
                humidity_changes[i] -= humidity_diff * 0.01
                humidity_changes[neighbor_idx] += humidity_diff * 0.01
                
                # Update plate boundary pressure effects
                current_plate.pressure_systems['boundary'] += pressure_diff * 0.005
                neighbor_plate.pressure_systems['boundary'] -= pressure_diff * 0.005

                # Temperature changes at plate boundaries
                temp_diff = current_plate.temperature - neighbor_plate.temperature
                temperature_changes[i] += temp_diff * 0.01
                temperature_changes[neighbor_idx] -= temp_diff * 0.01

    # Apply boundary effects
    elevations += boundary_effects

    # Apply pressure and temperature changes
    for plate in plates:
        plate_vertices = list(plate.vertices)
        if plate_vertices:
            avg_pressure_change = np.mean(pressure_changes[plate_vertices])
            avg_temp_change = np.mean(temperature_changes[plate_vertices])
            avg_humidity_change = np.mean(humidity_changes[plate_vertices])

            # Update plate properties (dampened changes)
            plate.pressure += avg_pressure_change * 0.1
            plate.temperature += avg_temp_change * 0.1
            plate.humidity += avg_humidity_change * 0.1
            plate.humidity = np.clip(plate.humidity, 10, 90)
            plate.move(days_elapsed, day_of_year, hour_of_day, active_storms, 
                  elevations, water_fraction, vertices)

            # Plate speed affected by temperature (warmer plates move faster)
            plate.speed_cm_yr *= (1 + 0.01 * (plate.temperature - 15))

    # Smooth elevations
    smoothed_elevations = np.zeros_like(elevations)
    for i in range(len(vertices)):
        neighbor_indices = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)
        neighbor_elevations = [elevations[j] for j in neighbor_indices]
        smoothed_elevations[i] = np.mean([elevations[i]] + neighbor_elevations)
        if elevations[i] > 0:  # Only erode land
            lat, lon = cartesian_to_lat_lon(*vertices[i])
            slope = calculate_slope(vertices, faces, i)  # Implement slope calculation (see below)
            erosion = calculate_erosion(
                elevations[i], rainfall[i], humidity[i], 
                temperatures[i], slope, wind_speed=5.0  # Default wind speed
            )
            elevations[i] -= erosion * days_elapsed / 365

    elevations = smoothed_elevations

    # Update vertex positions with new elevations
    norm_vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    vertices = norm_vertices * (PLANET_RADIUS_KM + elevations[:, np.newaxis])

    # Reassign plate assignments based on new positions
    new_plate_assignment = np.zeros_like(plate_assignment)
    for plate in plates:
        plate.vertices.clear()

    for i, vertex in enumerate(vertices):
        min_dist = float('inf')
        closest_plate = None

        for plate in plates:
            dist = np.linalg.norm(vertex - plate.center_point)
            if dist < min_dist:
                min_dist = dist
                closest_plate = plate

        if closest_plate:
            new_plate_assignment[i] = closest_plate.plate_id
            closest_plate.vertices.add(i)

    return vertices, plates, new_plate_assignment, elevations

def calculate_erosion(elevation, rainfall, humidity, temperature, slope, wind_speed):
    """Calculate erosion (mm/year) based on environmental factors.
    
    Args:
        elevation (float): Meters above sea level.
        rainfall (float): Annual rainfall (mm).
        humidity (float): Relative humidity (%).
        temperature (float): Annual mean temperature (°C).
        slope (float): Terrain steepness (radians).
        wind_speed (float): Annual mean wind speed (m/s).
    
    Returns:
        float: Erosion rate (mm/year).
    """
    # Water erosion (rainfall + slope-dependent)
    water_erosion = rainfall * 0.0001 * (1 + np.tan(slope))  # ~0.1 mm/yr per 1000mm rain
    
    # Wind erosion (humidity-dependent)
    wind_erosion = wind_speed**2 * (1 - humidity/100) * 0.001  # ~1 mm/yr at 10 m/s in deserts
    
    # Thermal weathering (freeze-thaw cycles)
    freeze_thaw_cycles = max(0, np.abs(temperature - 0) / 10)  # Peaks near 0°C
    thermal_erosion = freeze_thaw_cycles * 0.05
    
    # Total erosion (only on land)
    if elevation > 0:
        return water_erosion + wind_erosion + thermal_erosion
    return 0
