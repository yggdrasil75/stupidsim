import random
import numpy as np

from globals import PLANET_RADIUS_KM
from utils import cartesian_to_lat_lon, haversine_distance


class StormSystem:
    def __init__(self, center_lat, center_lon, pressure_anomaly, radius_km=500, movement_speed_km_day=50):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.pressure_anomaly = pressure_anomaly  # Negative for cyclones, positive for anticyclones
        self.radius_km = radius_km
        self.movement_speed_km_day = movement_speed_km_day
        self.movement_direction = np.random.uniform(0, 360)  # Degrees from north
        self.age_days = 0
        self.max_lifetime_days = np.random.uniform(5, 15)  # Storm lifespan
        self.intensity_change_rate = np.random.uniform(-0.05, 0.05)  # Daily intensity change
        
    def update(self, vertices, elevations, day_elapsed=1):
        """Update storm position and intensity over time."""
        self.age_days += day_elapsed
        
        # Change intensity over time
        self.pressure_anomaly *= (1 + self.intensity_change_rate * day_elapsed)
        
        # Move storm
        distance_km = self.movement_speed_km_day * day_elapsed
        self.center_lat += (distance_km / 111) * np.cos(np.radians(self.movement_direction))
        self.center_lon += (distance_km / (111 * np.cos(np.radians(self.center_lat)))) * np.sin(np.radians(self.movement_direction))
        
        # Change direction slightly (random walk)
        self.movement_direction += np.random.normal(0, 10)
        
        # Adjust movement based on global wind patterns (steering flow)
        if abs(self.center_lat) < 30:
            # Trade winds - generally westward
            self.movement_direction = 270 + np.random.normal(0, 20)
        elif abs(self.center_lat) < 60:
            # Westerlies - generally eastward
            self.movement_direction = 90 + np.random.normal(0, 20)
        else:
            # Polar easterlies
            self.movement_direction = 270 + np.random.normal(0, 30)
            
        # Storm dissipation
        if self.age_days > self.max_lifetime_days:
            self.pressure_anomaly *= 0.9  # Rapidly dissipate
            
    def calculate_pressure_effect(self, lat, lon):
        """Calculate pressure effect at given coordinates."""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon, PLANET_RADIUS_KM)
        
        if distance > self.radius_km:
            return 0
            
        # Gaussian influence
        influence = np.exp(-(distance**2)/(2*(self.radius_km/3)**2))
        return self.pressure_anomaly * influence


def generate_storm_systems(vertices, elevations, day_of_year, num_storms=5):
    """Generate new storm systems based on current conditions."""
    storms = []
    
    # More storms during seasonal extremes
    seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    for _ in range(int(num_storms * seasonal_factor)):
        # Prefer certain latitudes for storm formation
        if random.random() < 0.7:  # 70% chance in mid-latitudes
            lat = np.random.uniform(-60, 60)
            if abs(lat) < 30:  # Tropical storms
                pressure_anomaly = np.random.uniform(-15, -5)  # Strong low pressure
                radius = np.random.uniform(300, 800)
                speed = np.random.uniform(30, 60)
            else:  # Extratropical cyclones
                pressure_anomaly = np.random.uniform(-25, -10)
                radius = np.random.uniform(500, 1500)
                speed = np.random.uniform(50, 100)
        else:  # 30% chance elsewhere
            lat = np.random.uniform(-90, 90)
            pressure_anomaly = np.random.uniform(-10, 10)
            radius = np.random.uniform(200, 1000)
            speed = np.random.uniform(20, 80)
            
        lon = np.random.uniform(-180, 180)
        
        # Prefer forming over water
        nearest_vertex = min(range(len(vertices)), 
                           key=lambda i: haversine_distance(lat, lon, *cartesian_to_lat_lon(*vertices[i])))
        if elevations[nearest_vertex] < 0:  # Water
            pressure_anomaly *= 1.5  # Stronger over water
            
        storms.append(StormSystem(lat, lon, pressure_anomaly, radius, speed))
        
    return storms

def update_storm_systems(active_storms, vertices, elevations, day_elapsed):
    """Update all active storm systems."""
    # Update existing storms
    for storm in active_storms[:]:
        storm.update(vertices, elevations, day_elapsed)
        
        # Remove dissipated storms
        if abs(storm.pressure_anomaly) < 1:  # Fully dissipated
            active_storms.remove(storm)
            
    return active_storms
