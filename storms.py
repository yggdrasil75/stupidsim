import random
import numpy as np

from globals import PLANET_RADIUS_KM
from utils import calculate_bearing, cartesian_to_lat_lon, find_spherical_neighbors, haversine_distance


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
        
    def update(self, vertices, elevations, temperatures, pressures, faces, day_elapsed=1):
        # Use temperature gradient to steer storm
        nearest_vertex = self._find_nearest_vertex(vertices)
        temp = temperatures[nearest_vertex]
        
        # Move toward warmer areas (for tropical systems)
        if abs(self.center_lat) < 30 and temp > 26:
            self.movement_direction = self._calculate_steering_flow(pressures, vertices, faces)
        
        # Intensify over warm water
        if elevations[nearest_vertex] < 0 and temp > 26:
            self.pressure_anomaly *= 1.02  # 2% intensification per day
        
        # Weaken over land
        elif elevations[nearest_vertex] > 0:
            self.pressure_anomaly *= 0.95  # 5% weakening per day
    
    def _find_nearest_vertex(self, vertices):
        # Convert all vertices to lat/lon first
        vertex_coords = np.array([cartesian_to_lat_lon(*v) for v in vertices])
        vertex_lats = vertex_coords[:, 0]
        vertex_lons = vertex_coords[:, 1]
        
        # Calculate all distances at once
        distances = haversine_distance(
            self.center_lat, self.center_lon,
            vertex_lats, vertex_lons
        )
        
        return np.argmin(distances)
            
    def calculate_pressure_effect(self, lat, lon):
        """Calculate pressure effect at given coordinates."""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon, PLANET_RADIUS_KM)
        
        if distance > self.radius_km:
            return 0
            
        # Gaussian influence
        influence = np.exp(-(distance**2)/(2*(self.radius_km/3)**2))
        return self.pressure_anomaly * influence
    
    def calculate_wind_field(self, lat, lon):
        """Calculate wind vector at given coordinates"""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        
        # Add cyclonic rotation (counter-clockwise in NH, clockwise in SH)
        bearing = calculate_bearing(self.center_lat, self.center_lon, lat, lon)
        if distance < self.radius_km:
            rotation_dir = 1 if self.center_lat >= 0 else -1  # NH vs SH
            wind_dir = (bearing + 90 * rotation_dir) % 360
            
            # Wind speed based on pressure gradient and distance from center
            max_speed = -self.pressure_anomaly * 3  # Convert pressure to wind speed
            speed = max_speed * (distance/self.radius_km) * np.exp(1 - (distance/self.radius_km))
            
            return wind_dir, speed
        return None
    
    def is_in_rain_band(self, lat, lon):
        """Check if coordinates are within the storm's rain bands"""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        
        # Rain bands extend 1.5x the radius of the pressure anomaly
        if distance > self.radius_km * 1.5:
            return False
            
        # Spiral rain band pattern
        angle = np.degrees(np.arctan2(lon - self.center_lon, lat - self.center_lat))
        band_width = 30  # degrees
        return (distance % (self.radius_km/3)) < (self.radius_km/10)
    
    def get_rainfall_intensity(self, lat, lon):
        """Calculate rainfall intensity based on distance from storm center"""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        
        if distance > self.radius_km * 1.5:
            return 0
            
        # Max rainfall near the eyewall (0.8-1.2x radius)
        if 0.8 * self.radius_km < distance < 1.2 * self.radius_km:
            return min(300, abs(self.pressure_anomaly) * 5)  # Cap at 300mm
            
        # Decrease with distance
        return max(0, abs(self.pressure_anomaly) * 3 * (1 - distance/(self.radius_km*1.5)))
    
    def _calculate_steering_flow(self, pressures, vertices, faces):
        # Find nearest vertex
        nearest_idx = self._find_nearest_vertex(vertices)
        lat, lon = cartesian_to_lat_lon(*vertices[nearest_idx])
        
        # Find neighboring vertices within 500km
        neighbors = find_spherical_neighbors(vertices, faces, nearest_idx, 500)
        
        if not neighbors:
            return self.movement_direction  # No change if no neighbors
        
        # Calculate pressure gradient
        neighbor_pressures = [pressures[n] for n in neighbors]
        avg_pressure = np.mean(neighbor_pressures)
        pressure_diff = pressures[nearest_idx] - avg_pressure
        
        # Get coordinates of neighbors
        neighbor_coords = np.array([cartesian_to_lat_lon(*vertices[n]) for n in neighbors])
        neighbor_lats = neighbor_coords[:, 0]
        neighbor_lons = neighbor_coords[:, 1]
        
        # Calculate centroid of lower pressure area
        if pressure_diff > 0:  # We're in higher pressure - move toward lower
            low_pressure_neighbors = [n for n in neighbors if pressures[n] < avg_pressure]
            if low_pressure_neighbors:
                target_lat = np.mean([cartesian_to_lat_lon(*vertices[n])[0] for n in low_pressure_neighbors])
                target_lon = np.mean([cartesian_to_lat_lon(*vertices[n])[1] for n in low_pressure_neighbors])
            else:
                return self.movement_direction
        else:  # Already in low pressure - move down gradient
            pressure_grad_x = np.mean(np.sin(np.radians(neighbor_lons)) * (pressures[neighbors] - pressures[nearest_idx]))
            pressure_grad_y = np.mean(np.cos(np.radians(neighbor_lats)) * (pressures[neighbors] - pressures[nearest_idx]))
            
            # Move perpendicular to gradient (geostrophic flow)
            if self.center_lat >= 0:  # Northern hemisphere - counterclockwise
                target_lon = lon + pressure_grad_y * 100
                target_lat = lat - pressure_grad_x * 100
            else:  # Southern hemisphere - clockwise
                target_lon = lon - pressure_grad_y * 100
                target_lat = lat + pressure_grad_x * 100
        
        # Calculate bearing to target
        new_direction = calculate_bearing(lat, lon, target_lat, target_lon)
        
        # Smooth direction change (max 15 degrees per day)
        direction_change = (new_direction - self.movement_direction + 360) % 360
        if direction_change > 180:
            direction_change -= 360
        
        max_change = 15 * (self.radius_km/500)  # Larger storms turn slower
        actual_change = np.clip(direction_change, -max_change, max_change)
        
        return (self.movement_direction + actual_change) % 360


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

