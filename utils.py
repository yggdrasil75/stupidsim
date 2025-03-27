try:
    from utils_cpp import (
        calculate_mean_molecular_weight as cpp_calculate_mean_molecular_weight,
        determine_surface_type as cpp_determine_surface_type,
        cartesian_to_lat_lon as cpp_cartesian_to_lat_lon,
        lat_lon_to_cartesian as cpp_lat_lon_to_cartesian,
        haversine_distance as cpp_haversine_distance,
        calculate_bearing_math as cpp_calculate_bearing_math,
        calculate_bearing as cpp_calculate_bearing
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ utils not available, falling back to Python implementations")

from functools import lru_cache, wraps
import math
import time
import numpy as np

from globals import ATMOSPHERIC_COMPOSITION, MOLECULAR_WEIGHTS, PLANET_RADIUS_KM, function_times

# Timing decorator remains the same
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global function_times
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if func.__name__ not in function_times:
            function_times[func.__name__] = []
        function_times[func.__name__].append(elapsed_time)

        return result
    return wrapper

# Use C++ implementations if available, otherwise fall back to Python
if CPP_AVAILABLE:
    @lru_cache(maxsize=None)
    def calculate_mean_molecular_weight(humidity):
        return cpp_calculate_mean_molecular_weight(humidity)
    
    @lru_cache(maxsize=None)
    def determine_surface_type(elevation, temperature, water_fraction):
        return cpp_determine_surface_type(elevation, temperature, water_fraction)
    
    @lru_cache(maxsize=None)
    def cartesian_to_lat_lon(x, y, z):
        return cpp_cartesian_to_lat_lon(x, y, z)
    
    @lru_cache(maxsize=None)
    def lat_lon_to_cartesian(lat, lon, radius):
        return cpp_lat_lon_to_cartesian(lat, lon, radius)
    
    def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
        return cpp_haversine_distance(lat1, lon1, lat2, lon2, radius)
    
    @lru_cache(maxsize=None)
    def calculate_bearing_math(lat1, lon1, lat2, lon2):
        return cpp_calculate_bearing_math(lat1, lon1, lat2, lon2)
    
    @timing_decorator
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """Vectorized version using numpy arrays"""
        return cpp_calculate_bearing(
            np.asarray(lat1), 
            np.asarray(lon1), 
            np.asarray(lat2), 
            np.asarray(lon2)
        )



# from functools import lru_cache, wraps
# import math
# import time
# import numpy as np

# from globals import ATMOSPHERIC_COMPOSITION, MOLECULAR_WEIGHTS, PLANET_RADIUS_KM, function_times


# def timing_decorator(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         global function_times
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
        
#         # Record the time
#         if func.__name__ not in function_times:
#             function_times[func.__name__] = []
#         function_times[func.__name__].append(elapsed_time)

#         return result
#     return wrapper

# def print_function_times():
#     print("Function Execution Times:")
#     for func_name, times in function_times.items():
#         print(f"{func_name}: {sum(times):.4f}s over {len(times)} calls, "
#               f"average {sum(times)/len(times):.4f}s per call")

# @lru_cache(maxsize=None)
# def calculate_mean_molecular_weight(humidity):
#     """Calculate mean molecular weight of atmosphere based on humidity."""
#     adjusted_composition = ATMOSPHERIC_COMPOSITION.copy()
#     adjusted_composition['H2O'] = (humidity/100) * 0.04
    
#     total = 0
#     weighted_sum = 0
#     for gas, fraction in adjusted_composition.items():
#         weighted_sum += fraction * MOLECULAR_WEIGHTS[gas]
#         total += fraction
    
#     if total < 1:
#         weighted_sum += (1 - total) * MOLECULAR_WEIGHTS['N2']
    
#     return weighted_sum

# @lru_cache(maxsize=None)
# def determine_surface_type(elevation, temperature, water_fraction):
#     """Determine surface type for albedo calculation."""
#     if water_fraction > 0.9:
#         return 'water'
#     elif temperature < -5 and water_fraction > 0.1:  # Cold and wet = ice
#         return 'ice'
#     elif elevation < 0:  # Underwater
#         return 'water'
#     elif water_fraction > 0.3:  # Wet land
#         return 'forest' if temperature > 10 else 'grassland'
#     elif elevation > 4000:  # High altitude
#         return 'ice' if temperature < 0 else 'rock'
#     elif temperature > 30 and water_fraction < 0.1:  # Hot and dry
#         return 'desert'
#     else:
#         return 'grassland'

# @lru_cache(maxsize=None)
# def cartesian_to_lat_lon(x, y, z):
#     """Convert cartesian coordinates to latitude/longitude."""
#     lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
#     lon = np.degrees(np.arctan2(y, x))
#     return lat, lon

# @lru_cache(maxsize=None)
# def lat_lon_to_cartesian(lat, lon, radius):
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     x = radius * np.cos(lat_rad) * np.cos(lon_rad)
#     y = radius * np.cos(lat_rad) * np.sin(lon_rad)
#     z = radius * np.sin(lat_rad)
#     return x, y, z

# #@lru_cache(maxsize=None)
# def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
#     """Calculate great-circle distance between two points on a sphere."""
#     lat1_rad = np.radians(lat1)
#     lon1_rad = np.radians(lon1)
#     lat2_rad = np.radians(lat2)
#     lon2_rad = np.radians(lon2)

#     dlon = lon2_rad - lon1_rad
#     dlat = lat2_rad - lat1_rad

#     a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

#     distance = radius * c
#     return distance

#@lru_cache(maxsize=None)
def find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km):
    """Find neighbors within a certain distance on the sphere."""
    neighbors = set()
    center = vertices[vertex_idx]
    # First find direct face-connected neighbors
    for face in faces:
        if vertex_idx in face:
            for v in face:
                if v != vertex_idx:
                    neighbors.add(v)

    # Then check distance for all vertices
    final_neighbors = []
    for v in neighbors:
        # Calculate spherical distance
        lat1, lon1 = cartesian_to_lat_lon(*center)
        lat2, lon2 = cartesian_to_lat_lon(*vertices[v])
        dist = haversine_distance(lat1, lon1, lat2, lon2)

        if dist <= max_distance_km:
            final_neighbors.append(v)

    return final_neighbors

# #@lru_cache(maxsize=None)
def calculate_slope(vertices, faces, vertex_idx):
    """Estimate terrain slope (radians) at a vertex using neighboring faces."""
    neighbors = find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km=100)
    if not neighbors:
        return 0
  
    # Fit a plane to neighboring vertices
    points = vertices[neighbors + [vertex_idx]]
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]  # Plane normal vector
    
    # Slope = angle between normal and radial vector
    radial_vector = vertices[vertex_idx] / np.linalg.norm(vertices[vertex_idx])
    return np.arccos(np.clip(np.dot(normal, radial_vector), -1, 1))

# @lru_cache(maxsize=None)
# def calculate_bearing_math(lat1, lon1, lat2, lon2):
#     lat1_rad = math.radians(lat1)
#     lon1_rad = math.radians(lon1)
#     lat2_rad = math.radians(lat2)
#     lon2_rad = math.radians(lon2)

#     # Calculate the difference in longitudes
#     dLon = lon2_rad - lon1_rad

#     # Calculate bearing using the formula:
#     # θ = atan2(sin(Δlong)*cos(lat2), cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))
#     x = math.sin(dLon) * math.cos(lat2_rad)
#     y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
#         (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dLon)))

#     # Calculate the initial bearing in radians
#     initial_bearing = math.atan2(x, y)

#     # Convert from radians to degrees (0-360)
#     initial_bearing_deg = math.degrees(initial_bearing)
#     compass_bearing = (initial_bearing_deg + 360) % 360

#     return compass_bearing

# @timing_decorator
# @lru_cache(maxsize=None)
# def calculate_bearing(lat1, lon1, lat2, lon2):
#     """
#     Vectorized version of calculate_bearing for numpy arrays.
#     All inputs should be numpy arrays of the same shape.
#     """
#     lat1_rad = np.radians(lat1)
#     lon1_rad = np.radians(lon1)
#     lat2_rad = np.radians(lat2)
#     lon2_rad = np.radians(lon2)
    
#     dLon = lon2_rad - lon1_rad
    
#     x = np.sin(dLon) * np.cos(lat2_rad)
#     y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dLon)
    
#     initial_bearing = np.arctan2(x, y)
#     compass_bearing = (np.degrees(initial_bearing) + 360) % 360
    
#     return compass_bearing
