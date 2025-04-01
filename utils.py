
from functools import lru_cache, wraps
import math
import threading
import time
import numpy as np

from globals import ATMOSPHERIC_COMPOSITION, MOLECULAR_WEIGHTS, PLANET_RADIUS_KM, function_times

_GLOBAL_DISTANCE_CACHE = []
_CACHE_INITIALIZED = False
_CACHE_LOCK = threading.Lock() # Lock for thread-safe access to the cache

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global function_times
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Record the time
        if func.__name__ not in function_times:
            function_times[func.__name__] = []
        function_times[func.__name__].append(elapsed_time)

        return result
    return wrapper

@timing_decorator
@lru_cache(maxsize=None)
def calculate_mean_molecular_weight(humidity):
    """Calculate mean molecular weight of atmosphere based on humidity."""
    adjusted_composition = ATMOSPHERIC_COMPOSITION.copy()
    adjusted_composition['H2O'] = (humidity/100) * 0.04
    
    total = 0
    weighted_sum = 0
    for gas, fraction in adjusted_composition.items():
        weighted_sum += fraction * MOLECULAR_WEIGHTS[gas]
        total += fraction
    
    if total < 1:
        weighted_sum += (1 - total) * MOLECULAR_WEIGHTS['N2']
    
    return weighted_sum

@timing_decorator
@lru_cache(maxsize=None)
def determine_surface_type(elevation, temperature, water_fraction):
    """Determine surface type for albedo calculation."""
    if water_fraction > 0.9:
        return 'water'
    elif temperature < -5 and water_fraction > 0.1:  # Cold and wet = ice
        return 'ice'
    elif elevation < 0:  # Underwater
        return 'water'
    elif water_fraction > 0.3:  # Wet land
        return 'forest' if temperature > 10 else 'grassland'
    elif elevation > 4000:  # High altitude
        return 'ice' if temperature < 0 else 'rock'
    elif temperature > 30 and water_fraction < 0.1:  # Hot and dry
        return 'desert'
    else:
        return 'grassland'

@timing_decorator
@lru_cache(maxsize=None)
def cartesian_to_lat_lon(x, y, z):
    """Convert cartesian coordinates to latitude/longitude."""
    lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

@timing_decorator
@lru_cache(maxsize=None)
def lat_lon_to_cartesian(lat, lon, radius):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

@timing_decorator
@lru_cache(maxsize=None)
def haversine_distance(lat1, lon1, lat2, lon2, radius=PLANET_RADIUS_KM):
    """Calculate great-circle distance between two points on a sphere."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = radius * c
    return distance

@timing_decorator
def _calculate_distances_for_vertex_range(vertices, vertex_indices, distances_list):
    """Calculates distances for a subset of vertices and appends to a list."""
    for i in vertex_indices:
        for j in range(i + 1, len(vertices)): # Avoid redundant calculations and self-distances
            center = vertices[i]
            neighbor_vertex = vertices[j]

            lat1, lon1 = cartesian_to_lat_lon(*center)
            lat2, lon2 = cartesian_to_lat_lon(*neighbor_vertex)
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances_list.append((dist, (i, j)))

@timing_decorator
def initialize_distance_cache(vertices, num_threads=4): # Added num_threads parameter
    """Calculates and sorts all pairwise distances between vertices using threads."""
    global _GLOBAL_DISTANCE_CACHE, _CACHE_INITIALIZED, _CACHE_LOCK
    if _CACHE_INITIALIZED:
        return  # Already initialized

    distances = []
    num_vertices = len(vertices)
    threads = []
    vertices_per_thread = num_vertices // num_threads + (1 if num_vertices % num_threads != 0 else 0) # Distribute vertices roughly evenly

    for thread_id in range(num_threads):
        start_index = thread_id * vertices_per_thread
        end_index = min((thread_id + 1) * vertices_per_thread, num_vertices)
        vertex_indices_for_thread = range(start_index, end_index)
        if not vertex_indices_for_thread: # Skip if no vertices for this thread
            continue
        thread = threading.Thread(target=_calculate_distances_for_vertex_range, args=(vertices, vertex_indices_for_thread, distances))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join() # Wait for all threads to complete

    distances.sort(key=lambda item: item[0]) # Sort by distance after all threads are done
    with _CACHE_LOCK: # Acquire lock before updating global cache
        _GLOBAL_DISTANCE_CACHE = distances
        _CACHE_INITIALIZED = True

@timing_decorator
def find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km):
    """
    Find neighbors within a certain distance using pre-computed and sorted distances.
    """
    global _GLOBAL_DISTANCE_CACHE, _CACHE_INITIALIZED

    if not _CACHE_INITIALIZED:
        initialize_distance_cache(vertices)

    neighbors = set()
    # First find direct face-connected neighbors (still needed for initial neighbor set)
    for face in faces:
        if vertex_idx in face:
            for v in face:
                if v != vertex_idx:
                    neighbors.add(v)

    final_neighbors = []
    for dist, vertex_pair in _GLOBAL_DISTANCE_CACHE:
        if dist > max_distance_km:
            break # Since distances are sorted, we can stop searching

        v1, v2 = vertex_pair
        if v1 == vertex_idx:
            if v2 in neighbors: # Only consider face-connected neighbors initially
                final_neighbors.append(v2)
        elif v2 == vertex_idx:
            if v1 in neighbors: # Only consider face-connected neighbors initially
                final_neighbors.append(v1)

    return final_neighbors

#@lru_cache(maxsize=None)
@timing_decorator
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

@timing_decorator
@lru_cache(maxsize=None)
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Vectorized version of calculate_bearing for numpy arrays.
    All inputs should be numpy arrays of the same shape.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dLon = lon2_rad - lon1_rad
    
    x = np.sin(dLon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dLon)
    
    initial_bearing = np.arctan2(x, y)
    compass_bearing = (np.degrees(initial_bearing) + 360) % 360
    
    return compass_bearing

def print_function_times():
    print("Function Execution Times:")
    for func_name, times in function_times.items():
        print(f"{func_name}: {sum(times):.4f}s over {len(times)} calls, "
              f"average {sum(times)/len(times):.4f}s per call")