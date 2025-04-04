
from functools import lru_cache, wraps
import math
import threading
import time
import numpy as np

from globals import ATMOSPHERIC_COMPOSITION, MOLECULAR_WEIGHTS, PLANET_RADIUS_KM
from _icosphere import HaversineDistance, cartesianLatLon, findSphericalNeighbors, sphericalDistanceCartesian, latLonCartesian
from _icosphere import calculateSlope as calculate_slope


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
def calculate_slope(vertices, faces, vertex_idx):
    """Estimate terrain slope (radians) at a vertex using neighboring faces."""
    neighbors = findSphericalNeighbors(vertices, faces, vertex_idx, max_distance_km=100)
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
