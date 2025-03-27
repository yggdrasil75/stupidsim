
import numpy as np

from globals import ATMOSPHERIC_COMPOSITION, MOLECULAR_WEIGHTS, PLANET_RADIUS_KM


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

def cartesian_to_lat_lon(x, y, z):
    """Convert cartesian coordinates to latitude/longitude."""
    lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def lat_lon_to_cartesian(lat, lon, radius):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

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
