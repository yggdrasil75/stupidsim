import numpy as np

from globals import SEA_LEVEL_PRESSURE_HPA
from pressure import calculate_pressure, calculate_wind_patterns, is_upwind
from utils import cartesian_to_lat_lon, find_spherical_neighbors, haversine_distance


def calculate_humidity(pressure, temperature, water_fraction):
    """Calculate relative humidity based on pressure, temperature and nearby water."""
    # Simplified humidity calculation
    # saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))

    # actual vapor pressure (hPa)
    e = es * water_fraction * (pressure / SEA_LEVEL_PRESSURE_HPA)

    # relative humidity (%)
    rh = 100 * (e / es)
    return np.clip(rh, 0, 100)

def calculate_rainfall(lat, lon, elevation, temperature, pressure, humidity, vertices, vertex_idx, faces, elevations, max_distance_km=1000):
    """Calculate rainfall based on pressure systems, humidity, and topography."""
    # Calculate pressure gradient
    neighbors = find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km)
    neighbor_pressures = [calculate_pressure(elevations[n], temperature) for n in neighbors]
    avg_neighbor_pressure = np.mean(neighbor_pressures) if neighbor_pressures else pressure
    pressure_gradient = (pressure - avg_neighbor_pressure) / max_distance_km  # hPa/km
    
    # Calculate wind direction with improved Coriolis effect
    wind_dir = calculate_wind_patterns(lat, elevation, 0, pressure_gradient)

    # Calculate water availability (from nearby water bodies)
    water_availability = 0
    for n in neighbors:
        n_elevation = elevations[n] # ADDED elevations
        if n_elevation < 0:  # Water body
            n_lat, n_lon = cartesian_to_lat_lon(*vertices[n])
            dist = haversine_distance(lat, lon, n_lat, n_lon)
            water_availability += max(0, 1 - dist/500)  # Water influence up to 500km

    # Base precipitation based on pressure system
    if pressure < SEA_LEVEL_PRESSURE_HPA - 10:  # Low pressure system - more rain
        base_rain = 150
    elif pressure > SEA_LEVEL_PRESSURE_HPA + 10:  # High pressure system - less rain
        base_rain = 20
    else:  # Normal pressure
        base_rain = 80

    # Humidity effect
    humidity_effect = humidity * 1.5  # More humidity = more potential rain

    # Orographic precipitation
    orographic_effect = 0
    if elevation > 0:  # Only over land
        # Find highest point in wind direction
        max_upwind_elev = 0
        for n in neighbors:
            n_lat, n_lon = cartesian_to_lat_lon(*vertices[n])
            if is_upwind(lat, lon, n_lat, n_lon, wind_dir):
                max_upwind_elev = max(max_upwind_elev, elevations[n]) # ADDED elevations

        if elevation > max_upwind_elev:  # We're on windward side
            orographic_effect = min(100, elevation / 20)  # 5mm per 100m
        else:  # Leeward side - rain shadow
            orographic_effect = -50

    # Temperature effect (more evaporation when warmer - leading to more moisture in air and potentially rain)
    temp_effect = max(0, temperature - 10)  # More rain above 10°C

    # Simplified Convection Effect - Non-Hydrostatic Consideration
    convection_effect = 0
    if temperature > 25 and lat < 60 and lat > -60 and elevation < 2000: # Convection more likely in warmer, lower latitude, lower altitude areas
        convection_effect = (temperature - 25) * 1.5  # Increase rain with higher temps above 25C

    # Combine all effects
    rainfall_val = (base_rain + orographic_effect + temp_effect +
               water_availability * 50 + humidity_effect + convection_effect) # ADD convection_effect
    rainfall_val = max(0, min(300, rainfall_val))  # Cap between 0-300mm

    #print(f"Vertex {vertex_idx}: Pressure={pressure:.2f}, P_grad={pressure_gradient:.2f}, WindDir={wind_dir}, WaterAvail={water_availability:.2f}, BaseRain={base_rain}, Orographic={orographic_effect:.2f}, TempEff={temp_effect:.2f}, HumidEff={humidity_effect:.2f}, Rainfall={rainfall_val:.2f}") # PRINT STATEMENT
    return rainfall_val
