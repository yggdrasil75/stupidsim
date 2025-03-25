# utils.py
import numpy as np
import math
from constants import PLANET_RADIUS_KM, AXIAL_TILT_DEGREES, SEA_LEVEL_PRESSURE_HPA, GAS_CONSTANT, GRAVITY, ATMOSPHERIC_LAYERS

def generate_icosphere(subdivisions=3, radius=1.0):
    """Generate an icosphere mesh with given number of subdivisions."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0

    # Create initial icosahedron vertices
    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)
    ]

    # Normalize vertices to unit sphere
    vertices = [np.array(v)/np.linalg.norm(v) for v in vertices]

    # Create initial icosahedron faces
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]

    # Subdivide the mesh
    for _ in range(subdivisions):
        new_faces = []
        edge_vertices = {}

        for face in faces:
            # Get edge vertices
            edge_midpoints = []
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                key = tuple(sorted((v1, v2)))
                if key not in edge_vertices:
                    # Create new vertex at midpoint
                    mid = (vertices[v1] + vertices[v2]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_vertices[key] = len(vertices)
                    vertices.append(mid)
                edge_midpoints.append(edge_vertices[key])

            # Create 4 new faces
            a, b, c = face
            d, e, f = edge_midpoints
            new_faces.extend([
                (a, d, f),
                (d, b, e),
                (f, e, c),
                (d, e, f)
            ])

        faces = new_faces

    # Convert to numpy arrays
    vertices = np.array(vertices) * radius
    faces = np.array(faces)

    return vertices, faces

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

def calculate_pressure(elevation, temperature):
    """Calculate atmospheric pressure based on elevation and temperature."""
    # Using barometric formula with temperature consideration
    # P = P0 * exp(-elevation * g / (R * T))
    # Where:
    # P0 = sea level pressure (1013.25 hPa)
    # elevation in meters
    # g = gravity (9.81 m/s²)
    # R = specific gas constant for dry air (287.05 J/kg·K)
    # T = temperature in Kelvin

    # Convert elevation to meters and temperature to Kelvin
    elevation_m = elevation * 1000  # km to m
    temp_k = temperature + 273.15  # °C to K

    # Avoid division by zero for negative temperatures
    if temp_k <= 0:
        temp_k = 273.15  # Set to 0°C if temperature is below absolute zero

    pressure = SEA_LEVEL_PRESSURE_HPA * np.exp(-elevation_m * GRAVITY / (GAS_CONSTANT * temp_k))
    return pressure

def calculate_pressure_with_layers(elevation_km, surface_temp):
    """Calculate pressure considering atmospheric layers."""
    current_pressure = SEA_LEVEL_PRESSURE_HPA
    current_temp = surface_temp + 273.15  # Convert to Kelvin
    remaining_altitude = max(0, -elevation_km)  # Convert elevation to altitude (positive up)

    for layer in ATMOSPHERIC_LAYERS:
        layer_bottom, layer_top = layer["altitude_range"]
        gradient = layer["temp_gradient"]

        # Calculate how much of this layer we need to process
        layer_thickness = min(layer_top, remaining_altitude + layer_bottom) - layer_bottom
        if layer_thickness <= 0:
            continue

        # Calculate temperature at top of this segment
        temp_change = gradient * layer_thickness
        temp_top = current_temp + temp_change

        # Calculate pressure through this layer segment
        if gradient == 0:
            # Isothermal layer
            current_pressure *= np.exp(-GRAVITY * layer_thickness * 1000 /
                                     (GAS_CONSTANT * current_temp))
        else:
            # Non-isothermal layer
            current_pressure *= (temp_top / current_temp) ** (-GRAVITY / (gradient * 1000 * GAS_CONSTANT))

        current_temp = temp_top
        remaining_altitude -= layer_thickness

        if remaining_altitude <= 0:
            break

    return current_pressure

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

def calculate_solar_radiation(lat, lon, day_of_year, hour_of_day, elevation, pressure):
    """Calculate solar radiation at a given point considering axial tilt, time, and pressure."""
    solar_constant = 1361  # W/m^2

    # Calculate declination (seasonal variation due to axial tilt)
    axial_tilt_rad = np.radians(AXIAL_TILT_DEGREES)
    declination = axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

    # Calculate hour angle (time of day)
    hour_angle = np.radians((hour_of_day - 12) * 15)  # 15 degrees per hour

    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Calculate solar zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(declination) +
                 (np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)))

    # Only calculate radiation for daytime (cos_zenith > 0)
    if cos_zenith <= 0:
        return 0

    # Atmospheric absorption based on pressure
    pressure_ratio = pressure / SEA_LEVEL_PRESSURE_HPA
    air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - np.degrees(np.arccos(cos_zenith))) ** -1.6364)
    atmospheric_transmittance = (0.7 * pressure_ratio) ** air_mass

    # Elevation effect (thinner atmosphere at higher elevation)
    elevation_factor = 1 + (elevation / 10000)  # 10% increase per km

    # Total solar radiation
    radiation = solar_constant * cos_zenith * atmospheric_transmittance * elevation_factor
    return max(0, radiation)

def calculate_sun_direction(day_of_year, hour_of_day):
    """Calculate the direction vector to the sun based on time of year and time of day."""
    # Calculate declination (seasonal variation due to axial tilt)
    axial_tilt_rad = np.radians(AXIAL_TILT_DEGREES)
    declination = axial_tilt_rad * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

    # Calculate hour angle (time of day)
    hour_angle = np.radians(hour_of_day * 15)  # 15 degrees per hour

    # Calculate sun direction vector
    x = np.cos(declination) * np.cos(hour_angle)
    y = np.cos(declination) * np.sin(hour_angle)
    z = np.sin(declination)

    return np.array([x, y, z])

def calculate_solar_radiation_for_vertex(vertex, sun_direction, elevation, surface_pressure):
    """Calculate solar radiation considering atmospheric layers."""
    solar_constant = 1361  # W/m^2

    # Calculate atmospheric thickness based on elevation
    atmospheric_thickness = max(0, -elevation)  # Convert elevation to altitude

    # Calculate effective pressure (weighted average through atmosphere)
    # This is a simplified approach - more accurate would be to integrate through layers
    effective_pressure = surface_pressure * np.exp(-atmospheric_thickness / 8.5)  # Scale height approx

    # Rest of the calculation remains similar but uses effective_pressure
    surface_normal = vertex / np.linalg.norm(vertex)
    cos_zenith = np.dot(surface_normal, sun_direction)

    if cos_zenith <= 0:
        return 0

    pressure_ratio = effective_pressure / SEA_LEVEL_PRESSURE_HPA
    air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - np.degrees(np.arccos(cos_zenith))) ** -1.6364)
    atmospheric_transmittance = (0.7 * pressure_ratio) ** air_mass

    radiation = solar_constant * cos_zenith * atmospheric_transmittance
    return max(0, radiation)

def calculate_temperature_from_radiation(radiation, elevation, water_fraction, pressure):
    """Calculate temperature based on solar radiation and other factors."""
    # Base temperature from radiation
    base_temp = (radiation / 200) - 10  # Scale radiation to reasonable temps

    # Elevation effect (6.5°C per km)
    elevation_effect = -6.5 * (elevation / 1000)

    # Water moderating effect (water has higher heat capacity)
    water_moderation = water_fraction * 5  # Water makes temps more moderate

    # Pressure effect (low pressure systems are generally cooler)
    pressure_effect = (pressure - SEA_LEVEL_PRESSURE_HPA) * 0.02

    # Final temperature
    temperature = base_temp + elevation_effect + water_moderation + pressure_effect

    # Never below absolute zero
    return max(-273, temperature)

def calculate_temperature(lat, lon, day_of_year, hour_of_day, elevation, water_fraction, pressure):
    """Calculate temperature considering solar radiation, elevation, water bodies, and pressure."""
    # Calculate solar radiation with pressure effects
    radiation = calculate_solar_radiation(lat, lon, day_of_year, hour_of_day, elevation, pressure)

    # Base temperature from radiation
    base_temp = (radiation / 200) - 10  # Scale radiation to reasonable temps

    # Elevation effect (6.5°C per km)
    elevation_effect = -6.5 * (elevation / 1000)

    # Water moderating effect (water has higher heat capacity)
    water_moderation = water_fraction * 5  # Water makes temps more moderate

    # Pressure effect (low pressure systems are generally cooler)
    pressure_effect = (pressure - SEA_LEVEL_PRESSURE_HPA) * 0.02

    # Diurnal variation
    diurnal_variation = 10 * np.sin(np.radians(hour_of_day * 15))  # 15° per hour

    # Seasonal variation
    seasonal_variation = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

    # Final temperature
    temperature = (base_temp + elevation_effect + water_moderation +
                  pressure_effect + diurnal_variation + seasonal_variation)

    # Never below absolute zero
    return max(-273, temperature)

def calculate_temperature_with_altitude(surface_temp, elevation_km):
    """Calculate temperature at given altitude considering atmospheric layers."""
    current_temp = surface_temp
    altitude_km = max(0, -elevation_km)  # Convert elevation to altitude

    for layer in ATMOSPHERIC_LAYERS:
        layer_bottom, layer_top = layer["altitude_range"]
        gradient = layer["temp_gradient"]

        if altitude_km <= layer_bottom:
            continue

        # Calculate temperature through this layer
        thickness = min(layer_top, altitude_km) - layer_bottom
        current_temp += gradient * thickness

        if altitude_km <= layer_top:
            break

    return current_temp

def calculate_wind_patterns(lat, elevation, temperature_gradient, pressure_gradient):
    """Calculate prevailing wind direction based on latitude, elevation, and pressure gradient."""
    # Modified by elevation (mountains disrupt wind patterns)
    if elevation > 2000:
        return random.choice([0,1,2,3])  # Unpredictable in high mountains

    # Wind is influenced by both latitude patterns and pressure gradients
    lat_effect = 0
    if abs(lat) < 30:  # Trade winds
        if lat > 0: lat_effect = 1  # NE trade winds
        else: lat_effect = 3  # SE trade winds
    elif abs(lat) < 60:  # Westerlies
        lat_effect = 2  # From west
    else:  # Polar easterlies
        lat_effect = 0  # From east

    # Pressure gradient effect (wind flows from high to low pressure)
    if pressure_gradient > 1:  # Strong high pressure nearby
        return lat_effect
    elif pressure_gradient < -1:  # Strong low pressure nearby
        return (lat_effect + 2) % 4  # Reverse direction
    else:
        return lat_effect

def calculate_erosion(elevation, rainfall, humidity, wind_speed):
    """Calculate erosion based on rainfall, humidity, and wind."""
    # Water erosion
    water_erosion = rainfall * 0.001  # More rain = more erosion

    # Wind erosion (depends on humidity - drier areas have more wind erosion)
    wind_erosion = wind_speed * (1 - humidity/100) * 0.005

    # Total erosion (only applies to land above sea level)
    if elevation > 0:
        return water_erosion + wind_erosion
    return 0

def calculate_rainfall(lat, lon, elevation, temperature, pressure, humidity, vertices, vertex_idx, faces, elevations, max_distance_km=1000): # ADDED faces and elevations
    """Calculate rainfall based on pressure systems, humidity, and topography."""
    # Calculate pressure gradient
    neighbors = find_spherical_neighbors(vertices, faces, vertex_idx, max_distance_km) # ADDED faces
    neighbor_pressures = [calculate_pressure(elevations[n], temperature) for n in neighbors] # ADDED elevations
    avg_neighbor_pressure = np.mean(neighbor_pressures) if neighbor_pressures else pressure # Handle no neighbors
    pressure_gradient = pressure - avg_neighbor_pressure

    # Determine wind direction based on pressure gradient
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

def is_upwind(lat1, lon1, lat2, lon2, wind_dir):
    """Check if point 2 is upwind of point 1 based on wind direction."""
    # Simplified check based on wind direction categories
    if wind_dir == 0:  # East wind
        return lon2 < lon1
    elif wind_dir == 1:  # North wind
        return lat2 > lat1
    elif wind_dir == 2:  # West wind
        return lon2 > lon1
    else:  # South wind
        return lat2 < lat1