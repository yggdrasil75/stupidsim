import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import RadioButtons, Slider
import math

PLANET_RADIUS_KM = 6371.0
ORBITAL_DISTANCE_AU = 1.0
AXIAL_TILT_DEGREES = 23.5
MIN_PLATE_SPEED_CM_YR = 1.0  # ~1 cm/year (slow moving plates)
MAX_PLATE_SPEED_CM_YR = 10.0  # ~10 cm/year (fast moving plates)
SEA_LEVEL_PRESSURE_HPA = 1013.25  # Standard atmospheric pressure at sea level
GAS_CONSTANT = 287.05  # Specific gas constant for dry air (J/kg·K)
GRAVITY = 9.81  # m/s²
cbar_obj = None  # Global variable to store the colorbar object
HADLEY_CELL_WIDTH = 30  # Degrees latitude
FERREL_CELL_WIDTH = 30  # Degrees latitude
POLAR_CELL_WIDTH = 30   # Degrees latitude
CORIOLIS_FACTOR = 0.0001  # Simplified Coriolis effect factor
# Atmospheric layers (altitude in km, temperature gradient in °C/km)
ATMOSPHERIC_LAYERS = [
    {"name": "Troposphere", "altitude_range": (0, 12), "temp_gradient": -6.5},
    {"name": "Stratosphere", "altitude_range": (12, 50), "temp_gradient": 0.1},
    {"name": "Mesosphere", "altitude_range": (50, 80), "temp_gradient": -2.8},
    {"name": "Thermosphere", "altitude_range": (80, 700), "temp_gradient": 0.0}
]
# Atmospheric composition constants (by volume)
ATMOSPHERIC_COMPOSITION = {
    'N2': 0.7808,  # Nitrogen
    'O2': 0.2095,  # Oxygen
    'Ar': 0.0093,  # Argon
    'CO2': 0.0004, # Carbon dioxide
    'H2O': 0.01,   # Water vapor (variable)
    'CH4': 1.8e-6, # Methane
    'O3': 7.0e-6   # Ozone (variable)
}
# Specific gas constants (J/kg·K)
GAS_CONSTANTS = {
    'N2': 296.80,
    'O2': 259.84,
    'Ar': 208.13,
    'CO2': 188.92,
    'H2O': 461.50,
    'CH4': 518.28,
    'O3': 173.21
}
# Molecular weights (kg/mol)
MOLECULAR_WEIGHTS = {
    'N2': 0.0280134,
    'O2': 0.0319988,
    'Ar': 0.039948,
    'CO2': 0.0440095,
    'H2O': 0.01801528,
    'CH4': 0.0160425,
    'O3': 0.0479982
}
# Greenhouse gas absorption coefficients (W/m² per kg/m²)
GREENHOUSE_ABSORPTION = {
    'CO2': 0.05,
    'H2O': 0.1,
    'CH4': 0.03,
    'O3': 0.15
}

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

def generate_initial_world_spherical(subdivisions=3, radius=PLANET_RADIUS_KM, num_plates=5, surface_pressures=None):
    """Generate initial world with spherical mesh and plates."""
    vertices, faces = generate_icosphere(subdivisions, radius)

    # Add random elevation to vertices using spherical harmonics for more natural distribution
    elevations = np.zeros(len(vertices))
    if surface_pressures is None:
        surface_pressures = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        # Convert to spherical coordinates
        lat, lon = cartesian_to_lat_lon(*vertex)

        # Create more interesting initial elevations using noise
        noise = (np.sin(lon * 2) * np.cos(lat * 3) +
                 np.sin(lon * 5) * np.cos(lat * 2)) * 5
        elevations[i] = noise + np.random.uniform(-2, 2)
        surface_pressures[i] = SEA_LEVEL_PRESSURE_HPA # Initialize pressure to sea level

    # Scale vertices with elevations
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis] * (radius + elevations[:, np.newaxis])

    # Create plates
    plates = []
    plate_assignment = np.zeros(len(vertices), dtype=int)

    # Generate random plate center locations on the sphere using Fibonacci sphere algorithm
    plate_centers = []
    for i in range(num_plates):
        y = 1 - (i / float(num_plates - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = math.pi * (3 - math.sqrt(5)) * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        plate_centers.append(np.array([x, y, z]) * PLANET_RADIUS_KM)

    for i in range(num_plates):
        center_point = plate_centers[i]

        # Create random movement direction (tangent to sphere)
        tangent = np.cross(center_point, np.random.randn(3))
        tangent = tangent / np.linalg.norm(tangent)

        plate = Plate(i + 1, -1, tangent)  # Note: speed is set in Plate.__init__
        plate.center_point = center_point.copy()
        plates.append(plate)

    # Assign vertices to nearest plate center
    for i, vertex in enumerate(vertices):
        min_dist = float('inf')
        closest_plate = None

        for plate in plates:
            dist = np.linalg.norm(vertex - plate.center_point)
            if dist < min_dist:
                min_dist = dist
                closest_plate = plate

        if closest_plate:
            plate_assignment[i] = closest_plate.plate_id
            closest_plate.vertices.add(i)

    return vertices, faces, plates, plate_assignment, elevations, surface_pressures

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

def simulate_plate_tectonics_spherical(vertices, faces, plates, plate_assignment, elevations, days_elapsed=30, max_neighbor_distance_km=1000, step_size=0.03):
    """Simulate plate tectonics on spherical mesh with pressure and temperature effects."""
    # Move plates
    for plate in plates:
        plate.move(days_elapsed)

    # Calculate boundary effects
    boundary_effects = np.zeros_like(elevations)
    pressure_changes = np.zeros(len(vertices))
    temperature_changes = np.zeros(len(vertices))

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
                current_plate = plates[current_plate_id-1]
                neighbor_plate = plates[neighbor_plate_id-1]

                # Elevation adjustment based on plate movement
                boundary_effects[i] += 0.001 * step_size
                boundary_effects[neighbor_idx] -= 0.001 * step_size

                # Pressure changes at plate boundaries
                pressure_diff = current_plate.pressure - neighbor_plate.pressure
                pressure_changes[i] += pressure_diff * 0.01
                pressure_changes[neighbor_idx] -= pressure_diff * 0.01

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

            # Update plate properties (dampened changes)
            plate.pressure += avg_pressure_change * 0.1
            plate.temperature += avg_temp_change * 0.1

            # Plate speed affected by temperature (warmer plates move faster)
            plate.speed_cm_yr *= (1 + 0.01 * (plate.temperature - 15))

    # Smooth elevations
    smoothed_elevations = np.zeros_like(elevations)
    for i in range(len(vertices)):
        neighbor_indices = find_spherical_neighbors(vertices, faces, i, max_neighbor_distance_km)
        neighbor_elevations = [elevations[j] for j in neighbor_indices]
        smoothed_elevations[i] = np.mean([elevations[i]] + neighbor_elevations)

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

def calculate_global_circulation(lat, elevation):
    """Calculate large-scale atmospheric circulation patterns."""
    abs_lat = abs(lat)
    
    # Determine which cell the location is in
    if abs_lat < HADLEY_CELL_WIDTH:
        # Hadley Cell (0-30°)
        cell_type = "Hadley"
        # Rising air near equator, descending near 30°
        if lat > 0:  # Northern hemisphere
            pressure = SEA_LEVEL_PRESSURE_HPA - 10 * (1 - abs_lat/HADLEY_CELL_WIDTH)
        else:  # Southern hemisphere
            pressure = SEA_LEVEL_PRESSURE_HPA - 10 * (1 - abs_lat/HADLEY_CELL_WIDTH)
    elif abs_lat < HADLEY_CELL_WIDTH + FERREL_CELL_WIDTH:
        # Ferrel Cell (30-60°)
        cell_type = "Ferrel"
        # Rising air near 60°, descending near 30°
        relative_lat = (abs_lat - HADLEY_CELL_WIDTH) / FERREL_CELL_WIDTH
        if lat > 0:  # Northern hemisphere
            pressure = SEA_LEVEL_PRESSURE_HPA + 15 * (1 - relative_lat)
        else:  # Southern hemisphere
            pressure = SEA_LEVEL_PRESSURE_HPA + 15 * (1 - relative_lat)
    else:
        # Polar Cell (60-90°)
        cell_type = "Polar"
        # Rising air near 60°, descending near poles
        relative_lat = (abs_lat - HADLEY_CELL_WIDTH - FERREL_CELL_WIDTH) / POLAR_CELL_WIDTH
        pressure = SEA_LEVEL_PRESSURE_HPA - 5 * relative_lat
    
    # Adjust for elevation
    pressure *= np.exp(-elevation * 1000 / (GAS_CONSTANT * 288))  # Scale height adjustment
    
    return pressure, cell_type

def calculate_coriolis_effect(lat, wind_speed):
    """Calculate apparent deflection due to Coriolis effect."""
    # Coriolis parameter (f = 2Ωsinφ)
    f = 2 * 7.2921e-5 * np.sin(np.radians(lat))
    # Simplified deflection (degrees)
    deflection = np.degrees(f * wind_speed * CORIOLIS_FACTOR)
    return deflection

def calculate_coriolis_parameter(lat):
    """Calculate the Coriolis parameter (f) at a given latitude.
    
    Args:
        lat (float): Latitude in degrees
        
    Returns:
        float: Coriolis parameter in 1/s
    """
    omega = 7.2921e-5  # Earth's angular velocity in rad/s
    return 2 * omega * np.sin(np.radians(lat))

def calculate_wind_patterns(lat, elevation, temperature_gradient, pressure_gradient):
    """Calculate prevailing wind direction based on latitude, elevation, and pressure gradient.
    
    Args:
        lat (float): Latitude in degrees
        elevation (float): Elevation in km
        temperature_gradient (float): Temperature change over distance
        pressure_gradient (float): Pressure change over distance (hPa/km)
        
    Returns:
        int: Wind direction (0=E, 1=N, 2=W, 3=S)
    """
    # Modified by elevation (mountains disrupt wind patterns)
    if elevation > 2000:
        return random.choice([0,1,2,3])  # Unpredictable in high mountains
    
    # Calculate geostrophic wind
    distance_km = 100  # Assume pressure gradient over 100 km
    wind_speed, wind_dir_deg = calculate_geostrophic_wind(pressure_gradient, lat, distance_km)
    
    # Convert wind direction to our simplified 4-direction system
    if wind_dir_deg >= 315 or wind_dir_deg < 45:
        base_dir = 0  # East
    elif wind_dir_deg >= 45 and wind_dir_deg < 135:
        base_dir = 1  # North
    elif wind_dir_deg >= 135 and wind_dir_deg < 225:
        base_dir = 2  # West
    else:
        base_dir = 3  # South
    
    # Apply thermal wind effect (due to temperature gradient)
    thermal_effect = np.sign(temperature_gradient) * 0.5
    if lat > 0:  # Northern hemisphere
        thermal_dir = (base_dir + int(thermal_effect)) % 4
    else:  # Southern hemisphere
        thermal_dir = (base_dir - int(thermal_effect)) % 4
    
    # Final wind direction
    final_dir = thermal_dir
    
    return final_dir

def calculate_geostrophic_wind(pressure_gradient, lat, distance):
    """Calculate geostrophic wind speed and direction.
    
    Args:
        pressure_gradient (float): Pressure difference in hPa
        lat (float): Latitude in degrees
        distance (float): Distance between pressure measurements in km
        
    Returns:
        tuple: (speed in m/s, direction in degrees)
    """
    # Constants
    rho = 1.2  # Air density kg/m³
    f = calculate_coriolis_parameter(lat)
    
    # Avoid division by zero near equator
    if abs(f) < 1e-10:
        return 0, 0
    
    # Convert units
    dp = pressure_gradient * 100  # Convert hPa to Pa
    distance_m = distance * 1000  # Convert km to m
    
    # Calculate geostrophic wind speed (m/s)
    speed = (dp / distance_m) / (rho * abs(f))
    
    # Determine direction (90° to pressure gradient, clockwise in NH, counter-clockwise in SH)
    if lat >= 0:  # Northern hemisphere
        direction = 90  # Blows parallel to isobars, high pressure to right
    else:  # Southern hemisphere
        direction = 270  # Blows parallel to isobars, high pressure to left
    
    return speed, direction

def update_pressure_systems(vertices, elevations, temperatures, day_of_year, hour_of_day):
    """Update pressure systems with diurnal and seasonal effects."""
    pressures = np.zeros(len(vertices))
    
    for i, vertex in enumerate(vertices):
        lat, lon = cartesian_to_lat_lon(*vertex)
        elevation = elevations[i]
        temperature = temperatures[i]
        
        # Base pressure from elevation and temperature
        base_pressure = calculate_pressure_with_layers(elevation, temperature)
        
        # Global circulation pattern
        circulation_pressure, _ = calculate_global_circulation(lat, elevation)
        
        # Diurnal variation
        diurnal_variation = calculate_diurnal_pressure_variation(
            lat, lon, elevation, day_of_year, hour_of_day, temperature
        )
        
        # Seasonal variation
        seasonal_variation = calculate_seasonal_pressure_variation(lat, day_of_year)
        
        # Combine all effects
        combined_pressure = (
            0.6 * circulation_pressure +
            0.3 * base_pressure +
            0.05 * diurnal_variation +
            0.05 * seasonal_variation
        )
        
        pressures[i] = combined_pressure
    
    # Apply smoothing to create more coherent pressure systems
    smoothed_pressures = np.zeros_like(pressures)
    for i in range(len(vertices)):
        neighbors = find_spherical_neighbors(vertices, faces, i, 1000)
        neighbor_pressures = [pressures[j] for j in neighbors]
        smoothed_pressures[i] = np.mean([pressures[i]] + neighbor_pressures)
    
    return smoothed_pressures

def calculate_pressure_with_circulation(elevation, temperature, humidity):
    """Calculate atmospheric pressure accounting for global circulation."""
    # First calculate standard pressure
    standard_pressure = calculate_pressure_with_composition(elevation, temperature, humidity)
    
    # Get lat/lon from vertex (approximate - we don't have vertex info here)
    # In the main loop, we'll need to pass lat/lon to this function
    lat = 0  # Placeholder - actual implementation needs lat/lon
    lon = 0
    
    # Get global circulation effect
    global_pressure, _ = calculate_global_circulation(lat, elevation)
    
    # Combine effects (weighted average)
    combined_pressure = 0.7 * global_pressure + 0.3 * standard_pressure
    
    return combined_pressure

def calculate_effective_gas_constant(humidity):
    """Calculate effective gas constant based on atmospheric composition and humidity."""
    # Adjust water vapor content based on humidity
    adjusted_composition = ATMOSPHERIC_COMPOSITION.copy()
    adjusted_composition['H2O'] = (humidity/100) * 0.04  # Max ~4% at 100% humidity
    
    # Calculate weighted average gas constant
    total = 0
    weighted_sum = 0
    for gas, fraction in adjusted_composition.items():
        weighted_sum += fraction * GAS_CONSTANTS[gas]
        total += fraction
    
    # Normalize (remaining is other trace gases)
    if total < 1:
        weighted_sum += (1 - total) * GAS_CONSTANTS['N2']  # Assume remainder is N2
    
    return weighted_sum

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

def calculate_pressure_with_composition(elevation, temperature, humidity):
    """Calculate atmospheric pressure accounting for composition and humidity."""
    # Get adjusted gas constant
    R = calculate_effective_gas_constant(humidity)
    
    # Convert elevation to meters and temperature to Kelvin
    elevation_m = elevation * 1000
    temp_k = temperature + 273.15
    
    # Calculate scale height (H = RT/Mg)
    mean_molecular_weight = calculate_mean_molecular_weight(humidity)
    scale_height = (R * temp_k) / (mean_molecular_weight * GRAVITY)
    
    # Calculate pressure
    pressure = SEA_LEVEL_PRESSURE_HPA * np.exp(-elevation_m / scale_height)
    return pressure

def calculate_greenhouse_effect(temperature, humidity, pressure, co2_level=ATMOSPHERIC_COMPOSITION['CO2']):
    """Calculate greenhouse effect based on atmospheric composition."""
    # Calculate air density (kg/m³)
    R = calculate_effective_gas_constant(humidity)
    temp_k = temperature + 273.15
    density = pressure * 100 / (R * temp_k)  # Convert hPa to Pa
    
    # Calculate greenhouse gas column densities (kg/m²)
    # Approximate as density * scale height
    mean_molecular_weight = calculate_mean_molecular_weight(humidity)
    scale_height = (R * temp_k) / (mean_molecular_weight * GRAVITY)
    
    greenhouse_effect = 0
    for gas in ['CO2', 'H2O', 'CH4', 'O3']:
        if gas == 'CO2':
            concentration = co2_level
        elif gas == 'H2O':
            concentration = (humidity/100) * ATMOSPHERIC_COMPOSITION['H2O']
        else:
            concentration = ATMOSPHERIC_COMPOSITION[gas]
        
        column_density = density * scale_height * concentration
        greenhouse_effect += column_density * GREENHOUSE_ABSORPTION[gas]
    
    return greenhouse_effect

def calculate_temperature_with_greenhouse(radiation, elevation, water_fraction, pressure, humidity):
    """Calculate temperature including greenhouse effects."""
    # Base temperature from radiation
    base_temp = (radiation / 200) - 10
    
    # Elevation effect
    elevation_effect = -6.5 * (elevation / 1000)
    
    # Water moderating effect
    water_moderation = water_fraction * 5
    
    # Greenhouse effect
    greenhouse_effect = calculate_greenhouse_effect(
        base_temp + elevation_effect + water_moderation,
        humidity,
        pressure
    )
    
    # Final temperature with greenhouse effect
    temperature = base_temp + elevation_effect + water_moderation + greenhouse_effect
    
    return max(-273, temperature)

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

def calculate_seasonal_pressure_variation(lat, day_of_year):
    """Calculate seasonal pressure variations including shifting pressure belts."""
    # Seasonal shift of pressure belts (degrees latitude)
    seasonal_shift = AXIAL_TILT_DEGREES * np.sin(2*np.pi*(day_of_year-80)/365.25)
    
    # Base pressure anomalies by latitude
    if abs(lat) < 30 - abs(seasonal_shift):
        # Subtropical high pressure
        pressure_anomaly = 5 * np.cos(np.radians(lat*3))
    elif abs(lat) < 60 - abs(seasonal_shift/2):
        # Mid-latitude low pressure
        pressure_anomaly = -8 * np.sin(np.radians(lat*1.5))
    else:
        # Polar high pressure
        pressure_anomaly = 10 * np.cos(np.radians(lat))
    
    # Monsoon-like reversal near continents (simplified)
    if 10 < abs(lat) < 30:
        monsoon_factor = np.sin(2*np.pi*(day_of_year-105)/365.25)
        pressure_anomaly += monsoon_factor * 5
    
    return pressure_anomaly

def calculate_diurnal_pressure_variation(lat, lon, elevation, day_of_year, hour_of_day, temperature):
    """Calculate diurnal pressure variations including thermal tides."""
    # Base amplitude (hPa) - stronger at low latitudes and over land
    base_amplitude = 2.0  # Typical diurnal variation is 1-3 hPa
    
    # Enhanced over land (we'll use elevation as proxy for land/sea)
    if elevation > 0:
        base_amplitude *= 1.5
        
    # Latitude effect (stronger near equator)
    lat_factor = np.cos(np.radians(lat))**2
    
    # Solar heating effect (max at local noon)
    local_noon_offset = (lon / 15) % 24  # 15 degrees per hour
    local_hour = (hour_of_day - local_noon_offset) % 24
    solar_effect = np.cos(np.radians(local_hour * 15))  # 15 degrees per hour
    
    # Combined diurnal variation
    diurnal_variation = base_amplitude * lat_factor * solar_effect
    
    return diurnal_variation

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

def visualize_world_spherical(vertices, faces, data, ax, data_type='elevation', elevations=None):
    global cbar_obj
    ax.clear()

    if data_type == 'elevation':
        # Normalize elevations for coloring (using percentile to handle outliers)
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)
        norm_data = (data - vmin) / (vmax - vmin)
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors based on vertex elevations
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.terrain(face_data)
        cmap = 'terrain'
        title = 'Elevation'

    elif data_type == 'temperature':
        # Normalize temperature data (-20 to 40°C)
        norm_data = (data + 20) / 60  # Scale -20°C to 40°C to 0-1
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.coolwarm(face_data)
        cmap = 'coolwarm'
        title = 'Temperature (°C)'

    elif data_type == 'rainfall':
        # Normalize rainfall data (0 to 200mm)
        norm_data = data / 200
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.Blues(face_data)
        cmap = 'Blues'
        title = 'Rainfall (mm)'

    elif data_type == 'pressure':
        # Normalize pressure data (950 to 1050 hPa) at surface level
        # Use the temperature data passed to the function for pressure calculation
        surface_pressures = np.array([calculate_pressure_with_layers(elevations[i], data[i])
                                  for i in range(len(elevations))]) # Use 'data' which is temperatures
        norm_data = (surface_pressures - 950) / 100
        norm_data = np.clip(norm_data, 0, 1)

        # Create face colors
        face_data = np.mean(norm_data[faces], axis=1)
        colors = plt.cm.viridis(face_data)
        cmap = 'viridis'
        title = 'Surface Pressure (hPa)'


    # Plot the mesh with face colors
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces, color='white',
                          edgecolor='none', alpha=1.0)
    mesh.set_array(face_data)
    mesh.set_cmap(cmap)

    # Add or update colorbar
    if cbar_obj is None:
        cbar_obj = plt.colorbar(mesh, ax=ax, shrink=0.5)
    else:
        cbar_obj.mappable.set_clim(vmin=np.min(face_data), vmax=np.max(face_data)) # Optional: update color limits if needed
        cbar_obj.mappable.set_array(face_data) # Update data for the colorbar

    if data_type == 'elevation':
        cbar_obj.set_label('Elevation (m)')
    elif data_type == 'temperature':
        cbar_obj.set_label('Temperature (°C)')
    elif data_type == 'rainfall':
        cbar_obj.set_label('Rainfall (mm)')
    elif data_type == 'pressure':
        cbar_obj.set_label('Pressure (hPa)') # Set label for pressure
    cbar_obj.cmap = cmap

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    ax.view_init(elev=30, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)

if __name__ == "__main__":
    # Simulation parameters
    subdivisions = 3  # Controls mesh resolution (higher = more detailed)
    radius = PLANET_RADIUS_KM
    num_plates = 15 # Increased number of plates for more fragmentation
    num_steps = 12
    step_size = 0.03
    max_neighbor_distance_km = 1000  # Distance for plate boundary interactions
    days_per_step = 30  # Each step represents a month
    current_day = 0
    current_hour = 12  # Noon
    pressures = 0
    surface_pressures = None # Initialize surface_pressures to None

    sim_params = {
        'subdivisions': subdivisions,
        'radius': radius,
        'num_plates': num_plates,
        'num_steps': num_steps,
        'step_size': step_size,
        'max_neighbor_distance_km': max_neighbor_distance_km
    }


    vertices, faces, plates, plate_assignment, elevations, surface_pressures = generate_initial_world_spherical(
        subdivisions, radius, num_plates, surface_pressures
    )
    print("Starting a new simulation.")

    # Run simulation
    world_history = []
    for step in range(num_steps):
        print(f"Simulating step {step + 1}/{num_steps}")

        # Update time
        current_day += days_per_step
        current_day %= 365  # Wrap around year
        current_hour = (current_hour + 6) % 24  # Advance 6 hours each step

        # Plate tectonics simulation
        vertices, plates, plate_assignment, elevations = simulate_plate_tectonics_spherical(
            vertices, faces, plates, plate_assignment, elevations,
            days_per_step, max_neighbor_distance_km, step_size
        )

        # Calculate water fraction for each vertex (simplified)
        water_fraction = np.where(elevations < 0, 1.0, 0.0)

        # Calculate climate variables
        sun_direction = calculate_sun_direction(current_day, current_hour)
        temperatures = np.array([
            calculate_temperature_from_radiation(
                calculate_solar_radiation_for_vertex(vertex, sun_direction, elevations[i], SEA_LEVEL_PRESSURE_HPA),
                elevations[i],
                water_fraction[i],
                SEA_LEVEL_PRESSURE_HPA
            )
            for i, vertex in enumerate(vertices)
        ])
        rainfall = np.zeros(len(vertices))
        humidity_values = np.zeros(len(vertices))
        surface_pressures = update_pressure_systems(vertices, elevations, temperatures, current_day, current_hour)

        for i, vertex in enumerate(vertices):
            lat, lon = cartesian_to_lat_lon(*vertex)

            # Get plate properties
            plate_id = plate_assignment[i]
            if plate_id > 0:
                plate = plates[plate_id-1]
                plate_temp = plate.temperature
            else:
                plate_temp = 15

            # Calculate surface pressure considering atmospheric layers
            surface_pressures[i] = calculate_pressure_with_circulation(elevations[i], temperatures[i], humidity_values[i])

            # Calculate temperature at surface considering atmospheric layers
            temperatures[i] = calculate_temperature_with_greenhouse(calculate_solar_radiation_for_vertex(vertex, sun_direction, 
                                                elevations[i], surface_pressures[i]),
                elevations[i], 
                water_fraction[i], 
                surface_pressures[i],
                humidity_values[i]
            )

            # Calculate humidity using surface conditions
            humidity_values[i] = calculate_humidity(surface_pressures[i], temperatures[i], water_fraction[i])

            # Rainfall calculation remains similar but uses the layered pressure
            rainfall[i] = calculate_rainfall(lat, lon, elevations[i], temperatures[i],
                                        surface_pressures[i], humidity_values[i],
                                        vertices, i, faces, elevations)

        world_history.append((vertices.copy(), elevations.copy(),
                            temperatures.copy(), rainfall.copy(), surface_pressures.copy())) # Store surface_pressures as well

    print("Simulation complete. Preparing interactive visualization...")

    # Set up visualization
    fig = plt.figure(figsize=(12, 10))
    ax_3d = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25, left=0.3)

    current_step = 0
    vertices, elevations, temperatures, rainfall, surface_pressures = world_history[current_step] # Unpack surface_pressures
    visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation', elevations=elevations) # Initial call to create colorbar, pass elevations

    # Add radio buttons for view selection
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio = RadioButtons(rax, ('Elevation', 'Temperature', 'Rainfall', 'Pressure'))

    def update_view(label):
        vertices, elevations, temperatures, rainfall, surface_pressures = world_history[current_step] # Unpack surface_pressures
        if label == 'Elevation':
            visualize_world_spherical(vertices, faces, elevations, ax_3d, 'elevation', elevations=elevations)
        elif label == 'Temperature':
            visualize_world_spherical(vertices, faces, temperatures, ax_3d, 'temperature', elevations=elevations)
        elif label == 'Rainfall':
            visualize_world_spherical(vertices, faces, rainfall, ax_3d, 'rainfall', elevations=elevations)
        elif label == 'Pressure':
            visualize_world_spherical(vertices, faces, temperatures, ax_3d, 'pressure', elevations=elevations) # Pass temperatures as data and elevations
        fig.canvas.draw_idle()

    radio.on_clicked(update_view)

    # Add step slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    step_slider = Slider(
        ax=ax_slider,
        label='Step',
        valmin=0,
        valmax=num_steps - 1,
        valinit=current_step,
        valstep=1
    )

    def update_step(val):
        global current_step
        current_step = int(step_slider.val)
        label = radio.value_selected
        update_view(label)

    step_slider.on_changed(update_step)

    plt.show()
