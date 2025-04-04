from random import random
import numpy as np
from globals import ATMOSPHERIC_COMPOSITION, ATMOSPHERIC_LAYERS, AXIAL_TILT_DEGREES, CORIOLIS_FACTOR, FERREL_CELL_WIDTH, GAS_CONSTANT, GAS_CONSTANTS, GRAVITY, HADLEY_CELL_WIDTH, POLAR_CELL_WIDTH, SEA_LEVEL_PRESSURE_HPA
from utils import calculate_mean_molecular_weight, findSphericalNeighbors
from _icosphere import HaversineDistance, cartesianLatLon


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

def calculate_pressure_with_layers(elevation_km, surface_temp, lat=None):
    """Calculate pressure considering atmospheric layers with hydrostatic equilibrium and density variations.
    
    Args:
        elevation_km (float): Elevation in kilometers (negative for below sea level).
        surface_temp (float): Surface temperature in degrees Celsius.
        
    Returns:
        float: Pressure in hPa at the given elevation.
    """
    current_pressure = SEA_LEVEL_PRESSURE_HPA  # Start with sea level pressure
    current_temp = surface_temp + 273.15  # Convert to Kelvin
    current_altitude = 0.0  # Start at sea level
    remaining_altitude = max(0, -elevation_km)  # Convert elevation to altitude (positive up)
    
    # Calculate mean molecular weight at sea level (dry air approximation)
    mean_molecular_weight = calculate_mean_molecular_weight(0)  # 0% humidity for simplicity
    
    for layer in ATMOSPHERIC_LAYERS:
        # Determine altitude range based on latitude if available
        if lat is not None and layer.get("lat_variation"):
            abs_lat = abs(lat)
            if abs_lat < 30:
                layer_range = layer["lat_variation"]["equator"]
            elif abs_lat < 60:
                layer_range = layer["lat_variation"]["midlat"]
            else:
                layer_range = layer["lat_variation"]["polar"]
        else:
            layer_range = layer["altitude_range"]
        
        layer_bottom, layer_top = layer_range
        gradient = layer["temp_gradient"]
        
        # Calculate the thickness of the layer segment we need to process
        layer_thickness = min(layer_top, remaining_altitude + layer_bottom) - layer_bottom
        if layer_thickness <= 0:
            continue
        
        # Calculate temperature at the top of this layer segment
        temp_top = current_temp + gradient * layer_thickness
        
        if gradient == 0:
            # Isothermal layer
            scale_height = (GAS_CONSTANT * current_temp) / (mean_molecular_weight * GRAVITY)
            current_pressure *= np.exp(-layer_thickness * 1000 / scale_height)
        else:
            # Non-isothermal layer
            exponent = -GRAVITY * mean_molecular_weight / (GAS_CONSTANT * gradient * 1000)
            current_pressure *= (temp_top / current_temp) ** exponent
        
        current_temp = temp_top
        current_altitude += layer_thickness
        remaining_altitude -= layer_thickness
        
        if remaining_altitude <= 0:
            break
    
    return current_pressure

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

def update_pressure_systems(vertices, faces, elevations, temperatures, day_of_year, hour_of_day, active_storms):
    """Update pressure systems including dynamic storm systems."""
    pressures = np.zeros(len(vertices))
    
    for i, vertex in enumerate(vertices):
        lat, lon = cartesianLatLon(vertex)
        elevation = elevations[i]
        temperature = temperatures[i]
        
        # Base pressure from elevation and temperature
        base_pressure = calculate_pressure_with_layers(elevation, temperature, lat)
        
        # Global circulation pattern
        circulation_pressure, _ = calculate_global_circulation(lat, elevation)
        
        # Diurnal variation
        diurnal_variation = calculate_diurnal_pressure_variation(
            lat, lon, elevation, day_of_year, hour_of_day, temperature
        )
        
        # Seasonal variation
        seasonal_variation = calculate_seasonal_pressure_variation(lat, day_of_year)
        
        # Storm effects
        storm_effect = 0
        for storm in active_storms:
            for i, vertex in enumerate(vertices):
                lat, lon = cartesianLatLon(vertex)
                distance = HaversineDistance(storm.center_lat, storm.center_lon, lat, lon)
                
                if distance < storm.radius_km * 2:  # Wider influence area
                    # Create proper pressure gradient toward storm center
                    gradient_dir = 1 if storm.pressure_anomaly < 0 else -1  # Low vs high pressure
                    pressures[i] += gradient_dir * (storm.pressure_anomaly * 
                                                np.exp(-(distance**2)/(2*(storm.radius_km)**2)))
        
        # Combine all effects
        combined_pressure = (
            0.5 * circulation_pressure +
            0.3 * base_pressure +
            0.05 * diurnal_variation +
            0.05 * seasonal_variation +
            0.1 * storm_effect
        )
        
        pressures[i] = combined_pressure
    
    # Apply smoothing to create more coherent pressure systems
    smoothed_pressures = np.zeros_like(pressures)
    for i in range(len(vertices)):
        neighbors = findSphericalNeighbors(vertices, faces, i, 1000)
        neighbor_pressures = [pressures[j] for j in neighbors]
        smoothed_pressures[i] = np.mean([pressures[i]] + neighbor_pressures)
    
    return smoothed_pressures

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
