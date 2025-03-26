import numpy as np

from globals import ATMOSPHERIC_COMPOSITION, ATMOSPHERIC_LAYERS, AXIAL_TILT_DEGREES, GRAVITY, GREENHOUSE_ABSORPTION, SEA_LEVEL_PRESSURE_HPA
from pressure import calculate_effective_gas_constant
from utils import calculate_mean_molecular_weight


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
