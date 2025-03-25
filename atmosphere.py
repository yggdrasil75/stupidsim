# atmosphere.py
import numpy as np
from typing import Tuple

# Constants
GAS_CONSTANT_DRY_AIR = 287.05  # J/kg·K
GRAVITY = 9.80665  # m/s²
SEA_LEVEL_PRESSURE_PA = 101325.0  # Pa (1 atm)
SEA_LEVEL_TEMPERATURE_K = 288.15  # 15°C in Kelvin
MOLAR_MASS_AIR = 0.0289644  # kg/mol
UNIVERSAL_GAS_CONSTANT = 8.31447  # J/(mol·K)

# Atmospheric layer boundaries (in meters above sea level)
TROPOPAUSE_HEIGHT = 11000  # ~11 km
STRATOPAUSE_HEIGHT = 50000  # ~50 km
MESOPAUSE_HEIGHT = 85000  # ~85 km
THERMOPAUSE_HEIGHT = 600000  # ~600 km

class AtmosphericLayer:
    def __init__(self, name: str, base_height: float, base_temp: float, lapse_rate: float):
        """
        Initialize an atmospheric layer.
        
        Args:
            name: Name of the layer (e.g., "troposphere")
            base_height: Height at bottom of layer (m)
            base_temp: Temperature at bottom of layer (K)
            lapse_rate: Temperature change with altitude (K/m)
        """
        self.name = name
        self.base_height = base_height
        self.base_temp = base_temp
        self.lapse_rate = lapse_rate
    
    def temperature_at_altitude(self, altitude: float) -> float:
        """Calculate temperature at given altitude within this layer."""
        return self.base_temp + self.lapse_rate * (altitude - self.base_height)
    
    def pressure_at_altitude(self, altitude: float, base_pressure: float) -> float:
        """
        Calculate pressure at given altitude within this layer using barometric formula.
        
        Args:
            altitude: Height above sea level (m)
            base_pressure: Pressure at base of this layer (Pa)
            
        Returns:
            Pressure in Pascals
        """
        if self.lapse_rate == 0:
            # Isothermal layer
            exponent = -GRAVITY * MOLAR_MASS_AIR * (altitude - self.base_height) / \
                     (UNIVERSAL_GAS_CONSTANT * self.base_temp)
            return base_pressure * np.exp(exponent)
        else:
            # Non-isothermal layer
            exponent = -GRAVITY * MOLAR_MASS_AIR / (UNIVERSAL_GAS_CONSTANT * self.lapse_rate)
            temp_ratio = self.temperature_at_altitude(altitude) / self.base_temp
            return base_pressure * (temp_ratio ** exponent)

class Atmosphere:
    def __init__(self):
        """Initialize Earth-like atmosphere with standard layers."""
        # Define atmospheric layers (from surface up)
        self.layers = [
            # name, base height (m), base temp (K), lapse rate (K/m)
            AtmosphericLayer("troposphere", 0, SEA_LEVEL_TEMPERATURE_K, -0.0065),
            AtmosphericLayer("tropopause", TROPOPAUSE_HEIGHT, 216.65, 0),  # Isothermal
            AtmosphericLayer("stratosphere", TROPOPAUSE_HEIGHT, 216.65, 0.001),
            AtmosphericLayer("stratopause", STRATOPAUSE_HEIGHT, 270.65, 0),
            AtmosphericLayer("mesosphere", STRATOPAUSE_HEIGHT, 270.65, -0.0028),
            AtmosphericLayer("mesopause", MESOPAUSE_HEIGHT, 180.65, 0),
            AtmosphericLayer("thermosphere", MESOPAUSE_HEIGHT, 180.65, 0.003),
            AtmosphericLayer("exosphere", THERMOPAUSE_HEIGHT, 1000.0, 0)  # Simplified
        ]
    
    def get_layer_for_altitude(self, altitude: float) -> Tuple[AtmosphericLayer, float]:
        """
        Get the atmospheric layer and base pressure for a given altitude.
        
        Args:
            altitude: Height above sea level (m)
            
        Returns:
            Tuple of (layer, base_pressure)
        """
        current_pressure = SEA_LEVEL_PRESSURE_PA
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer = self.layers[i + 1]
                if altitude < next_layer.base_height:
                    return layer, current_pressure
                
                # Calculate pressure at top of this layer
                current_pressure = layer.pressure_at_altitude(
                    next_layer.base_height, current_pressure
                )
            else:
                return layer, current_pressure
        
        return self.layers[-1], current_pressure
    
    def calculate_conditions(self, altitude: float, surface_temp_k: float = None) -> dict:
        """
        Calculate atmospheric conditions at a given altitude.
        
        Args:
            altitude: Height above sea level (m)
            surface_temp_k: Optional surface temperature to override standard atmosphere
            
        Returns:
            Dictionary with temperature (K), pressure (Pa), and layer name
        """
        if surface_temp_k is not None and altitude == 0:
            # Use provided surface temperature if at surface level
            return {
                "temperature": surface_temp_k,
                "pressure": SEA_LEVEL_PRESSURE_PA,
                "layer": "surface"
            }
        
        # Find the appropriate layer
        layer, base_pressure = self.get_layer_for_altitude(altitude)
        
        # Calculate temperature and pressure
        temperature = layer.temperature_at_altitude(altitude)
        pressure = layer.pressure_at_altitude(altitude, base_pressure)
        
        return {
            "temperature": temperature,
            "pressure": pressure,
            "layer": layer.name
        }
    
    def calculate_surface_pressure(self, elevation: float, surface_temp_k: float) -> float:
        """
        Calculate adjusted surface pressure based on elevation and surface temperature.
        
        Args:
            elevation: Height above sea level (m) - can be negative for below sea level
            surface_temp_k: Surface temperature in Kelvin
            
        Returns:
            Surface pressure in Pascals
        """
        if elevation == 0:
            return SEA_LEVEL_PRESSURE_PA
        
        # For elevations below sea level, we need to handle differently
        if elevation < 0:
            # Simplified: assume pressure increases by 1 atm per 10m depth
            return SEA_LEVEL_PRESSURE_PA + (-elevation * 10000)
        
        # For above sea level, use barometric formula with actual surface temperature
        # This is more accurate than the standard atmosphere model for surface conditions
        exponent = -GRAVITY * MOLAR_MASS_AIR * elevation / \
                 (UNIVERSAL_GAS_CONSTANT * surface_temp_k)
        return SEA_LEVEL_PRESSURE_PA * np.exp(exponent)
    
    def calculate_density(self, pressure: float, temperature: float) -> float:
        """
        Calculate air density using ideal gas law.
        
        Args:
            pressure: Pressure in Pascals
            temperature: Temperature in Kelvin
            
        Returns:
            Air density in kg/m³
        """
        return pressure / (GAS_CONSTANT_DRY_AIR * temperature)
    
    def calculate_scale_height(self, temperature: float) -> float:
        """
        Calculate atmospheric scale height.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Scale height in meters
        """
        return (UNIVERSAL_GAS_CONSTANT * temperature) / (MOLAR_MASS_AIR * GRAVITY)