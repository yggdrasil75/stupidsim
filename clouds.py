import numpy as np
from globals import CLOUD_ALBEDO_VALUES, MAX_CLOUD_COVERAGE, SEA_LEVEL_PRESSURE_HPA

class CloudSystem:
    def __init__(self, vertices, elevations):
        self.vertices = vertices
        self.elevations = elevations
        self.cloud_coverage = np.zeros(len(vertices))  # 0-1 scale
        self.cloud_type = np.zeros(len(vertices), dtype=int)  # Index for cloud type
        self.cloud_albedo = np.zeros(len(vertices))
        
    def update_clouds(self, temperatures, humidities, pressures, rainfall, day_of_year):
        # Convert inputs to numpy arrays if they aren't already
        temperatures = np.asarray(temperatures)
        humidities = np.asarray(humidities)
        pressures = np.asarray(pressures)
        rainfall = np.asarray(rainfall)
        
        # Calculate dew points
        dewpoints = temperatures - ((100 - humidities) / 5)
        
        # Relative humidity effect (0-1 scale)
        rh_effect = np.clip((humidities - 50) / 50, 0, 1)
        
        # Pressure effect - more clouds in low pressure areas
        pressure_norm = (1013 - pressures) / 30  # Normalize pressure anomaly
        pressure_effect = np.clip(pressure_norm, 0, 1)
        
        # Stability effect - less clouds when surface temp is much higher than dewpoint
        temp_dew_diff = temperatures - dewpoints
        stability_effect = 1 - np.clip((temp_dew_diff - 5) / 15, 0, 1)
        
        # Rainfall effect - recent rain indicates existing clouds
        rainfall_effect = np.clip(rainfall / 100, 0, 0.3)
        
        # Seasonal variation
        seasonal_var = 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        
        # Combine all effects with appropriate weights
        cloud_prob = (
            0.5 * rh_effect + 
            0.3 * pressure_effect + 
            0.2 * rainfall_effect
        ) * stability_effect + seasonal_var
        
        # Ensure reasonable bounds
        cloud_prob = np.clip(cloud_prob, 0.05, 0.95)
        
        # Determine cloud types
        cloud_type_idx = np.ones(len(self.vertices), dtype=int)  # Default to alto clouds
        
        # High clouds (cirrus)
        high_cloud_mask = self.elevations > 6000
        cloud_type_idx[high_cloud_mask] = 0
        
        # Storm clouds (cumulonimbus)
        storm_cloud_mask = rainfall > 10
        cloud_type_idx[storm_cloud_mask] = 4
        
        # Cumulus clouds (unstable air)
        cumulus_mask = (cloud_prob > 0.6) & (temp_dew_diff < 3) & ~storm_cloud_mask
        cloud_type_idx[cumulus_mask] = 3
        
        # Stratus clouds (stable air)
        stratus_mask = (cloud_prob > 0.4) & ~cumulus_mask & ~storm_cloud_mask
        cloud_type_idx[stratus_mask] = 2
        
        # Update cloud properties
        self.cloud_coverage = cloud_prob
        self.cloud_type = cloud_type_idx
        
        # Set albedo based on cloud type
        cloud_types = list(CLOUD_ALBEDO_VALUES.keys())
        for i, idx in enumerate(cloud_type_idx):
            self.cloud_albedo[i] = CLOUD_ALBEDO_VALUES[cloud_types[idx]]
            
    def get_effective_albedo(self, surface_albedo):
        """Calculate combined surface and cloud albedo"""
        return surface_albedo * (1 - self.cloud_coverage) + self.cloud_albedo * self.cloud_coverage