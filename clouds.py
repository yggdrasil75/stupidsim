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
        """Update cloud coverage based on current atmospheric conditions"""
        for i in range(len(self.vertices)):
            # Base cloud formation probability
            cloud_prob = min(0.9, humidities[i]/100 * 0.7 + 
                           (1 - pressures[i]/SEA_LEVEL_PRESSURE_HPA) * 0.3)
            
            # Temperature effect (more clouds at moderate temps)
            temp_effect = 1 - abs(temperatures[i] - 20)/30
            cloud_prob *= max(0.1, temp_effect)
            
            # Rainfall effect (more clouds where it's raining)
            cloud_prob = min(1.0, cloud_prob + rainfall[i]/300)
            
            # Seasonal variation
            seasonal_var = 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            cloud_prob = np.clip(cloud_prob + seasonal_var, 0, MAX_CLOUD_COVERAGE)
            
            # Determine cloud type based on elevation and conditions
            if self.elevations[i] > 4000 and temperatures[i] < -20:
                cloud_type_idx = 0  # cirrus
            elif rainfall[i] > 50:
                cloud_type_idx = 4  # cumulonimbus
            elif cloud_prob > 0.7:
                cloud_type_idx = 3  # cumulus
            elif cloud_prob > 0.4:
                cloud_type_idx = 2  # stratus
            else:
                cloud_type_idx = 1  # alto
            
            self.cloud_coverage[i] = cloud_prob
            self.cloud_type[i] = cloud_type_idx
            self.cloud_albedo[i] = CLOUD_ALBEDO_VALUES[list(CLOUD_ALBEDO_VALUES.keys())[cloud_type_idx]]
            
    def get_effective_albedo(self, surface_albedo):
        """Calculate combined surface and cloud albedo"""
        return surface_albedo * (1 - self.cloud_coverage) + self.cloud_albedo * self.cloud_coverage