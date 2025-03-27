#pragma once

#include <vector>
#include <cmath>
#include <random>

class StormSystem {
private:
    double center_lat;
    double center_lon;
    double pressure_anomaly;
    double radius_km;
    double movement_speed_km_day;
    double movement_direction;
    int age_days;
    double max_lifetime_days;
    double intensity_change_rate;

    int find_nearest_vertex(const std::vector<std::vector<double>>& vertices);
    double calculate_steering_flow(const std::vector<double>& pressures, 
                                 const std::vector<std::vector<double>>& vertices,
                                 const std::vector<std::vector<int>>& faces);

public:
    StormSystem(double lat, double lon, double anomaly, double radius = 500.0, double speed = 50.0);
    
    void update(const std::vector<std::vector<double>>& vertices,
               const std::vector<double>& elevations,
               const std::vector<double>& temperatures,
               const std::vector<double>& pressures,
               const std::vector<std::vector<int>>& faces,
               int day_elapsed = 1);
    
    double calculate_pressure_effect(double lat, double lon, double planet_radius_km) const;
    std::pair<double, double> calculate_wind_field(double lat, double lon) const;
    bool is_in_rain_band(double lat, double lon) const;
    double get_rainfall_intensity(double lat, double lon) const;
    
    // Getters
    double get_latitude() const { return center_lat; }
    double get_longitude() const { return center_lon; }
    double get_radius() const { return radius_km; }
    double get_pressure_anomaly() const { return pressure_anomaly; }
};

std::vector<StormSystem> generate_storm_systems(const std::vector<std::vector<double>>& vertices,
                                               const std::vector<double>& elevations,
                                               int day_of_year,
                                               int num_storms = 5);