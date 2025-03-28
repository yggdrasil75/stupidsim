#include "storms.h"
#include <algorithm>
#include <numeric>
#include "utils.cpp"  // Assuming you have C++ utils for geographic calculations

StormSystem::StormSystem(double lat, double lon, double anomaly, double radius, double speed) :
    center_lat(lat), center_lon(lon), pressure_anomaly(anomaly), 
    radius_km(radius), movement_speed_km_day(speed) {
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dir_dist(0.0, 360.0);
    std::uniform_real_distribution<> life_dist(5.0, 15.0);
    std::uniform_real_distribution<> int_dist(-0.05, 0.05);
    
    movement_direction = dir_dist(gen);
    max_lifetime_days = life_dist(gen);
    intensity_change_rate = int_dist(gen);
    age_days = 0;
}

void StormSystem::update(const std::vector<std::vector<double>>& vertices, const std::vector<double>& elevations,
                       const std::vector<double>& temperatures, const std::vector<double>& pressures,
                       const std::vector<std::vector<int>>& faces,
                       int day_elapsed) {
    
    int nearest_vertex = find_nearest_vertex(vertices);
    double temp = temperatures[nearest_vertex];
    
    // Move toward warmer areas (for tropical systems)
    if (std::abs(center_lat) < 30.0 && temp > 26.0) {
        movement_direction = calculate_steering_flow(pressures, vertices, faces);
    }
    
    // Intensify over warm water
    if (elevations[nearest_vertex] < 0.0 && temp > 26.0) {
        pressure_anomaly *= 1.02;  // 2% intensification per day
    } else if (elevations[nearest_vertex] > 0.0) {  // Weaken over land
        pressure_anomaly *= 0.95;  // 5% weakening per day
    }
    
    age_days += day_elapsed;
}

int StormSystem::find_nearest_vertex(const vector<vector<Vertex>>& vertices) {
    int nearest = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    for (int i = 0; i < vertices.size(); ++i) {
        pair<double, double> latlon = cartesian_to_lat_lon(vertices[i].x, vertices[i].y, vertices[i].z);
        double dist = haversine_distance(center_lat, center_lon, latlon.first, latlon.second);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

double StormSystem::calculate_pressure_effect(double lat, double lon, double planet_radius_km) const {
    double distance = haversine_distance(center_lat, center_lon, lat, lon, planet_radius_km);
    
    if (distance > radius_km) {
        return 0.0;
    }
    
    // Gaussian influence
    double influence = std::exp(-(distance * distance) / (2.0 * std::pow(radius_km / 3.0, 2)));
    return pressure_anomaly * influence;
}

std::pair<double, double> StormSystem::calculate_wind_field(double lat, double lon) const {
    double storm_distance = haversine_distance(center_lat, center_lon, lat, lon);
    
    if (storm_distance < radius_km) {
        double bearing = calculate_bearing_math(center_lat, center_lon, lat, lon);
        int rotation_dir = (center_lat >= 0) ? 1 : -1;  // NH vs SH
        double wind_dir = fmod(bearing + 90.0 * rotation_dir, 360.0);
        
        // Wind speed based on pressure gradient and distance from center
        double max_speed = -pressure_anomaly * 3.0;  // Convert pressure to wind speed
        double speed = max_speed * (storm_distance / radius_km) * std::exp(1.0 - (storm_distance / radius_km));
        
        return {wind_dir, speed};
    }
    return {0.0, 0.0};  // No wind outside storm radius
}

bool StormSystem::is_in_rain_band(double lat, double lon) const {
    double distance = haversine_distance(center_lat, center_lon, lat, lon);
    
    // Rain bands extend 1.5x the radius of the pressure anomaly
    if (distance > radius_km * 1.5) {
        return false;
    }
    
    // Spiral rain band pattern
    double angle = std::atan2(lon - center_lon, lat - center_lat) * 180.0 / PI;
    double band_width = 30.0;  // degrees
    return fmod(distance, radius_km / 3.0) < (radius_km / 10.0);
}

double StormSystem::get_rainfall_intensity(double lat, double lon) const {
    double distance = haversine_distance(center_lat, center_lon, lat, lon);
    
    if (distance > radius_km * 1.5) {
        return 0.0;
    }
    
    // Max rainfall near the eyewall (0.8-1.2x radius)
    if (0.8 * radius_km < distance && distance < 1.2 * radius_km) {
        return std::min(300.0, std::abs(pressure_anomaly) * 5.0);  // Cap at 300mm
    }
    
    // Decrease with distance
    return std::max(0.0, std::abs(pressure_anomaly) * 3.0 * (1.0 - distance / (radius_km * 1.5)));
}

double StormSystem::calculate_steering_flow(const std::vector<double>& pressures, 
                                          const std::vector<std::vector<Vertex>>& vertices,
                                          const std::vector<std::vector<Face>>& faces) {
    int nearest_idx = find_nearest_vertex(vertices);
    auto latlon = cartesian_to_lat_lon(vertices[nearest_idx][0], vertices[nearest_idx][1], vertices[nearest_idx][2]);
    double lat = latlon.first;
    double lon = latlon.second;
    
    // Find neighboring vertices within 500km
    vector<Vertex>& neighbors = find_spherical_neighbors(vertices, faces, nearest_idx, 500.0);
    
    if (neighbors.empty()) {
        return movement_direction;  // No change if no neighbors
    }
    
    // Calculate pressure gradient
    double sum_pressure = 0.0;
    for (int n : neighbors) {
        sum_pressure += pressures[n];
    }
    double avg_pressure = sum_pressure / neighbors.size();
    double pressure_diff = pressures[nearest_idx] - avg_pressure;
    
    double target_lat, target_lon;
    
    if (pressure_diff > 0.0) {  // We're in higher pressure - move toward lower
        std::vector<int> low_pressure_neighbors;
        for (int n : neighbors) {
            if (pressures[n] < avg_pressure) {
                low_pressure_neighbors.push_back(n);
            }
        }
        
        if (!low_pressure_neighbors.empty()) {
            double sum_lat = 0.0, sum_lon = 0.0;
            for (int n : low_pressure_neighbors) {
                auto ll = cartesian_to_lat_lon(vertices[n][0], vertices[n][1], vertices[n][2]);
                sum_lat += ll.first;
                sum_lon += ll.second;
            }
            target_lat = sum_lat / low_pressure_neighbors.size();
            target_lon = sum_lon / low_pressure_neighbors.size();
        } else {
            return movement_direction;
        }
    } else {  // Already in low pressure - move down gradient
        double sum_sin_lon_pressure = 0.0;
        double sum_cos_lat_pressure = 0.0;
        
        for (int n : neighbors) {
            auto ll = cartesian_to_lat_lon(vertices[n][0], vertices[n][1], vertices[n][2]);
            sum_sin_lon_pressure += std::sin(ll.second * PI / 180.0) * (pressures[n] - pressures[nearest_idx]);
            sum_cos_lat_pressure += std::cos(ll.first * PI / 180.0) * (pressures[n] - pressures[nearest_idx]);
        }
        
        double pressure_grad_x = sum_sin_lon_pressure / neighbors.size();
        double pressure_grad_y = sum_cos_lat_pressure / neighbors.size();
        
        // Move perpendicular to gradient (geostrophic flow)
        if (center_lat >= 0.0) {  // Northern hemisphere - counterclockwise
            target_lon = lon + pressure_grad_y * 100.0;
            target_lat = lat - pressure_grad_x * 100.0;
        } else {  // Southern hemisphere - clockwise
            target_lon = lon - pressure_grad_y * 100.0;
            target_lat = lat + pressure_grad_x * 100.0;
        }
    }
    
    // Calculate bearing to target
    double new_direction = calculate_bearing(lat, lon, target_lat, target_lon);
    
    // Smooth direction change (max 15 degrees per day)
    double direction_change = fmod(new_direction - movement_direction + 360.0, 360.0);
    if (direction_change > 180.0) {
        direction_change -= 360.0;
    }
    
    double max_change = 15.0 * (radius_km / 500.0);  // Larger storms turn slower
    direction_change = std::max(-max_change, std::min(max_change, direction_change));
    
    return fmod(movement_direction + direction_change, 360.0);
}

std::vector<StormSystem> generate_storm_systems(const std::vector<std::vector<double>>& vertices,
                                              const std::vector<double>& elevations,
                                              int day_of_year,
                                              int num_storms) {
    std::vector<StormSystem> storms;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // More storms during seasonal extremes
    double seasonal_factor = 1.0 + 0.5 * std::sin(2.0 * PI * (day_of_year - 80) / 365.25);
    int num_to_generate = static_cast<int>(num_storms * seasonal_factor);
    
    std::uniform_real_distribution<> lat_dist(-90.0, 90.0);
    std::uniform_real_distribution<> lon_dist(-180.0, 180.0);
    
    for (int i = 0; i < num_to_generate; ++i) {
        double lat, pressure_anomaly, radius, speed;
        
        // Prefer certain latitudes for storm formation
        if (std::generate_canonical<double, 10>(gen) < 0.7) {  // 70% chance in mid-latitudes
            std::uniform_real_distribution<> mid_lat_dist(-60.0, 60.0);
            lat = mid_lat_dist(gen);
            
            if (std::abs(lat) < 30.0) {  // Tropical storms
                std::uniform_real_distribution<> trop_press_dist(-15.0, -5.0);
                std::uniform_real_distribution<> trop_radius_dist(300.0, 800.0);
                std::uniform_real_distribution<> trop_speed_dist(30.0, 60.0);
                pressure_anomaly = trop_press_dist(gen);
                radius = trop_radius_dist(gen);
                speed = trop_speed_dist(gen);
            } else {  // Extratropical cyclones
                std::uniform_real_distribution<> extra_press_dist(-25.0, -10.0);
                std::uniform_real_distribution<> extra_radius_dist(500.0, 1500.0);
                std::uniform_real_distribution<> extra_speed_dist(50.0, 100.0);
                pressure_anomaly = extra_press_dist(gen);
                radius = extra_radius_dist(gen);
                speed = extra_speed_dist(gen);
            }
        } else {  // 30% chance elsewhere
            lat = lat_dist(gen);
            std::uniform_real_distribution<> press_dist(-10.0, 10.0);
            std::uniform_real_distribution<> radius_dist(200.0, 1000.0);
            std::uniform_real_distribution<> speed_dist(20.0, 80.0);
            pressure_anomaly = press_dist(gen);
            radius = radius_dist(gen);
            speed = speed_dist(gen);
        }
        
        double lon = lon_dist(gen);
        
        // Find nearest vertex
        int nearest_vertex = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < vertices.size(); ++i) {
            auto latlon = cartesian_to_lat_lon(vertices[i][0], vertices[i][1], vertices[i][2]);
            double dist = haversine_distance(lat, lon, latlon.first, latlon.second);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_vertex = i;
            }
        }
        
        // Prefer forming over water
        if (elevations[nearest_vertex] < 0.0) {  // Water
            pressure_anomaly *= 1.5;  // Stronger over water
        }
        
        storms.emplace_back(lat, lon, pressure_anomaly, radius, speed);
    }
    
    return storms;
}