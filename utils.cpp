#include <cmath>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

const double PI = 3.14159265358979323846;
const double EARTH_RADIUS_KM = 6371.0;  // Should match PLANET_RADIUS_KM

// Atmospheric composition and molecular weights would be passed from Python
// For demonstration, we'll define them here but they should be configurable
unordered_map<string, double> ATMOSPHERIC_COMPOSITION = {
    {"N2", 0.78}, {"O2", 0.21}, {"Ar", 0.0093}, {"CO2", 0.0004}, {"H2O", 0.0}
};

unordered_map<string, double> MOLAR_MASSES = {
    {"N2", 28.0134}, {"O2", 31.9988}, {"Ar", 39.948}, {"CO2", 44.0095}, {"H2O", 18.01528}
};

double calculate_mean_molecular_weight(double humidity) {
    auto adjusted_composition = ATMOSPHERIC_COMPOSITION;
    adjusted_composition["H2O"] = (humidity/100) * 0.04;
    
    double total = 0;
    double weighted_sum = 0;
    for (const auto& [gas, fraction] : adjusted_composition) {
        weighted_sum += fraction * MOLAR_MASSES[gas];
        total += fraction;
    }
    
    if (total < 1) {
        weighted_sum += (1 - total) * MOLAR_MASSES["N2"];
    }
    
    return weighted_sum;
}

string determine_surface_type(double elevation, double temperature, double water_fraction) {
    if (water_fraction > 0.9) {
        return "water";
    } else if (temperature < -5 && water_fraction > 0.1) {
        return "ice";
    } else if (elevation < 0) {
        return "water";
    } else if (water_fraction > 0.3) {
        return (temperature > 10) ? "forest" : "grassland";
    } else if (elevation > 4000) {
        return (temperature < 0) ? "ice" : "rock";
    } else if (temperature > 30 && water_fraction < 0.1) {
        return "desert";
    } else {
        return "grassland";
    }
}

tuple<double, double> cartesian_to_lat_lon(double x, double y, double z) {
    double lat = asin(z / sqrt(x*x + y*y + z*z)) * 180.0 / PI;
    double lon = atan2(y, x) * 180.0 / PI;
    return make_tuple(lat, lon);
}

tuple<double, double, double> lat_lon_to_cartesian(double lat, double lon, double radius) {
    double lat_rad = lat * PI / 180.0;
    double lon_rad = lon * PI / 180.0;
    double x = radius * cos(lat_rad) * cos(lon_rad);
    double y = radius * cos(lat_rad) * sin(lon_rad);
    double z = radius * sin(lat_rad);
    return make_tuple(x, y, z);
}

double haversine_distance(double lat1, double lon1, double lat2, double lon2, double radius = EARTH_RADIUS_KM) {
    double lat1_rad = lat1 * PI / 180.0;
    double lon1_rad = lon1 * PI / 180.0;
    double lat2_rad = lat2 * PI / 180.0;
    double lon2_rad = lon2 * PI / 180.0;

    double dlon = lon2_rad - lon1_rad;
    double dlat = lat2_rad - lat1_rad;

    double a = pow(sin(dlat / 2), 2) + cos(lat1_rad) * cos(lat2_rad) * pow(sin(dlon / 2), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return radius * c;
}

double calculate_bearing_math(double lat1, double lon1, double lat2, double lon2) {
    double lat1_rad = lat1 * PI / 180.0;
    double lon1_rad = lon1 * PI / 180.0;
    double lat2_rad = lat2 * PI / 180.0;
    double lon2_rad = lon2 * PI / 180.0;

    double dLon = lon2_rad - lon1_rad;

    double x = sin(dLon) * cos(lat2_rad);
    double y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dLon);

    double initial_bearing = atan2(x, y);
    double compass_bearing = fmod(initial_bearing * 180.0 / PI + 360, 360);

    return compass_bearing;
}

// Vectorized version using NumPy arrays
py::array_t<double> calculate_bearing(py::array_t<double> lat1, py::array_t<double> lon1, 
                                     py::array_t<double> lat2, py::array_t<double> lon2) {
    py::buffer_info buf_lat1 = lat1.request();
    py::buffer_info buf_lon1 = lon1.request();
    py::buffer_info buf_lat2 = lat2.request();
    py::buffer_info buf_lon2 = lon2.request();

    if (buf_lat1.size != buf_lon1.size || buf_lat1.size != buf_lat2.size || buf_lat1.size != buf_lon2.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    auto result = py::array_t<double>(buf_lat1.size);
    py::buffer_info buf_result = result.request();

    double* ptr_lat1 = static_cast<double*>(buf_lat1.ptr);
    double* ptr_lon1 = static_cast<double*>(buf_lon1.ptr);
    double* ptr_lat2 = static_cast<double*>(buf_lat2.ptr);
    double* ptr_lon2 = static_cast<double*>(buf_lon2.ptr);
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (size_t i = 0; i < buf_lat1.size; i++) {
        double lat1_rad = ptr_lat1[i] * PI / 180.0;
        double lon1_rad = ptr_lon1[i] * PI / 180.0;
        double lat2_rad = ptr_lat2[i] * PI / 180.0;
        double lon2_rad = ptr_lon2[i] * PI / 180.0;

        double dLon = lon2_rad - lon1_rad;

        double x = sin(dLon) * cos(lat2_rad);
        double y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dLon);

        double initial_bearing = atan2(x, y);
        ptr_result[i] = fmod(initial_bearing * 180.0 / PI + 360, 360);
    }

    return result;
}

PYBIND11_MODULE(utils_cpp, m) {
    m.def("calculate_mean_molecular_weight", &calculate_mean_molecular_weight, 
          "Calculate mean molecular weight of atmosphere based on humidity");
    m.def("determine_surface_type", &determine_surface_type, 
          "Determine surface type for albedo calculation");
    m.def("cartesian_to_lat_lon", &cartesian_to_lat_lon, 
          "Convert cartesian coordinates to latitude/longitude");
    m.def("lat_lon_to_cartesian", &lat_lon_to_cartesian, 
          "Convert latitude/longitude to cartesian coordinates");
    m.def("haversine_distance", &haversine_distance, 
          "Calculate great-circle distance between two points on a sphere");
    m.def("calculate_bearing_math", &calculate_bearing_math, 
          "Calculate bearing between two points (single values)");
    m.def("calculate_bearing", &calculate_bearing, 
          "Calculate bearing between two points (vectorized with numpy arrays)");
}