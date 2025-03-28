#include <cmath>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>
#include "globals.cpp"

using namespace std;

const double PI = 3.14159265358979323846;
const double EARTH_RADIUS_KM = 6371.0;  // Should match PLANET_RADIUS_KM
struct Vertex {
	double x, y, z;
};
struct Face {
	Vertex a, b, c;
};

// Atmospheric composition and molecular weights would be passed from Python
// For demonstration, we'll define them here but they should be configurable

double calculate_mean_molecular_weight(double humidity) {
    auto adjusted_composition = ATMOSPHERIC_COMPOSITION;
    adjusted_composition["H2O"] = (humidity/100) * 0.04;
    
    double total = 0;
    double weighted_sum = 0;
    for (const auto& [gas, fraction] : adjusted_composition) {
        weighted_sum += fraction * MOLECULAR_WEIGHTS[gas];
        total += fraction;
    }
    
    if (total < 1) {
        weighted_sum += (1 - total) * MOLECULAR_WEIGHTS["N2"];
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

pair<double, double> cartesian_to_lat_lon(double x, double y, double z) {
    double lat = asin(z / sqrt(x*x + y*y + z*z)) * 180.0 / PI;
    double lon = atan2(y, x) * 180.0 / PI;
    return make_pair(lat, lon);
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

vector<Vertex> find_spherical_neighbors(vector<Vertex>& vertices, vector<Face>& faces, 
									int vertexID, double maxDistanceKm) {
	std::unordered_set<Vertex> neighbors;
	const Vertex& center = vertices[vertexID];
	
	for (const Face& face : faces){
		if (face.a == vertexID || face.b == vertexID || face.c == vertexID) {
			if (face.a != vertexID){
				neighbors.insert(face.a);
			}
			if (face.b != vertexID){
				neighbors.insert(face.b);
			}
			if (face.c != vertexID){
				neighbors.insert(face.c);
			}
		}
	}

	vector<Vertex> neighbors2;
	auto& [lat1, lon1] = cartesian_to_lat_lon(center.x, center.y, center.z);
	for (Vertex v: neighbors){
		const Vertex& vertex = vertices[v];
		auto& [lat2, lon2] = cartesian_to_lat_lon(vertex.x, vertex.y, vertex.z);
		double dist = haversine_distance(lat1, lon1, lat2, lon2);

		if (dist <= maxDistanceKm){
			neighbors2.push_back(v);
		}
	}
	return neighbors2;
}

// Vectorized version using NumPy arrays
vector<double> calculate_bearing(vector<double> lat1, vector<double> lon1, 
									vector<double> lat2, vector<double> lon2) {

    if (lat1.size() != lon1.size() || lat1.size() != lat2.size() || lat1.size() != lon2.size()) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    vector<double> result(lat1.size());

    for (size_t i = 0; i < lat1.size(); i++) {
        double lat1_rad = lat1[i] * PI / 180.0;
        double lon1_rad = lon1[i] * PI / 180.0;
        double lat2_rad = lat2[i] * PI / 180.0;
        double lon2_rad = lon2[i] * PI / 180.0;

        double dLon = lon2_rad - lon1_rad;

        double x = sin(dLon) * cos(lat2_rad);
        double y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dLon);

        double initial_bearing = atan2(x, y);
        result[i] = fmod(initial_bearing * 180.0 / PI + 360, 360);
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