#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thread>
#include <future>
#include <numeric>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

struct Vertex {
    double x, y, z, elevation;
    double surface_water;  // Water on the surface (rivers, lakes, etc.)
    double groundwater;    // Water underground
    Vertex(double x=0, double y=0, double z=0) : x(x), y(y), z(z), elevation(0.0), surface_water(0.0), groundwater(0.0) {}
    Vertex(double x, double y, double z, double elevation) : x(x), y(y), z(z), elevation(elevation), surface_water(0.0), groundwater(0.0) {}
    Vertex operator+(const Vertex& other) const {
        return Vertex(x + other.x, y + other.y, z + other.z);
    }
    Vertex operator-(const Vertex& other) const {
        return Vertex(x - other.x, y - other.y, z - other.z);
    }
    Vertex operator/(double scalar) const {
        return Vertex(x / scalar, y / scalar, z / scalar);
    }
    Vertex operator*(double scalar) const {
        return Vertex(x * scalar, y * scalar, z * scalar);
    }
    double length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    Vertex normalized() const {
        double len = length();
        return Vertex(x/len, y/len, z/len);
    }
    bool operator<(const Vertex& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    bool operator==(const Vertex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    bool operator!=(const Vertex& other) const {
        return !(*this == other);
    }
};

double dotProduct(const Vertex& a, const Vertex& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Face {
    int a, b, c;
    //Vertex a, b, c;
    double average_elevation;
    double initAverage;
    double surface_water;
    double average_atmospheric_water;
    double temperature;

    //Face(Vertex a, Vertex b, Vertex c) : a(a), b(b), c(c), average_elevation(0.0) {}
    Face(int a, int b, int c) : a(a), b(b), c(c), average_elevation(0.0) {}
    //Face(Vertex a, Vertex b, Vertex c, double averageElevation) : a(a), b(b), c(c), average_elevation(averageElevation) {}

    bool operator<(const Face& other) const {
      if(a != other.a) return a < other.a;
      if(b != other.b) return b < other.b;
      return c < other.c;
    }
    bool operator==(const Face& other) const {
        int these[3] = {a, b, c};
        int those[3] = {other.a, other.b, other.c};
        return 
        return these[0] == those[0] &&
            these[1] == those[1] &&
            these[2] == those[2];
    }
    bool operator!=(const Face& other) const {
        return !(*this == other);
    }
    double area(WorldState world) const {
        double dotproda = dotProduct(world.vertexIndices[face.a], world.vertexIndices[face.b]);
        double dotprodb = dotProduct(world.vertexIndices[face.b], world.vertexIndices[face.c]);
        double dotprodc = dotProduct(world.vertexIndices[face.c], world.vertexIndices[face.a]);

        double s = (dotproda + dotprodb + dotprodc) / 2.0;
        double tan_half_E_squared = tan(s / 2.0) * tan((s - dotproda) / 2.0) * tan((s-dotprodb) / 2.0) * tan((s - dotprodc) / 2.0);
        double E = 4.0 * atan(sqrt(tan_half_E_squared));
        return E * world.radius * world.radius;
    }
};

// Define a struct to hold the world state
struct WorldState {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::map<Vertex, size_t> vertexIndices;
    double total_surface_water;
    double total_ground_water;
    double total_atmospheric_water;
    double radius;

    WorldState(std::vector<Vertex> vertices, std::vector<Face> faces, std::map<Vertex, size_t> vertexIndices, double radius) : vertices(vertices), faces(faces), vertexIndices(vertexIndices), radius(radius) {};
};

Vertex crossProduct(const Vertex& a, const Vertex& b) {
    return Vertex(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(const Vertex& v) const {
            size_t h1 = hash<double>{}(v.x);
            size_t h2 = hash<double>{}(v.y);
            size_t h3 = hash<double>{}(v.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace {
    std::unordered_map<int, std::vector<Vertex>> neighborCache;
    std::mutex cacheMutex;
}

std::vector<Face> subdivide_icosphere(size_t subdivisions, std::vector<Vertex>& vertices, std::vector<Face>& faces) {
    // Create a map from Vertex to its index in the vertices vector
    std::unordered_map<Vertex, uint32_t> vertex_to_index;
    for (uint32_t i = 0; i < vertices.size(); ++i) {
        vertex_to_index[vertices[i]] = i;
    }

    for (size_t _ = 0; _ < subdivisions; ++_) {
        std::vector<Face> new_faces;
        new_faces.reserve(faces.size() * 4);

        // Map from edge (pair of vertex indices) to new vertex
        std::unordered_map<uint64_t, Vertex> edge_vertices;

        for (const Face& face : faces) {
            const Vertex& a = vertices[face.a];
            const Vertex& b = vertices[face.b];
            const Vertex& c = vertices[face.c];

            // Get or create midpoints for each edge
            Vertex mid_ab, mid_bc, mid_ca;

            // Edge AB
            uint64_t key_ab = (static_cast<uint64_t>(std::min(vertex_to_index[a], vertex_to_index[b])) << 32) | std::max(vertex_to_index[a], vertex_to_index[b]);
            auto it_ab = edge_vertices.find(key_ab);
            if (it_ab == edge_vertices.end()) {
                mid_ab = (a + b) / 2.0;
                mid_ab = mid_ab.normalized();
                edge_vertices[key_ab] = mid_ab;
                vertices.push_back(mid_ab);
                vertex_to_index[mid_ab] = vertices.size() - 1;
            } else {
                mid_ab = it_ab->second;
            }

            // Edge BC
            uint64_t key_bc = (static_cast<uint64_t>(std::min(vertex_to_index[b], vertex_to_index[c])) << 32) | std::max(vertex_to_index[b], vertex_to_index[c]);
            auto it_bc = edge_vertices.find(key_bc);
            if (it_bc == edge_vertices.end()) {
                mid_bc = (b + c) / 2.0;
                mid_bc = mid_bc.normalized();
                edge_vertices[key_bc] = mid_bc;
                vertices.push_back(mid_bc);
                vertex_to_index[mid_bc] = vertices.size() - 1;
            } else {
                mid_bc = it_bc->second;
            }

            // Edge CA
            uint64_t key_ca = (static_cast<uint64_t>(std::min(vertex_to_index[c], vertex_to_index[a])) << 32) | std::max(vertex_to_index[c], vertex_to_index[a]);
            auto it_ca = edge_vertices.find(key_ca);
            if (it_ca == edge_vertices.end()) {
                mid_ca = (c + a) / 2.0;
                mid_ca = mid_ca.normalized();
                edge_vertices[key_ca] = mid_ca;
                vertices.push_back(mid_ca);
                vertex_to_index[mid_ca] = vertices.size() - 1;
            } else {
                mid_ca = it_ca->second;
            }
            new_faces.emplace_back(vertex_to_index[a], vertex_to_index[mid_ab], vertex_to_index[mid_ca]);
            new_faces.emplace_back(vertex_to_index[mid_ab], vertex_to_index[b], vertex_to_index[mid_bc]);
            new_faces.emplace_back(vertex_to_index[mid_ca], vertex_to_index[mid_bc], vertex_to_index[c]);
            new_faces.emplace_back(vertex_to_index[mid_ab], vertex_to_index[mid_bc], vertex_to_index[mid_ca]);
        }

        faces = std::move(new_faces);

        for (Vertex& v : vertices) {
            v = v.normalized();
        }
    }

    return faces;
}

// Modified generate_icosphere to return face indices
std::tuple<std::vector<Vertex>, std::vector<Face>, std::map<Vertex, size_t>>
    generate_icosphere(size_t subdivisions=3, double radius=1.0) {
    // Golden ratio
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;

    // Create initial icosahedron vertices
    std::vector<Vertex> vertices = {
        Vertex(-1,  t, 0), Vertex(1,  t, 0), Vertex(-1, -t, 0), Vertex(1, -t, 0),
        Vertex(0, -1,  t), Vertex(0, 1,  t), Vertex(0, -1, -t), Vertex(0, 1, -t),
        Vertex( t, 0, -1), Vertex(t, 0, 1), Vertex(-t, 0, -1), Vertex(-t, 0, 1)
    };

    // Normalize vertices to unit sphere
    for (auto& v : vertices) {
        v = v.normalized();
    }

    // Create initial icosahedron faces
    std::vector<Face> faces = {
        Face(0, 1, 7),
        Face(0, 5, 1),
        Face(0, 7, 10),
        Face(0, 10, 11),
        Face(0, 11, 5),
        Face(1, 5, 9),
        Face(2, 4, 11),
        Face(3, 9, 4),
        Face(3, 4, 2),
        Face(3, 2, 6),
        Face(3, 6, 8),
        Face(3, 8, 9),
        Face(4, 9, 5),
        Face(5, 11, 4),
        Face(6, 2, 10),
        Face(7, 1, 8),
        Face(8, 6, 7),
        Face(9, 8, 1),
        Face(10, 7, 6),
        Face(11, 10, 2)
    };

    faces = subdivide_icosphere(subdivisions, vertices, faces);

    // Collect all unique vertices and create face indices
    std::vector<Vertex> unique_vertices;
    std::map<Vertex, size_t> vertex_indices;
    std::vector<Face> unique_faces; // Store face indices

    for (const Face& face : faces) {
        std::array<uint32_t, 3> current_face_indices;
        int index_count = 0;
        for (const int& v : {face.a, face.b, face.c}) {
            if (vertex_indices[v] == vertex_indices.end()) {
                vertex_indices[v] = unique_vertices.size();
                unique_vertices.push_back(v);
            }
            current_face_indices[index_count++] = vertex_indices[v];
        }
    }

    for (const Face& face : faces){
        unique_faces.emplace_back(
            unique_vertices[face.a],
            unique_vertices[face.b],
            unique_vertices[face.c]
        );
    }

    // Scale vertices by radius
    for (Vertex& v : unique_vertices) {
        v.x *= radius;
        v.y *= radius;
        v.z *= radius;
    }

    return {unique_vertices, unique_faces, vertex_indices}; // Return face indices
}

std::tuple<double, double> cartesianLatLon(Vertex vertex){
    double lat_rad = std::asin(vertex.z / std::sqrt(std::pow(vertex.x, 2) + std::pow(vertex.y, 2) + std::pow(vertex.z, 2)));
    double lat_deg = lat_rad * (180.0 / M_PI);

    double lon_rad = std::atan2(vertex.y, vertex.x);
    double lon_deg = lon_rad * (180.0 / M_PI);

    return {lat_deg, lon_deg};
}

Vertex latLonCartesian(double lat, double lon, double radius){
    double lonRad = lon * (M_PI / 180);
    double latRad = lat * (M_PI / 180);

    double x = radius * std::cos(latRad) * std::cos(lonRad);
    double y = radius * std::cos(latRad) * std::cos(lonRad);
    double z = radius * std::sin(latRad);

    return Vertex(x, y, z);

}

double HaversineDistance(double lat1, double lon1, double lat2, double lon2, double radius){
    double lat1Rad = lat1 * M_PI / 180.0;
    double lon1Rad = lon1 * M_PI / 180.0;
    double lat2Rad = lat2 * M_PI / 180.0;
    double lon2Rad = lon2 * M_PI / 180.0;

    double dlon = lon2Rad - lon1Rad;
    double dlat = lat2Rad - lat1Rad;

    double a = pow(sin(dlat / 2), 2) + cos(lat1Rad) * cos(lat2Rad) * pow(sin(dlon / 2), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return radius * c;
}

double sphericalDistanceCartesian(const Vertex& v1, const Vertex& v2, double radius) {
    double dot_product = dotProduct(v1.normalized(), v2.normalized());
    double angle = std::acos(std::clamp(dot_product, -1.0, 1.0)); // Angle in radians
    return radius * angle; // Distance along the sphere's surface
}

double calculateBearing(double lat1, double lon1, double lat2, double lon2) {
    double lat1_rad = lat1 * M_PI / 180.0;
    double lon1_rad = lon1 * M_PI / 180.0;
    double lat2_rad = lat2 * M_PI / 180.0;
    double lon2_rad = lon2 * M_PI / 180.0;

    // Calculate the difference in longitudes
    double dLon = lon2_rad - lon1_rad;

    // Calculate bearing using the formula:
    // θ = atan2(sin(Δlong)*cos(lat2), cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))
    double x = sin(dLon) * cos(lat2_rad);
    double y = (cos(lat1_rad) * sin(lat2_rad) -
               (sin(lat1_rad) * cos(lat2_rad) * cos(dLon)));

    // Calculate the initial bearing in radians
    double initial_bearing = atan2(x, y);

    // Convert from radians to degrees (0-360)
    double initial_bearing_deg = initial_bearing * 180.0 / M_PI;
    double compass_bearing = fmod(initial_bearing_deg + 360.0, 360.0);

    return compass_bearing;
}

std::vector<Face*> getAdjacentFaces(const std::vector<Face>& all_faces, const Face& face) {
    std::vector<Face*> adjacent_faces;
    for (const Face& f : all_faces) {
        if (f == face) continue;
        
        // Check if faces share at least one vertex
        if (f.a == face.a || f.a == face.b || f.a == face.c ||
            f.b == face.a || f.b == face.b || f.b == face.c ||
            f.c == face.a || f.c == face.b || f.c == face.c) {
            adjacent_faces.push_back(const_cast<Face*>(&f));
        }
    }
    return adjacent_faces;
}

std::vector<Vertex> findSphericalNeighborsThreaded(const std::vector<Vertex>& vertices, const std::vector<Face>& faces, int vertexID, double maxDistanceKM, double radius) {
    const Vertex& center = vertices[vertexID];
    std::vector<Vertex> all_neighbors_with_duplicates;

    std::vector<Vertex> process_face = [&](const Face& face) {
        std::vector<Vertex> local_neighbors;
        bool isInFace = (face.a == vertices[center]) || (face.b == vertices[center]) || (face.c == vertices[center]);

        if (isInFace) {
            if (face.a == vertices[center]) {
                local_neighbors.push_back(face.b);
                local_neighbors.push_back(face.c);
            } else if (face.b == vertices[center]) {
                local_neighbors.push_back(face.a);
                local_neighbors.push_back(face.c);
            } else {
                local_neighbors.push_back(face.a);
                local_neighbors.push_back(face.b);
            }
        }
        return local_neighbors;
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback if concurrency is not detectable
    std::vector<std::future<std::vector<Vertex>>> futures;
    size_t faces_per_thread = (faces.size() + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_index = i * faces_per_thread;
        size_t end_index = std::min(start_index + faces_per_thread, faces.size());
        futures.push_back(std::async(std::launch::async, [&, start_index, end_index]() {
            std::vector<Vertex> thread_neighbors;
            for (size_t j = start_index; j < end_index; ++j) {
                std::vector<Vertex> face_neighbors = process_face(faces[j]);
                thread_neighbors.insert(thread_neighbors.end(), thread_neighbors.begin(), thread_neighbors.end());
            }
            return thread_neighbors;
        }));
    }

    for (auto& future : futures) {
        std::vector<Vertex> thread_result = future.get();
        all_neighbors_with_duplicates.insert(all_neighbors_with_duplicates.end(), thread_result.begin(), thread_result.end());
    }


    std::sort(all_neighbors_with_duplicates.begin(), all_neighbors_with_duplicates.end());
    all_neighbors_with_duplicates.erase(std::unique(all_neighbors_with_duplicates.begin(), all_neighbors_with_duplicates.end()), all_neighbors_with_duplicates.end());

    std::vector<Vertex> finalNeighbors;
    for (const Vertex& v : all_neighbors_with_duplicates) {
        double dist = sphericalDistanceCartesian(v, center, radius);
        if (dist <= maxDistanceKM) {
            finalNeighbors.push_back(v);
        }
    }
    return finalNeighbors;
}

double calculateSlope(std::vector<Vertex> vertices, std::vector<Face> faces, int vertexID, double radius){
    std::vector<Vertex> neighbors = findSphericalNeighborsThreaded(vertices, faces, vertexID, 1000.0, radius); // radius is placeholder here, adjust if needed, max_distance_km=100
    if (neighbors.empty()) {
        return 0.0;
    }

    Vertex vertex = vertices[vertexID];
    std::vector<Vertex> faceNorms;

    for (const auto& face : faces) {
        bool is_part_of_face = false;
        if (face.a == vertex || face.b == vertex || face.c == vertex) {
            is_part_of_face = true;
        }

        if (is_part_of_face) {
            Vertex v1 = face.b - face.a;
            Vertex v2 = face.c - face.a;
            Vertex normal = crossProduct(v1, v2).normalized();
            faceNorms.push_back(normal);
        }
    }

    if (faceNorms.empty()) return 0.0;

    Vertex averageNormal(0, 0, 0);
    for (const auto& normal : faceNorms) {
        averageNormal = averageNormal + normal;
    }
    averageNormal = averageNormal.normalized();

    Vertex radV = vertex.normalized();
    double dotProd = dotProduct(averageNormal, radV);
    double dotProdClipped = std::clamp(dotProd, -1.0, 1.0);

    return std::acos(dotProdClipped);
}

WorldState initializeWorld(size_t subdivisions = 3, double elevationRange = 10000.0, double totalWaterZL = 1386.0) {
    auto [vertices, faces, vertexIndices] = generate_icosphere(subdivisions, elevationRange);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> elevationDistrib(-elevationRange / 2.0, elevationRange / 2.0);
    std::uniform_real_distribution<> tempDistrib(-20.0, 30.0); // Temperature in Celsius
    std::uniform_real_distribution<> offsetPercentage(-0.01, 0.01);
    
    // Convert zettaliters to cubic meters (1 ZL = 1e18 m³)
    const double totalWaterM3 = totalWaterZL * 1e18;
    
    WorldState world(vertices, faces, vertexIndices, elevationRange);

    // Calculate total surface area for water distribution
    double totalSurfaceArea = 0.0;
    for (const Face& face : world.faces) {
        totalSurfaceArea += face.area(world.radius);
    }

    // Initialize vertices with elevation
    for (Vertex& vertex : world.vertices) {
        vertex.elevation = elevationDistrib(gen);
    }

    // Distribute water - using a 70/30 split between surface water and groundwater
    const double surfaceWaterFraction = 0.7;
    const double groundwaterFraction = 0.3;
    
    const double totalSurfaceWaterM3 = totalWaterM3 * surfaceWaterFraction;
    const double totalGroundwaterM3 = totalWaterM3 * groundwaterFraction;
    
    // First pass to calculate elevation-based water distribution weights
    std::vector<double> faceWeights(world.faces.size());
    double totalWeight = 0.0;
    
    for (size_t i = 0; i < world.faces.size(); ++i) {
        const Face& face = world.faces[i];
        double avgElev = (face.a.elevation + face.b.elevation + face.c.elevation) / 3.0;
        
        // Lower elevations get more water weight
        faceWeights[i] = 1.0 / (1.0 + std::exp(avgElev / (elevationRange * 0.1)));
        totalWeight += faceWeights[i] * face.area(world.radius);
    }
    
    // Second pass to distribute water
    for (size_t i = 0; i < world.faces.size(); ++i) {
        Face& face = world.faces[i];
        // Directly access vertex elevations using face.a, face.b, face.c
        double sumElevation = face.a.elevation + face.b.elevation + face.c.elevation;  
        double vertexAverage = sumElevation / 3.0; // 3 vertices per face
        face.initAverage = vertexAverage;

        double offset = offsetPercentage(gen) * elevationRange;
        face.average_elevation = vertexAverage + offset;

        face.average_elevation = std::clamp(face.average_elevation, -elevationRange / 2.0, elevationRange / 2.0);

        
        double faceArea = face.area(world.radius);
        
        // Calculate water amounts for this face
        double faceWaterFraction = (faceWeights[i] * faceArea) / totalWeight;
        double faceSurfaceWater = totalSurfaceWaterM3 * faceWaterFraction;
        double faceGroundwater = totalGroundwaterM3 * faceWaterFraction;
        
        // Distribute to vertices (simple even distribution)
        face.a.surface_water += faceSurfaceWater / 3.0;
        face.b.surface_water += faceSurfaceWater / 3.0;
        face.c.surface_water += faceSurfaceWater / 3.0;
        
        face.a.groundwater += faceGroundwater / 3.0;
        face.b.groundwater += faceGroundwater / 3.0;
        face.c.groundwater += faceGroundwater / 3.0;
        
        // Update world totals
        world.total_surface_water += faceSurfaceWater;
        
        // Set face averages
        face.surface_water = faceSurfaceWater;
        
        
        // Initialize atmospheric water (small fraction of surface water)
        face.average_atmospheric_water = faceSurfaceWater * 0.01;
        world.total_atmospheric_water += face.average_atmospheric_water;
        
        // Temperature
        face.temperature = tempDistrib(gen);
    }

    return world;
}

void simulateSurfaceWaterFlow(WorldState& world) {
    // Simple water flow simulation based on elevation and existing water
    for (size_t i = 0; i < world.vertices.size(); ++i) {
        Vertex& vertex = world.vertices[i];
        if (vertex.surface_water <= 0) continue;

        // Find neighboring vertices
        auto neighbors = findSphericalNeighborsThreaded(world.vertices, world.faces, i, 1000.0, world.radius);
        
        if (neighbors.empty()) continue;

        // Find the lowest neighbor
        Vertex* lowest_neighbor = nullptr;
        double lowest_elevation = vertex.elevation;
        
        for (Vertex& neighbor : neighbors) {
            double neighbor_elevation = neighbor.elevation + (neighbor.surface_water * 0.1); // Water increases effective elevation
            if (neighbor_elevation < lowest_elevation) {
                lowest_elevation = neighbor_elevation;
                lowest_neighbor = &neighbor;
            }
        }

        // Move some water to the lowest neighbor
        if (lowest_neighbor) {
            double flow_amount = std::min(vertex.surface_water * 0.1, 10.0); // Move up to 10% or 10 units
            vertex.surface_water -= flow_amount;
            lowest_neighbor->surface_water += flow_amount;
            
            // Update world totals (though they should remain the same)
            world.total_surface_water -= flow_amount;
            world.total_surface_water += flow_amount;
        }
    }
}

void simulateAtmosphericWater(WorldState& world) {
    // Simple atmospheric water (clouds) simulation based on temperature and existing water
    for (Face& face : world.faces) {
        // Evaporation from surface water to atmosphere
        double evaporation_rate = 0.01 * (1.0 + face.temperature / 30.0); // Higher temp = more evaporation
        double total_evaporation = 0.0;
        
        // Evaporate from each vertex in the face
        for (Vertex* vertex : {&face.a, &face.b, &face.c}) {
            double evap_amount = std::min(vertex->surface_water * evaporation_rate, 1.0);
            vertex->surface_water -= evap_amount;
            face.average_atmospheric_water += evap_amount / 3.0;
            total_evaporation += evap_amount;
        }
        
        world.total_surface_water -= total_evaporation;
        world.total_atmospheric_water += total_evaporation;

        // Precipitation when atmospheric water is high
        if (face.average_atmospheric_water > 50.0 && face.temperature < 20.0) {
            double precipitation = face.average_atmospheric_water * 0.1;
            face.average_atmospheric_water -= precipitation;
            
            // Distribute precipitation to vertices
            double per_vertex = precipitation / 3.0;
            face.a.surface_water += per_vertex;
            face.b.surface_water += per_vertex;
            face.c.surface_water += per_vertex;
            
            world.total_atmospheric_water -= precipitation;
            world.total_surface_water += precipitation;
        }

        // Atmospheric water movement (simple diffusion)
        for (Face* neighbor_face : getAdjacentFaces(world.faces, face)) {
            if (neighbor_face->average_atmospheric_water < face.average_atmospheric_water) {
                double transfer = (face.average_atmospheric_water - neighbor_face->average_atmospheric_water) * 0.05;
                face.average_atmospheric_water -= transfer;
                neighbor_face->average_atmospheric_water += transfer;
            }
        }
    }
}

WorldState step(WorldState world) {
    // Simulate water flow
    simulateSurfaceWaterFlow(world);
    
    // Simulate atmospheric water
    simulateAtmosphericWater(world);
    
    // Update face averages
    for (Face& face : world.faces) {
        double sumElevation = face.a.elevation + face.b.elevation + face.c.elevation;  
        double vertexAverage = sumElevation / 3.0; // 3 vertices per face
        face.average_elevation = face.average_elevation - face.initAverage + vertexAverage;
        face.surface_water = (face.a.surface_water + face.b.surface_water + face.c.surface_water) / 3.0;
    }
    return world;
}

std::vector<WorldState> run_simulation(size_t subdivisions=3, double elevationRange=10000.0, int steps = 15, double totalWaterZL = 1386.0) {
    WorldState world = initializeWorld(subdivisions, elevationRange, totalWaterZL);
    std::vector<WorldState> worldhistory;
    worldhistory.push_back(world); // Store initial state

    for (int i = 0; i < steps; ++i) {
        world = step(world);
        worldhistory.push_back(world);
    }
    return worldhistory;
}

PYBIND11_MODULE(icosphere, m) {
    m.doc() = "Icosphere generation and utility library";

    py::class_<WorldState>(m, "WorldState")
        .def(py::init<std::vector<Vertex>, std::vector<Face>, std::map<Vertex, size_t>, double>())
        .def_readwrite("vertices", &WorldState::vertices)
        .def_readwrite("faces", &WorldState::faces)
        .def_readwrite("vertex_indices", &WorldState::vertexIndices)
        .def_readwrite("total_surface_water", &WorldState::total_surface_water)
        .def_readwrite("total_ground_water", &WorldState::total_ground_water)
        .def_readwrite("total_atmospheric_water", &WorldState::total_atmospheric_water)
        .def_readwrite("radius", &WorldState::radius);

    m.def("run_simulation", &run_simulation, "Run the world simulation for a given number of steps.",
          py::arg("subdivisions"), py::arg("elevationRange"), py::arg("steps"), py::arg("totalWaterZL"));

    py::class_<Vertex>(m, "Vertex")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vertex::x)
        .def_readwrite("y", &Vertex::y)
        .def_readwrite("z", &Vertex::z)
        .def_readwrite("elevation", &Vertex::elevation)
        .def("__repr__", [](const Vertex &v) {
            return "<icosphere.Vertex x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
            ", elevation=" + std::to_string(v.elevation) + ">";
        });


    py::class_<Face>(m, "Face")
        .def(py::init<Vertex, Vertex, Vertex>())
        .def_readwrite("a", &Face::a)
        .def_readwrite("b", &Face::b)
        .def_readwrite("c", &Face::c)
        .def_readwrite("average_elevation", &Face::average_elevation)
        .def_readwrite("surface_water", &Face::surface_water)
        .def_readwrite("average_atmospheric_water", &Face::average_atmospheric_water)
        .def_readwrite("temperature", &Face::temperature)
        .def("__repr__", [](const Face &f) {
            return "<icosphere.Face a=" + std::to_string(f.a.x) + "," + std::to_string(f.a.y) + "," + std::to_string(f.a.z) +
                   " b=" + std::to_string(f.b.x) + "," + std::to_string(f.b.y) + "," + std::to_string(f.b.z) +
                   " c=" + std::to_string(f.c.x) + "," + std::to_string(f.c.y) + "," + std::to_string(f.c.z) +
                   " average_elevation=" + std::to_string(f.average_elevation) + ">";
        });
}