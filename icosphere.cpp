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
    Vertex(double x=0, double y=0, double z=0) : x(x), y(y), z(z), elevation(0.0) {}
    Vertex(double x, double y, double z, double elevation) : x(x), y(y), z(z), elevation(elevation) {}
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

Vertex crossProduct(const Vertex& a, const Vertex& b) {
    return Vertex(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

double dotProduct(const Vertex& a, const Vertex& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Face {
    Vertex a, b, c;
    double average_elevation;

    Face(Vertex a, Vertex b, Vertex c) : a(a), b(b), c(c), average_elevation(0.0) {}
    Face(Vertex a, Vertex b, Vertex c, double averageElevation) : a(a), b(b), c(c), average_elevation(averageElevation) {}

    bool operator<(const Face& other) const {
      if(a != other.a) return a < other.a;
      if(b != other.b) return b < other.b;
      return c < other.c;
    }
    bool operator==(const Face& other) const {
        Vertex this_vertices[3] = {a, b, c};
        Vertex other_vertices[3] = {other.a, other.b, other.c};
        std::sort(this_vertices, this_vertices + 3);
        std::sort(other_vertices, other_vertices + 3);

        return this_vertices[0] == other_vertices[0] &&
               this_vertices[1] == other_vertices[1] &&
               this_vertices[2] == other_vertices[2];
    }
    bool operator!=(const Face& other) const {
        return !(*this == other);
    }
};

// Define a struct to hold the world state
struct WorldState {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::map<Vertex, size_t> vertexIndices;

    WorldState() = default;
    WorldState(const std::vector<Vertex>& v, const std::vector<Face>& f, const std::map<Vertex, size_t>& vi) : vertices(v), faces(f), vertexIndices(vi) {}
};

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
            const Vertex& a = face.a;
            const Vertex& b = face.b;
            const Vertex& c = face.c;

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
            new_faces.emplace_back(a, mid_ab, mid_ca);
            new_faces.emplace_back(mid_ab, b, mid_bc);
            new_faces.emplace_back(mid_ca, mid_bc, c);
            new_faces.emplace_back(mid_ab, mid_bc, mid_ca);
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
        Face(vertices[0], vertices[1], vertices[7]),
        Face(vertices[0], vertices[5], vertices[1]),
        Face(vertices[0], vertices[7], vertices[10]),
        Face(vertices[0], vertices[10], vertices[11]),
        Face(vertices[0], vertices[11], vertices[5]),
        Face(vertices[1], vertices[5], vertices[9]),
        Face(vertices[2], vertices[4], vertices[11]),
        Face(vertices[3], vertices[9], vertices[4]),
        Face(vertices[3], vertices[4], vertices[2]),
        Face(vertices[3], vertices[2], vertices[6]),
        Face(vertices[3], vertices[6], vertices[8]),
        Face(vertices[3], vertices[8], vertices[9]),
        Face(vertices[4], vertices[9], vertices[5]),
        Face(vertices[5], vertices[11], vertices[4]),
        Face(vertices[6], vertices[2], vertices[10]),
        Face(vertices[7], vertices[1], vertices[8]),
        Face(vertices[8], vertices[6], vertices[7]),
        Face(vertices[9], vertices[8], vertices[1]),
        Face(vertices[10], vertices[7], vertices[6]),
        Face(vertices[11], vertices[10], vertices[2])
    };

    faces = subdivide_icosphere(subdivisions, vertices, faces);

    // Collect all unique vertices and create face indices
    std::vector<Vertex> unique_vertices;
    std::map<Vertex, size_t> vertex_indices;
    std::vector<Face> unique_faces; // Store face indices

    for (const Face& face : faces) {
        std::array<uint32_t, 3> current_face_indices;
        int index_count = 0;
        for (const Vertex& v : {face.a, face.b, face.c}) {
            if (vertex_indices.find(v) == vertex_indices.end()) {
                vertex_indices[v] = unique_vertices.size();
                unique_vertices.push_back(v);
            }
            current_face_indices[index_count++] = vertex_indices[v];
        }
    }

    for (const Face& face : faces){
        unique_faces.emplace_back(
            unique_vertices[vertex_indices[face.a]],
            unique_vertices[vertex_indices[face.b]],
            unique_vertices[vertex_indices[face.c]]
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

WorldState initializeWorld(size_t subdivisions=3, double radius=1.0, double elevationRange=10000.0) {
    auto [vertices, faces, vertexIndices] = generate_icosphere(subdivisions, radius);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-elevationRange / 2.0, elevationRange / 2.0);

    for (Vertex& vertex : vertices){
        vertex.elevation = distrib(gen);
    }
    for (Face& face : faces) {
        face.average_elevation = distrib(gen);
    }

    return WorldState(vertices, faces, vertexIndices);
}

WorldState step(WorldState world) {
    // In this basic step, the world state doesn't change.
    // Erosion or other simulation logic would be implemented here in future steps.
    return world;
}

std::vector<WorldState> run_simulation(size_t subdivisions=3, double radius=1.0, double elevationRange=10000.0, int steps = 15) {
    WorldState world = initializeWorld(subdivisions, radius, elevationRange);
    std::vector<WorldState> worldhistory;
    WorldState current_world = world;
    worldhistory.push_back(current_world); // Store initial state

    for (int i = 0; i < steps; ++i) {
        current_world = step(current_world);
        worldhistory.push_back(current_world); // Store faces after each step
    }
    return worldhistory;
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

std::vector<Vertex> findSphericalNeighborsThreaded(const std::vector<Vertex>& vertices, const std::vector<Face>& faces,
                     int vertexID, double maxDistanceKM, double radius) {
    const Vertex& center = vertices[vertexID];
    std::vector<Vertex> all_neighbors_with_duplicates;

    auto process_face = [&](const Face& face) {
        std::vector<Vertex> local_neighbors;
        bool isInFace = (face.a == center) || (face.b == center) || (face.c == center);

        if (isInFace) {
            if (face.a == center) {
                local_neighbors.push_back(face.b);
                local_neighbors.push_back(face.c);
            } else if (face.b == center) {
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

PYBIND11_MODULE(icosphere, m) {
    m.doc() = "Icosphere generation and utility library";

    py::class_<WorldState>(m, "WorldState")
        .def(py::init<>())
        .def_readwrite("vertices", &WorldState::vertices)
        .def_readwrite("faces", &WorldState::faces)
        .def_readwrite("vertex_indices", &WorldState::vertexIndices);

    m.def("initializeWorld", &initializeWorld, "Generate an basic world",
          py::arg("subdivisions") = 3, py::arg("radius") = 1.0,
          py::arg("elevationRange") = 10000.0);

    m.def("step", &step, "Advance the world simulation by one step.", py::arg("world"));

    m.def("run_simulation", &run_simulation, "Run the world simulation for a given number of steps.",
          py::arg("subdivisions"), py::arg("radius"), py::arg("elevationRange"), py::arg("steps"));

    m.def("cartesianLatLon", &cartesianLatLon,
        "Convert a Cartesian vertex to latitude and longitude.", py::arg("vertex"));

    m.def("latLonCartesian", &latLonCartesian,
        "Convert latitude and longitude to a Cartesian vertex.", py::arg("lat"),
        py::arg("lon"), py::arg("radius"));

    m.def("HaversineDistance", &HaversineDistance,
        "Calculate the Haversine distance between two lat/lon points.",
            py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"), py::arg("radius"));

    m.def("sphericalDistanceCartesian", &sphericalDistanceCartesian,
        "Calculate spherical distance between two Cartesian vertices.",
            py::arg("v1"), py::arg("v2"), py::arg("radius"));

    m.def("calculateBearing", &calculateBearing, "Calculate the bearing between two lat/lon points.",
            py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"));

    m.def("findSphericalNeighbors", &findSphericalNeighborsThreaded,
        "Find neighboring vertices within a given spherical distance.",
            py::arg("vertices"), py::arg("faces"), py::arg("centerID"),
            py::arg("maxDistanceKM"), py::arg("radius"));

    m.def("calculateSlope", &calculateSlope,
        "Calculate the slope at a vertex based on surrounding face normals.",
            py::arg("vertices"), py::arg("faces"), py::arg("centerID"), py::arg("radius"));


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
        .def("__repr__", [](const Face &f) {
            return "<icosphere.Face a=" + std::to_string(f.a.x) + "," + std::to_string(f.a.y) + "," + std::to_string(f.a.z) +
                   " b=" + std::to_string(f.b.x) + "," + std::to_string(f.b.y) + "," + std::to_string(f.b.z) +
                   " c=" + std::to_string(f.c.x) + "," + std::to_string(f.c.y) + "," + std::to_string(f.c.z) +
                   " average_elevation=" + std::to_string(f.average_elevation) + ">";
        });
}