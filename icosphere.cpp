#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Vertex {
    double x, y, z;
    Vertex(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}
    Vertex operator+(const Vertex& other) const {
        return Vertex(x + other.x, y + other.y, z + other.z);
    }
    Vertex operator/(double scalar) const {
        return Vertex(x / scalar, y / scalar, z / scalar);
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

struct Face {
    Vertex a, b, c;
    Face(Vertex a, Vertex b, Vertex c) : a(a), b(b), c(c) {}
};

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
            uint64_t key_ab = (static_cast<uint64_t>(vertex_to_index[a]) << 32) | vertex_to_index[b];
            auto it_ab = edge_vertices.find(key_ab);
            if (it_ab == edge_vertices.end()) {
                mid_ab = (a + b) / 2.0;
                edge_vertices[key_ab] = mid_ab;
                vertices.push_back(mid_ab);
                vertex_to_index[mid_ab] = vertices.size() - 1;
            } else {
                mid_ab = it_ab->second;
            }
            
            // Similar for BC and CA edges...
            
            // Create 4 new faces
            new_faces.emplace_back(a, mid_ab, mid_ca);
            new_faces.emplace_back(mid_ab, b, mid_bc);
            new_faces.emplace_back(mid_ca, mid_bc, c);
            new_faces.emplace_back(mid_ab, mid_bc, mid_ca);
        }
        
        faces = std::move(new_faces);
    }
    
    // Normalize all vertices
    for (Vertex& v : vertices) {
        v = v.normalized();
    }
    
    return faces;
}

std::tuple<std::vector<Vertex>, std::vector<Face>, std::map<Vertex, size_t>> generate_icosphere(size_t subdivisions=3, double radius=1.0) {
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
        Face(vertices[0], vertices[11], vertices[5]),
        Face(vertices[0], vertices[5], vertices[1]),
        Face(vertices[0], vertices[1], vertices[7]),
        Face(vertices[0], vertices[7], vertices[10]),
        Face(vertices[0], vertices[10], vertices[11]),
        Face(vertices[1], vertices[5], vertices[9]),
        Face(vertices[5], vertices[11], vertices[4]),
        Face(vertices[11], vertices[10], vertices[2]),
        Face(vertices[10], vertices[7], vertices[6]),
        Face(vertices[7], vertices[1], vertices[8]),
        Face(vertices[3], vertices[9], vertices[4]),
        Face(vertices[3], vertices[4], vertices[2]),
        Face(vertices[3], vertices[2], vertices[6]),
        Face(vertices[3], vertices[6], vertices[8]),
        Face(vertices[3], vertices[8], vertices[9]),
        Face(vertices[4], vertices[9], vertices[5]),
        Face(vertices[2], vertices[4], vertices[11]),
        Face(vertices[6], vertices[2], vertices[10]),
        Face(vertices[8], vertices[6], vertices[7]),
        Face(vertices[9], vertices[8], vertices[1])
    };

    faces = subdivide_icosphere(subdivisions, vertices, faces);

    // Collect all unique vertices and create face indices
    std::vector<Vertex> unique_vertices;
    std::map<Vertex, size_t> vertex_indices;
    std::vector<Face> unique_faces;

    for (const Face& face : faces) {
        // Add vertices if they don't exist
        for (const Vertex& v : {face.a, face.b, face.c}) {
            if (vertex_indices.find(v) == vertex_indices.end()) {
                vertex_indices[v] = unique_vertices.size();
                unique_vertices.push_back(v);
            }
        }
        
        // Add face indices
        unique_faces.push_back(face);
    }

    // Scale vertices by radius
    for (Vertex& v : unique_vertices) {
        v.x *= radius;
        v.y *= radius;
        v.z *= radius;
    }

    return {unique_vertices, unique_faces, vertex_indices};
}

std::tuple<double, double> cartesianLatLon(Vertex vertex){
    _Float64 lat_rad = asinf64(vertex.z / sqrtf64(powf64(vertex.x, 2) + powf64(vertex.y, 2) + powf64(vertex.z, 2)));
    _Float64 lat_deg = lat_rad * (180.0 / M_PI);

    double lon_rad = atan2f64(vertex.y, vertex.x);
    double lon_deg = lon_rad * (180.0 / M_PI);

    return {lat_deg, lon_deg};
}

Vertex latLonCartesian(_Float64 lat, _Float64 lon, double radius){
    _Float64 lonRad = lon * (M_PI / 180);
    _Float64 latRad = lat * (M_PI / 180);

    double x = radius * cosf64(latRad) * cosf64(lonRad);
    double y = radius * cosf64(latRad) * sinf64(lonRad);
    double z = radius * sinf64(latRad);

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

double haversineDistanceVertex(Vertex v1, Vertex v2, double radius){
    double dlon = v2.x - v1.x;
    double dlat = v2.y - v1.y;
    double a = pow(sin(dlat / 2), 2) + cos(v1.y) * cos(v2.y) * pow(sin(dlon / 2), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return radius * c;
}

std::vector<Vertex> findSphericalNeighbors(std::vector<Vertex> vertices, std::map<int, Face> faces,
                     int vertexID, double maxDistanceKM, double radius) {
    std::vector<Vertex> neighbors;
    const Vertex& center = vertices[vertexID];
    
    // Find all vertices in adjacent faces
    for (const auto& facepair : faces) {
        Face face = facepair.second;
        // Check if the center vertex is part of this face
        bool isInFace = (face.a == center) || (face.b == center) || (face.c == center);
        
        if (isInFace) {
            // Add the other two vertices of the face
            if (face.a == center) {
                neighbors.push_back(face.b);
                neighbors.push_back(face.c);
            } 
            else if (face.b == center) {
                neighbors.push_back(face.a);
                neighbors.push_back(face.c);
            } 
            else { // face.c == center
                neighbors.push_back(face.a);
                neighbors.push_back(face.b);
            }
        }
    }
    
    // Remove duplicates (requires operator< for Vertex)
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    
    // Filter by distance
    std::vector<Vertex> finalNeighbors;
    for (const Vertex& v : neighbors) {
        double dist = haversineDistanceVertex(v, center, radius);
        if (dist <= maxDistanceKM) {
            finalNeighbors.push_back(v);
        }
    }
    
    return finalNeighbors;
}

PYBIND11_MODULE(icosphere, m) {
    m.def("generate_icosphere", &generate_icosphere, 
          "Generate an icosphere mesh",
          py::arg("subdivisions") = 3,
          py::arg("radius") = 1.0);

    m.def("cartesianLatLon", &cartesianLatLon, "convert vertex to lat/lon", py::arg("vertex"));

    m.def("latLonCartesian", &latLonCartesian, "convert lat/lon to cartesian", py::arg("lat"), py::arg("lon"), py::arg("radius"));

    m.def("HaversineDistance", &HaversineDistance, "calculates the distance between 2 lat/lon points", 
            py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"), py::arg("radius"));

    m.def("haversineDistanceVertex", &haversineDistanceVertex, "uses cartesian to get teh distance between 2 points",
            py::arg("v1"), py::arg("v2"), py::arg("radius"));

    m.def("findSphericalNeighbors", &findSphericalNeighbors, "gets neighbors within distance",
            py::arg("vertices"), py::arg("faces"), py::arg("centerID"), py::arg("maxDistanceKM"), py::arg("radius"));

    py::class_<Vertex>(m, "Vertex")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vertex::x)
        .def_readwrite("y", &Vertex::y)
        .def_readwrite("z", &Vertex::z);

    py::class_<Face>(m, "Face")
        .def_readwrite("a", &Face::a)
        .def_readwrite("b", &Face::b)
        .def_readwrite("c", &Face::c);
}