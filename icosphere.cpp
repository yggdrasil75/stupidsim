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
};

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


std::tuple<std::vector<Vertex>, std::vector<Face>> generate_icosphere(size_t subdivisions=3, double radius=1.0) {
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

    // Collect all unique vertices and create face indices
    std::vector<Vertex> unique_vertices;
    std::map<Vertex, size_t> vertex_indices;
    std::vector<Face> unique_faces;

    faces = subdivide_icosphere(subdivisions, vertices, faces);

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

    return {unique_vertices, unique_faces};
}



PYBIND11_MODULE(icosphere, m) {
    m.def("generate_icosphere", &generate_icosphere, 
          "Generate an icosphere mesh",
          py::arg("subdivisions") = 3,
          py::arg("radius") = 1.0);
    
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