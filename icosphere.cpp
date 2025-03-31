#include <vector>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;

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
};

struct Face {
    Vertex a, b, c;
    Face(Vertex a, Vertex b, Vertex c) : a(a), b(b), c(c) {}
};

pair<vector<Vertex>, vector<vector<int32_t>>> generate_icosphere(int subdivisions=3, double radius=1.0) {
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

    // Subdivide the mesh
    for (int i = 0; i < subdivisions; ++i) {
        std::vector<Face> new_faces;
        std::map<std::pair<Vertex, Vertex>, Vertex> edge_vertices;

        for (const Face& face : faces) {
            // Get edge vertices
            Vertex edge_midpoints[3];
			Vertex edges[3] = {face.a, face.b, face.c};
            for (int i = 0; i < 3; ++i) {
                Vertex v1 = face[i];
                Vertex v2 = face[(i+1)%3];
                pair<Vertex, Vertex> key = v1 < v2 ? make_pair(v1, v2): make_pair(v2, v1);
                
                if (edge_vertices.find(key) == edge_vertices.end()) {
                    // Create new vertex at midpoint
                    Vertex mid = (v1 + v2) / 2.0;
                    mid = mid.normalized();
					edge_vertices[key] = mid;
                    //edge_vertices[key] = vertices.size();
                    //vertices.push_back(mid);
                }
                edge_midpoints.push_back(edge_vertices[key]);
            }

            // Create 4 new faces
            Vertex a = face.a, b = face.b, c = face.c;
            Vertex d = edge_midpoints[0], e = edge_midpoints[1], f = edge_midpoints[2];
            new_faces.push_back({a, d, f});
            new_faces.push_back({d, b, e});
            new_faces.push_back({f, e, c});
            new_faces.push_back({d, e, f});
        }

        faces = new_faces;
    }

    // Collect all unique vertices
    vector<Vertex> unique_vertices;
	// Collect all Vertex Index values
    map<Vertex, size_t> vertex_indices;
	// Collect Faces
    vector<std::vector<size_t>> face_indices;

    for (const Face& face : faces) {
        // Add vertices if they don't exist
        for (const Vertex& v : {face.a, face.b, face.c}) {
            if (vertex_indices.find(v) == vertex_indices.end()) {
                vertex_indices[v] = unique_vertices.size();
                unique_vertices.push_back(v);
            }
        }
        
        // Add face indices
        face_indices.push_back({
            vertex_indices[face.v0],
            vertex_indices[face.v1],
            vertex_indices[face.v2]
        });
    }

    // Scale vertices by radius
    for (Vertex& v : unique_vertices) {
        v.x *= radius;
        v.y *= radius;
        v.z *= radius;
    }

    return {unique_vertices, face_indices};
}

PYBIND11_MODULE(icosphere, m) {
	m.def("generate_icosphere", &generate_icosphere, "Generates an icosphere mesh with detail based on subdivisions",
	py::arg("subdivisions") = 3, py::arg("radius") = 1.0);

	py::class_<Vertex>(m, "Vertex").def_readwrite("x", &Vertex::x).def_readwrite("y", &Vertex::y).def_readwrite("z", &Vertex::z);
}

int main() {
}
