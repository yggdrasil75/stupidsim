#include <vector>
#include "shape.cpp"
#include "utils.cpp"
#include <complex>
#include <unordered_map>

const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;

struct worldState {
	std::vector<Vertex> vertices;
	std::vector<Face> faces;
	float radius;
	int subdivisions;
	worldState(float radius, int subdivisions) {
		shapeBase();
		subdivide();
	}
	void shapeBase() {
		throw NotImplementedException();
	}
	void subdivide() {
		throw NotImplementedException();
	}
};

struct icosphere : worldState {
	icosphere(float radius, int subdivisions) : worldState(radius, subdivisions) { }
	void shapeBase() {
		std::vector<Vertex> verticesRaw = {
			Vertex(-1, PHI, 0), Vertex(1, PHI, 0), Vertex(-1, -PHI, 0), Vertex(1, PHI, 0),
			Vertex(0, -1, PHI), Vertex(0, 1, PHI), Vertex(0, -1, -PHI), Vertex(0, 1, -PHI),
			Vertex(PHI, 0, -1), Vertex(PHI, 0, 1), Vertex(-PHI, 0, -1), Vertex(-PHI, 0, 1),
		};

		for (int i = 0; i < verticesRaw.size(); i++){
			verticesRaw[i].normalize();
			vertices.push_back(verticesRaw[i]);
		}

		faces = {
			Face(0, 11, 5), Face(0, 5, 1), Face(0, 1, 7), Face(0, 7, 10), Face(0, 10, 11),
			Face(1, 5, 9), Face(5, 11, 4), Face(11, 10, 2), Face(10, 7, 6), Face(7, 1, 8),
			Face(3, 9, 4), Face(3, 4, 2), Face(3, 2, 6), Face(3, 6, 8), Face(3, 8, 9),
			Face(4, 9, 5), Face(2, 4, 11), Face(6, 2, 10), Face(8, 6, 7), Face(9, 8, 1)
		};
	};
	void subdivide() {
		
		std::unordered_map<Vertex, uint32_t> vertex_to_index;
		for (uint32_t i = 0; i < vertices.size(); i++) {
			vertex_to_index[vertices[i]] = i;
		}
	
		for (uint64_t _ = 0; _ < subdivisions; _++) {
			std::vector<Face> new_faces;
			new_faces.reserve(faces.size() * 4);
	
			// Map from edge (pair of vertex indices) to new vertex index
			std::unordered_map<uint64_t, uint64_t> edge_vertices;
			auto get_or_create_midpoint = [&](uint32_t i1, uint32_t i2) -> uint32_t {
				// Always order the indices consistently
				if (i1 > i2) std::swap(i1, i2);
				
				uint64_t key = (static_cast<uint64_t>(i1) << 32) | i2;
				
				if (auto it = edge_vertices.find(key); it != edge_vertices.end()) {
					return it->second;
				}
				
				const Vertex& v1 = vertices[i1];
				const Vertex& v2 = vertices[i2];
				Vertex mid = (v1 + v2).normalize();  // Normalize immediately
				
				// Check if this vertex already exists
				if (auto it = vertex_to_index.find(mid); it != vertex_to_index.end()) {
					edge_vertices[key] = it->second;
					return it->second;
				}
				
				vertices.push_back(mid);
				uint32_t new_index = vertices.size() - 1;
				vertex_to_index[mid] = new_index;
				edge_vertices[key] = new_index;
				return new_index;
			};
			for (const Face& face : faces) {
				int facea = face.vertices[0];
				int faceb = face.vertices[1];
				int facec = face.vertices[2];
				const Vertex& a = vertices[facea];
				const Vertex& b = vertices[faceb];
				const Vertex& c = vertices[facec];
				uint64_t mid_ab = get_or_create_midpoint(facea, faceb);
				uint64_t mid_bc = get_or_create_midpoint(faceb, facec);
				uint64_t mid_ca = get_or_create_midpoint(facec, facea);
				new_faces.emplace_back(facea, mid_ab, mid_ca);
				new_faces.emplace_back(mid_ab, faceb, mid_bc);
				new_faces.emplace_back(mid_ca, mid_bc, facec);
				new_faces.emplace_back(mid_ab, mid_bc, mid_ca);
			}
			faces = std::move(new_faces);
	
			for (Vertex& v : vertices) {
				v.normalize();
			}
		}
	}
};