#include <vector>
#include <complex>
#include <algorithm>

struct Face {
	std::vector<int> vertices;
	Face(std::vector<int> othervertices) : vertices(othervertices) {}
	Face(int x, int y, int z) {
		vertices.insert(vertices.end(), {x, y, z});
	}
};

struct Vertex {
	float x, y, z;
	Vertex(double x, double y, double z) : x(x), y(y), z(z) {}

	Vertex normalize(){
		double len = std::sqrt(x * x + y * y + z * z);
		x = x / len;
		y = y / len;
		z = z / len;
		return *this;
	}

	Vertex operator+(const Vertex& other) const {
		return Vertex(x + other.x, y + other.y, z + other.z);
	}
};

namespace std {
    template<> struct hash<Vertex> {
        uint64_t operator()(const Vertex& v) const {
            uint64_t h1 = hash<double>{}(v.x);
            uint64_t h2 = hash<double>{}(v.y);
            uint64_t h3 = hash<double>{}(v.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}