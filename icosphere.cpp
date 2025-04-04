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
    double surfaceWater;  // Water on the surface (rivers, lakes, etc.)
    double groundwater;    // Water underground
    Vertex(double x=0, double y=0, double z=0) : x(x), y(y), z(z), elevation(0.0), surfaceWater(0.0), groundwater(0.0) {}
    Vertex(double x, double y, double z, double elevation) : x(x), y(y), z(z), elevation(elevation), surfaceWater(0.0), groundwater(0.0) {}
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
    double dot(const Vertex& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    Vertex cross(const Vertex& other){
        return Vertex(y * other.z - z * other.y,
                      z * other.x - x * other.z,
                      x * other.y - y * other.x);
    }
};

double dotProduct(const Vertex& a, const Vertex& b) {
    //assert(!std::isnan(a.x) && !std::isnan(a.y) && !std::isnan(a.z));
    //assert(!std::isnan(b.x) && !std::isnan(b.y) && !std::isnan(b.z));
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Face {
    size_t a, b, c;  // Vertex indices
    double average_elevation;
    double initAverage;
    double surfaceWater;
    double groundWater;
    double average_atmospheric_water;
    double temperature;
    double friction;

    Face(size_t a, size_t b, size_t c) : a(a), b(b), c(c), average_elevation(0.0), initAverage(0.0), 
        surfaceWater(0.0), groundWater(0.0), average_atmospheric_water(0.0), temperature(0.0), friction(0.0) {}
    Face(size_t a, size_t b, size_t c, double averageElevation) : a(a), b(b), c(c), average_elevation(averageElevation) {}

    bool operator<(const Face& other) const {
        if (a != other.a) return a < other.a;
        if (b != other.b) return b < other.b;
        return c < other.c;
    }
    bool operator==(const Face& other) const {
        std::array<size_t, 3> this_vertices = {a, b, c};
        std::array<size_t, 3> other_vertices = {other.a, other.b, other.c};
        std::sort(this_vertices.begin(), this_vertices.end());
        std::sort(other_vertices.begin(), other_vertices.end());

        return this_vertices[0] == other_vertices[0] &&
               this_vertices[1] == other_vertices[1] &&
               this_vertices[2] == other_vertices[2];
    }
    bool operator!=(const Face& other) const {
        return !(*this == other);
    }
    double area(const std::vector<Vertex>& vertices, double radius) const {
        return (radius * M_PI * 4.0) / vertices.size();
        // const Vertex& va = vertices[a];
        // const Vertex& vb = vertices[b];
        // const Vertex& vc = vertices[c];
    
    
        // double dotproda = dotProduct(va, vb);
        // double dotprodb = dotProduct(vb, vc);
        // double dotprodc = dotProduct(vc, va);
        // std::cout << "a, b, c: " << va.x << ", " << va.y << ", " << va.z << ", " << vb.x << ", " << vb.y << ", " << vb.z << ", " << vc.x << ", " << vc.y << ", " << vc.z << " dot prods: " << dotproda << ", " << dotprodb << ", " << dotprodc << std::endl;
        // double s = (dotproda + dotprodb + dotprodc) / 2.0;
        // double tan_half_E_squared = tan(s / 2.0) * tan((s - dotproda) / 2.0) * tan((s-dotprodb) / 2.0) * tan((s - dotprodc) / 2.0);
        // double E = 4.0 * atan(sqrt(tan_half_E_squared));
        // std::cout << "s: " << s << " gibberish: " << tan_half_E_squared << " E: " << E << std::endl;
        // return E * radius * radius;

    }
};

// struct Plate {
//     std::vector<size_t> vertices;
    
// }

struct WorldState {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::map<Vertex, size_t> vertexIndices;
    double total_surface_water;
    double total_ground_water;
    double total_atmospheric_water;
    double radius;
    double timestepSeconds;
    double gravity;

    double timestepHour(){
        return timestepSeconds / 3600.0;
    }
    double timestepDay(){
        return timestepSeconds / 86400.0;
    }
    double timestepYear(){
        return timestepSeconds / 31556952.0;
    }

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

std::vector<Face> subdivide_icosphere(size_t subdivisions, std::vector<Vertex>& vertices, std::vector<Face>& faces) {
    // Create a map from Vertex to its index in the vertices vector
    std::unordered_map<Vertex, uint32_t> vertex_to_index;
    for (uint32_t i = 0; i < vertices.size(); i++) {
        vertex_to_index[vertices[i]] = i;
    }

    for (size_t _ = 0; _ < subdivisions; _++) {
        std::vector<Face> new_faces;
        new_faces.reserve(faces.size() * 4);

        // Map from edge (pair of vertex indices) to new vertex index
        std::unordered_map<uint64_t, size_t> edge_vertices;

        for (const Face& face : faces) {
            const Vertex& a = vertices[face.a];
            const Vertex& b = vertices[face.b];
            const Vertex& c = vertices[face.c];

            // Get or create midpoints for each edge
            size_t mid_ab, mid_bc, mid_ca;

            // Edge AB
            uint64_t key_ab = (static_cast<uint64_t>(std::min(face.a, face.b)) << 32 | std::max(face.a, face.b));
            auto it_ab = edge_vertices.find(key_ab);
            if (it_ab == edge_vertices.end()) {
                Vertex mid = (a + b) / 2.0;
                mid = mid.normalized();
                vertices.push_back(mid);
                mid_ab = vertices.size() - 1;
                edge_vertices[key_ab] = mid_ab;
                vertex_to_index[mid] = mid_ab;
            } else {
                mid_ab = it_ab->second;
            }

            // Edge BC
            uint64_t key_bc = (static_cast<uint64_t>(std::min(face.b, face.c)) << 32 | std::max(face.b, face.c));
            auto it_bc = edge_vertices.find(key_bc);
            if (it_bc == edge_vertices.end()) {
                Vertex mid = (b + c) / 2.0;
                mid = mid.normalized();
                vertices.push_back(mid);
                mid_bc = vertices.size() - 1;
                edge_vertices[key_bc] = mid_bc;
                vertex_to_index[mid] = mid_bc;
            } else {
                mid_bc = it_bc->second;
            }

            // Edge CA
            uint64_t key_ca = (static_cast<uint64_t>(std::min(face.c, face.a)) << 32 | std::max(face.c, face.a));
            auto it_ca = edge_vertices.find(key_ca);
            if (it_ca == edge_vertices.end()) {
                Vertex mid = (c + a) / 2.0;
                mid = mid.normalized();
                vertices.push_back(mid);
                mid_ca = vertices.size() - 1;
                edge_vertices[key_ca] = mid_ca;
                vertex_to_index[mid] = mid_ca;
            } else {
                mid_ca = it_ca->second;
            }

            new_faces.emplace_back(face.a, mid_ab, mid_ca);
            new_faces.emplace_back(mid_ab, face.b, mid_bc);
            new_faces.emplace_back(mid_ca, mid_bc, face.c);
            new_faces.emplace_back(mid_ab, mid_bc, mid_ca);
        }

        faces = std::move(new_faces);

        for (Vertex& v : vertices) {
            v = v.normalized();
        }
    }

    return faces;
}

std::vector<Face> getCrossedFaces(const WorldState& world, size_t startIdx, size_t endIdx) {
    std::vector<Face> crossedFaces;
    
    // Get the start and end vertices
    const Vertex& start = world.vertices[startIdx];
    const Vertex& end = world.vertices[endIdx];
    
    // Early exit if start and end are the same
    if (startIdx == endIdx) return crossedFaces;
    
    // Create the geodesic direction vector
    Vertex direction = end - start;
    
    // Track current position and face
    size_t currentFaceIdx = SIZE_MAX;
    Vertex currentPos = start;
    
    // Find the initial face (one that contains both startIdx and is crossed by the direction)
    for (size_t i = 0; i < world.faces.size(); ++i) {
        const Face& face = world.faces[i];
        
        // Check if face contains the start vertex
        if (face.a != startIdx && face.b != startIdx && face.c != startIdx) continue;
        
        // Get the other two vertices of the face
        size_t v1 = (face.a == startIdx) ? face.b : face.a;
        size_t v2 = (face.c == startIdx) ? face.b : face.c;
        if (v1 == startIdx) v1 = face.c;
        
        const Vertex& vert1 = world.vertices[v1];
        const Vertex& vert2 = world.vertices[v2];
        
        // Check if direction points between these two vertices
        Vertex edge1 = vert1 - start;
        Vertex edge2 = vert2 - start;
        
        Vertex cross1 = edge1.normalized().cross(direction.normalized());
        Vertex cross2 = direction.normalized().cross(edge2.normalized());
        
        if (cross1.dot(cross2) > 0) {
            currentFaceIdx = i;
            crossedFaces.push_back(face);
            break;
        }
    }
    
    if (currentFaceIdx == SIZE_MAX) return crossedFaces; // No initial face found
    
    // Traverse through faces until we reach the end vertex
    while (true) {
        const Face& currentFace = world.faces[currentFaceIdx];
        
        // Check if we've reached the end vertex
        if (currentFace.a == endIdx || currentFace.b == endIdx || currentFace.c == endIdx) {
            break;
        }
        
        // Find the edge we cross
        size_t nextFaceIdx = SIZE_MAX;
        
        // For each edge of the current face, check if the geodesic crosses it
        std::array<std::pair<size_t, size_t>, 3> edges = {{
            {currentFace.a, currentFace.b},
            {currentFace.b, currentFace.c},
            {currentFace.c, currentFace.a}
        }};
        
        for (const auto& edge : edges) {
            // Skip edges that include the start vertex (we already handled the initial face)
            if (edge.first == startIdx || edge.second == startIdx) continue;
            
            const Vertex& v1 = world.vertices[edge.first];
            const Vertex& v2 = world.vertices[edge.second];
            
            // Calculate intersection between geodesic and edge
            Vertex edgeVec = v2 - v1;
            Vertex normal = direction.cross(edgeVec);
            
            if (normal.length() < 1e-10) continue; // parallel, no intersection
            
            // Find adjacent face sharing this edge
            for (size_t i = 0; i < world.faces.size(); ++i) {
                if (i == currentFaceIdx) continue;
                
                const Face& candidate = world.faces[i];
                bool sharesEdge = false;
                
                // Check if candidate shares this edge
                if ((candidate.a == edge.first || candidate.b == edge.first || candidate.c == edge.first) &&
                    (candidate.a == edge.second || candidate.b == edge.second || candidate.c == edge.second)) {
                    sharesEdge = true;
                }
                
                if (sharesEdge) {
                    // Check if direction points into this face
                    size_t thirdVertex = (candidate.a != edge.first && candidate.a != edge.second) ? candidate.a :
                                       (candidate.b != edge.first && candidate.b != edge.second) ? candidate.b :
                                       candidate.c;
                    
                    const Vertex& thirdVert = world.vertices[thirdVertex];
                    Vertex faceNormal = (v2 - v1).cross(thirdVert - v1);
                    
                    if (direction.dot(faceNormal) > 0) {
                        nextFaceIdx = i;
                        break;
                    }
                }
            }
            
            if (nextFaceIdx != SIZE_MAX) break;
        }
        
        if (nextFaceIdx == SIZE_MAX || nextFaceIdx == currentFaceIdx) {
            break; // No next face found or stuck in loop
        }
        
        // Add the new face to our list and continue
        crossedFaces.push_back(world.faces[nextFaceIdx]);
        currentFaceIdx = nextFaceIdx;
    }
    
    return crossedFaces;
}

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

    // Create vertex index map
    std::map<Vertex, size_t> vertex_indices;
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertex_indices[vertices[i]] = i;
    }

    // Scale vertices by radius
    for (Vertex& v : vertices) {
        v.x *= radius;
        v.y *= radius;
        v.z *= radius;
    }

    return {vertices, faces, vertex_indices};
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

std::vector<size_t> findSphericalNeighborsThreaded(WorldState world, size_t vertexID, double maxDistanceKM) {
    const Vertex& center = world.vertices[vertexID];
    std::vector<size_t> neighbors;
    std::unordered_set<size_t> visited;
    std::vector<size_t> to_process;

    // Start with direct neighbors
    for (const Face& face : world.faces) {
        if (face.a == vertexID || face.b == vertexID || face.c == vertexID) {
            if (face.a != vertexID && !visited.count(face.a)) {
                visited.insert(face.a);
                to_process.push_back(face.a);
            }
            if (face.b != vertexID && !visited.count(face.b)) {
                visited.insert(face.b);
                to_process.push_back(face.b);
            }
            if (face.c != vertexID && !visited.count(face.c)) {
                visited.insert(face.c);
                to_process.push_back(face.c);
            }
        }
    }

    // Expand to neighbors within distance
    while (!to_process.empty()) {
        size_t current_id = to_process.back();
        to_process.pop_back();

        double dist = sphericalDistanceCartesian(world.vertices[current_id], center, world.radius);
        if (dist <= maxDistanceKM) {
            neighbors.push_back(current_id);

            // Add this vertex's neighbors to processing queue
            for (const Face& face : world.faces) {
                if (face.a == current_id || face.b == current_id || face.c == current_id) {
                    if (face.a != current_id && !visited.count(face.a)) {
                        visited.insert(face.a);
                        to_process.push_back(face.a);
                    }
                    if (face.b != current_id && !visited.count(face.b)) {
                        visited.insert(face.b);
                        to_process.push_back(face.b);
                    }
                    if (face.c != current_id && !visited.count(face.c)) {
                        visited.insert(face.c);
                        to_process.push_back(face.c);
                    }
                }
            }
        }
    }

    return neighbors;
}

std::vector<Face> findSphericalNeighborFacesThreaded(WorldState world, Face face, double maxDistanceKM) {
    //const Vertex& center = vertices[vertexID];
    std::vector<Vertex> vertices = world.vertices;
    const Vertex& center = Vertex((vertices[face.a].x + vertices[face.b].x + vertices[face.c].x) / 3.0, 
                                    (vertices[face.a].y + vertices[face.b].y + vertices[face.c].y) / 3.0, 
                                    (vertices[face.a].z + vertices[face.b].z + vertices[face.c].z) / 3.0);
    std::vector<Face> neighbor_faces;

    for (size_t i = 0; i < world.faces.size(); ++i) {
        const Face& cFace = world.faces[i];
        
        // Get all three vertices of the face
        const Vertex& v1 = vertices[cFace.a];
        const Vertex& v2 = vertices[cFace.b];
        const Vertex& v3 = vertices[cFace.c];

        // Calculate distance from center to each vertex
        double d1 = sphericalDistanceCartesian(v1, center, world.radius);
        double d2 = sphericalDistanceCartesian(v2, center, world.radius);
        double d3 = sphericalDistanceCartesian(v3, center, world.radius);

        // Check if any vertex is within max distance
        if (d1 <= maxDistanceKM || d2 <= maxDistanceKM || d3 <= maxDistanceKM) {
            neighbor_faces.push_back(cFace);
            continue;
        }
    }

    return neighbor_faces;
}

double calculateSlope(const std::vector<Vertex>& vertices, const std::vector<Face>& faces, size_t vertexID, double radius) {
    const Vertex& vertex = vertices[vertexID];
    std::vector<Vertex> faceNormals;

    // Find all faces containing this vertex
    for (const Face& face : faces) {
        if (face.a == vertexID || face.b == vertexID || face.c == vertexID) {
            const Vertex& v1 = vertices[face.b] - vertices[face.a];
            const Vertex& v2 = vertices[face.c] - vertices[face.a];
            Vertex normal = crossProduct(v1, v2).normalized();
            faceNormals.push_back(normal);
        }
    }

    if (faceNormals.empty()) return 0.0;

    // Calculate average normal
    Vertex averageNormal(0, 0, 0);
    for (const Vertex& normal : faceNormals) {
        averageNormal = averageNormal + normal;
    }
    averageNormal = averageNormal.normalized();

    // Calculate angle between average normal and radial vector
    Vertex radial = vertex.normalized();
    double dotProd = dotProduct(averageNormal, radial);
    double angle = std::acos(std::clamp(dotProd, -1.0, 1.0));

    return angle * (180.0 / M_PI); // Convert to degrees
}

WorldState initializeWorld(size_t subdivisions = 3, double elevationRange = 10000.0, double totalWaterZL = 1386.0) {
    auto [vertices, faces, vertexIndices] = generate_icosphere(subdivisions, elevationRange);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> elevationDistrib(-elevationRange / 2.0, elevationRange / 2.0);
    std::uniform_real_distribution<> tempDistrib(-20.0, 30.0); // Temperature in Celsius
    std::uniform_real_distribution<> offsetPercentage(-0.01, 0.01);
    std::uniform_real_distribution<> frictioncoefficientoffset(0.0, 0.1);
    
    // Convert zettaliters to cubic meters (1 ZL = 1e18 m³)
    const long double totalWaterM3 = totalWaterZL * 1e18;
    
    WorldState world(vertices, faces, vertexIndices, elevationRange);

    // Calculate total surface area for water distribution
    double totalSurfaceArea = 0.0;
    for (const Face& face : world.faces) {
        totalSurfaceArea += face.area(world.vertices, world.radius);
    }

    // Initialize vertices with elevation
    for (Vertex& vertex : world.vertices) {
        vertex.elevation = elevationDistrib(gen);
    }

    const double surfaceWaterFraction = 0.9725;
    const double groundwaterFraction = 0.027;
    const double atmosphericWaterFraction = 0.0005;
    
    const double totalSurfaceWaterM3 = totalWaterM3 * surfaceWaterFraction;
    const double totalGroundwaterM3 = totalWaterM3 * groundwaterFraction;
    const double totalAtmosphericwaterM3 = totalWaterM3 * atmosphericWaterFraction;
    
    // First pass to calculate elevation-based water distribution weights
    std::vector<double> faceWeights(world.faces.size());
    double totalWeight = 0.0;
    
    for (size_t i = 0; i < world.faces.size(); ++i) {
        const Face& face = world.faces[i];
        double avgElev = (world.vertices[face.a].elevation + 
                         world.vertices[face.b].elevation + 
                         world.vertices[face.c].elevation) / 3.0;
        
        // Lower elevations get more water weight
        faceWeights[i] = 1.0 / (1.0 + std::exp(avgElev / (elevationRange * 0.1)));
        totalWeight += faceWeights[i] * face.area(world.vertices, world.radius);
    }
    
    
    // Second pass to distribute water
    for (size_t i = 0; i < world.faces.size(); ++i) {
        Face& face = world.faces[i];
        face.initAverage = (world.vertices[face.a].elevation + world.vertices[face.b].elevation + world.vertices[face.c].elevation) / 3.0;
        double offset = offsetPercentage(gen) * elevationRange;
        face.average_elevation = face.initAverage + offset;
        face.average_elevation = std::clamp(face.average_elevation, -elevationRange / 2.0, elevationRange / 2.0);
        double faceArea = face.area(world.vertices, world.radius);
        
        // Calculate water amounts for this face
        double faceWaterFraction = (faceWeights[i] * faceArea) / totalWeight;
        
        double faceSurfaceWater = totalSurfaceWaterM3 * faceWaterFraction;
        //std::cout << "face area is currently: " << faceArea << " or better still: " << face.area(world.vertices, world.radius) << std::endl;
        double faceGroundwater = totalGroundwaterM3 * faceWaterFraction;
        
        // Distribute to vertices (simple even distribution)
        world.vertices[face.a].surfaceWater += faceSurfaceWater / 3.0;
        world.vertices[face.b].surfaceWater += faceSurfaceWater / 3.0;
        world.vertices[face.c].surfaceWater += faceSurfaceWater / 3.0;
        
        world.vertices[face.a].groundwater += faceGroundwater / 3.0;
        world.vertices[face.b].groundwater += faceGroundwater / 3.0;
        world.vertices[face.c].groundwater += faceGroundwater / 3.0;
        
        // Update world totals
        world.total_surface_water += faceSurfaceWater;
        
        // Set face averages
        face.surfaceWater = faceSurfaceWater;
        face.groundWater = faceGroundwater;
        
        
        // Initialize atmospheric water (small fraction of surface water)
        face.average_atmospheric_water = faceSurfaceWater * 0.01;
        world.total_atmospheric_water += face.average_atmospheric_water;
        
        // Temperature
        face.temperature = tempDistrib(gen);
        // TODO: replace later with something to calculate a realistic friction. ie: ratio of 3 types of rock
        face.friction = frictioncoefficientoffset(gen); 
        //std::cout << "face " << i << " is currently flooded with " << faceSurfaceWater << std::endl;
    }
    world.gravity = 9.81;
    return world;
}

void simulateSurfaceWaterFlow(WorldState& world) {
    for (size_t i = 0; i < world.vertices.size(); ++i) {
        Vertex& vertex = world.vertices[i];
        if (vertex.surfaceWater <= 0) continue;

        // Find neighboring vertices
        std::vector<size_t> neighbors = findSphericalNeighborsThreaded(world, i, 2000.0);
        
        if (neighbors.empty()) continue;
        double outflow = 0.0;
        std::vector<std::pair<size_t, double>> potentialFlows;
        // Find the lowest neighbor
        // size_t lowest_neighbor = i;
        // double lowest_elevation = vertex.elevation + (vertex.surfaceWater * 0.1);
        
        for (size_t neighborID : neighbors) {
            Vertex& neighbor = world.vertices[neighborID];
            std::vector<Face> crossed = getCrossedFaces(world, i, neighborID);
            double crossedfriction = 0.0;
            for (Face crossedface : crossed){
                crossedfriction += crossedface.friction;
            }
            double facefrictions = crossedfriction / crossed.size();
            double sourceHead = vertex.elevation + (vertex.surfaceWater * 0.001);
            double targetHead = neighbor.elevation + (neighbor.surfaceWater * 0.001);
            if (sourceHead <= targetHead) continue;
            double distance = sphericalDistanceCartesian(vertex, neighbor, world.radius);
            double slope = (sourceHead - targetHead) / distance;
            double flow = (1.0/facefrictions) * sqrt(slope) * (vertex.surfaceWater * 0.001);
            double flowAmount = flow * world.timestepHour() * 1000.0;
            potentialFlows.emplace_back(neighborID, flowAmount);
            outflow += flowAmount;
        }

        double maxOut = vertex.surfaceWater * 0.5;
        if (outflow > maxOut) {
            double scaleFactor = maxOut / outflow;
            for (std::pair<size_t, double>& flow : potentialFlows) {
                flow.second *= scaleFactor;
            }
            outflow = maxOut;
        }
        
        for (const std::pair<size_t, double>& flow : potentialFlows) {
            double actual = std::min(flow.second, vertex.surfaceWater);
            vertex.surfaceWater -= actual;
            world.vertices[flow.first].surfaceWater += actual;
        }
    }
    for (Face& face : world.faces){
        face.surfaceWater = (world.vertices[face.a].surfaceWater + world.vertices[face.b].surfaceWater + world.vertices[face.c].surfaceWater) / 3.0;
    }
}

void simulateAtmosphericWater(WorldState& world) {
    // Simple atmospheric water (clouds) simulation based on temperature and existing water
    for (Face& face : world.faces) {
        Vertex& va = world.vertices[face.a];
        Vertex& vb = world.vertices[face.b];
        Vertex& vc = world.vertices[face.c];
        
        // Evaporation from surface water to atmosphere
        double evaporation_rate = 0.01 * (1.0 + face.temperature / 30.0); // Higher temp = more evaporation
        double total_evaporation = 0.0;
        
        // Evaporate from each vertex in the face
        for (Vertex* vertex : {&va, &vb, &vc}) {
            double evap_amount = std::min(vertex->surfaceWater * evaporation_rate, 1.0);
            vertex->surfaceWater -= evap_amount;
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
            va.surfaceWater += per_vertex;
            vb.surfaceWater += per_vertex;
            vc.surfaceWater += per_vertex;
            
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
        Vertex& va = world.vertices[face.a];
        Vertex& vb = world.vertices[face.b];
        Vertex& vc = world.vertices[face.c];
        
        double sumElevation = va.elevation + vb.elevation + vc.elevation;  
        double vertexAverage = sumElevation / 3.0; // 3 vertices per face
        face.average_elevation = face.average_elevation - face.initAverage + vertexAverage;
        //face.surfaceWater = (va.surfaceWater + vb.surfaceWater + vc.surfaceWater) / 3.0;
    }
    return world;
}

std::vector<WorldState> run_simulation(size_t subdivisions=3, double elevationRange=10000.0, 
                    int steps = 15, double totalWaterZL = 1386.0, double timestepSeconds = 3600.0) {
    WorldState world = initializeWorld(subdivisions, elevationRange, totalWaterZL);
    world.timestepSeconds = timestepSeconds;
    std::vector<WorldState> worldhistory;
    worldhistory.push_back(world); // Store initial state

    for (int i = 0; i < steps; ++i) {
        world = step(world);
        worldhistory.push_back(world);
        std::cout << "Simulating step " << i << "/" << steps << std::endl;
    }
    return worldhistory;
}

PYBIND11_MODULE(icosphere, m) {
    m.doc() = "Icosphere generation and utility library";

    py::class_<WorldState>(m, "WorldState")
        .def_readwrite("vertices", &WorldState::vertices)
        .def_readwrite("faces", &WorldState::faces)
        .def_readwrite("vertex_indices", &WorldState::vertexIndices)
        .def_readwrite("total_surface_water", &WorldState::total_surface_water)
        .def_readwrite("total_ground_water", &WorldState::total_ground_water)
        .def_readwrite("total_atmospheric_water", &WorldState::total_atmospheric_water);

    m.def("initializeWorld", &initializeWorld, "Generate an basic world",
          py::arg("subdivisions") = 3,
          py::arg("elevationRange") = 10000.0,
          py::arg("totalWaterZL") = 1386.0);

    m.def("run_simulation", &run_simulation, "Run the world simulation for a given number of steps.",
          py::arg("subdivisions") = 3,
          py::arg("elevationRange") = 10000.0,
          py::arg("steps") = 15,
          py::arg("totalWaterZL") = 1386.0,
          py::arg("timestepSeconds") = 3600.0);

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
        .def(py::init<size_t, size_t, size_t>())
        .def_readwrite("a", &Face::a)
        .def_readwrite("b", &Face::b)
        .def_readwrite("c", &Face::c)
        .def_readwrite("average_elevation", &Face::average_elevation)
        .def_readwrite("surfaceWater", &Face::surfaceWater)
        .def_readwrite("average_atmospheric_water", &Face::average_atmospheric_water)
        .def_readwrite("temperature", &Face::temperature)
        .def("__repr__", [](const Face &f) {
            return "<icosphere.Face a=" + std::to_string(f.a) + 
                   " b=" + std::to_string(f.b) + 
                   " c=" + std::to_string(f.c) + 
                   " average_elevation=" + std::to_string(f.average_elevation) + ">";
        });
}