#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>

struct Point3D { 
    float x, y, z;
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
    Point3D& operator=(const float* data) {
        x = data[0];
        y = data[1];
        z = data[2];
        return *this;
    }
    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template<>
    struct hash<Point3D> {
        size_t operator()(const Point3D& p) const {
            return hash<float>()(p.x) ^ hash<float>()(p.y) ^ hash<float>()(p.z);
        }
    };
}

struct color4D {
    float r, g, b, a;
    color4D() : r(0), g(0), b(0), a(0) {}
    color4D(float r, float g, float b) : r(r), g(g), b(b), a(1) {}
    color4D(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
    color4D& operator=(const float* data) {
        r = data[0];
        g = data[1];
        b = data[2];
        a = data[3];
        return *this;
    }
};

class VoxelGrid {
private:
    float voxel_size;
    std::unordered_map<Point3D, std::vector<std::pair<Point3D, color4D>>> grid;
    
    Point3D getVoxelKey(const Point3D& point) const {
        int voxel_x = std::floor(point.x / voxel_size);
        int voxel_y = std::floor(point.y / voxel_size);
        int voxel_z = std::floor(point.z / voxel_size);
        return Point3D(voxel_x * voxel_size, voxel_y * voxel_size, voxel_z * voxel_size);
    }

public:
    VoxelGrid(float size = 0.1f) : voxel_size(size) {}
    void addPoint(const Point3D& point, const color4D& color) {
        Point3D voxel_key = getVoxelKey(point);
        grid[voxel_key].push_back(std::make_pair(point, color));
    }
    void addPoints(const std::vector<Point3D>& points, const std::vector<color4D>& colors) {
        if (points.size() != colors.size()) {
            std::cerr << "Error: Points and colors vectors must have the same size" << std::endl;
            return;
        }
        
        for (size_t i = 0; i < points.size(); ++i) {
            addPoint(points[i], colors[i]);
        }
    }
    std::vector<std::pair<Point3D, color4D>> getPointsInVoxel(const Point3D& point) const {
        Point3D voxel_key = getVoxelKey(point);
        auto it = grid.find(voxel_key);
        if (it != grid.end()) {
            return it->second;
        }
        return {};
    }
    std::vector<Point3D> getVoxelKeys() const {
        std::vector<Point3D> keys;
        for (const auto& pair : grid) {
            keys.push_back(pair.first);
        }
        return keys;
    }
    size_t getNumVoxels() const {
        return grid.size();
    }
    size_t getNumPoints() const {
        size_t total = 0;
        for (const auto& pair : grid) {
            total += pair.second.size();
        }
        return total;
    }
    void getVoxelStats(size_t& min_points, size_t& max_points, double& avg_points) const {
        if (grid.empty()) {
            min_points = max_points = avg_points = 0;
            return;
        }
        
        min_points = std::numeric_limits<size_t>::max();
        max_points = 0;
        size_t total_points = 0;
        
        for (const auto& pair : grid) {
            size_t count = pair.second.size();
            min_points = std::min(min_points, count);
            max_points = std::max(max_points, count);
            total_points += count;
        }
        
        avg_points = static_cast<double>(total_points) / grid.size();
    }
    std::tuple<std::vector<Point3D>, std::vector<color4D>> getAllPoints() const {
        std::vector<Point3D> points;
        std::vector<color4D> colors;
        
        for (const auto& voxel : grid) {
            for (const auto& point_color : voxel.second) {
                points.push_back(point_color.first);
                colors.push_back(point_color.second);
            }
        }
        
        return std::make_tuple(points, colors);
    }
    
    void clear() {
        grid.clear();
    }
    void setVoxelSize(float size) {
        voxel_size = size;
        clear();
    }
    
    float getVoxelSize() const {
        return voxel_size;
    }
    bool hasVoxel(const Point3D& point) const {
        Point3D voxel_key = getVoxelKey(point);
        return grid.find(voxel_key) != grid.end();
    }
    bool removePoint(const Point3D& point) {
        Point3D voxel_key = getVoxelKey(point);
        auto it = grid.find(voxel_key);
        if (it != grid.end()) {
            auto& points = it->second;
            for (auto pt_it = points.begin(); pt_it != points.end(); ++pt_it) {
                if (pt_it->first.x == point.x && pt_it->first.y == point.y && pt_it->first.z == point.z) {
                    points.erase(pt_it);
                    if (points.empty()) {
                        grid.erase(it);
                    }
                    return true;
                }
            }
        }
        return false;
    }
    void printStats() const {
        size_t min_pts, max_pts;
        double avg_pts;
        getVoxelStats(min_pts, max_pts, avg_pts);
        
        std::cout << "Voxel Grid Statistics:" << std::endl;
        std::cout << "  Voxel size: " << voxel_size << std::endl;
        std::cout << "  Number of voxels: " << getNumVoxels() << std::endl;
        std::cout << "  Total points: " << getNumPoints() << std::endl;
        std::cout << "  Points per voxel - Min: " << min_pts << ", Max: " << max_pts << ", Avg: " << avg_pts << std::endl;
    }
};

float fade(const float& a) {
    return a * a * a * (10 + a * (-15 + a * 6));
}

float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
  if (x < lowerlimit) return lowerlimit;
  if (x > upperlimit) return upperlimit;
  return x;
}

float pascalTri(const float& a, const float& b) {
    int result = 1;
    for (int i = 0; i < b; ++i){
        result *= (a - 1) / (i + 1);
    }
    return result;
}

float genSmooth(int N, float x) {
    x = clamp(x, 0, 1);
    float result = 0;
    for (int n = 0; n <= N; ++n){
        result += pascalTri(-N - 1, n) * pascalTri(2 * N + 1, N-1) * pow(x, N + n + 1);
    }
    return result;
}

float inverse_smoothstep(float x) {
  return 0.5 - sin(asin(1.0 - 2.0 * x) / 3.0);
}

float lerp(const float& t, const float& a, const float& b) {
    return a + t * (b - a);
}

float grad(const int& hash, const float& b, const float& c, const float& d) {
    int h = hash & 15;
    float u = (h < 8) ? c : b;
    float v = (h < 4) ? b : ((h == 12 || h == 14) ? c : d);
    return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
}

float pnoise3d(const int p[512], const float& xf, const float& yf, const float& zf) {
    int floorx = std::floor(xf);
    int floory = std::floor(yf);
    int floorz = std::floor(zf);
    int iX = floorx & 255;
    int iY = floory & 255;
    int iZ = floorz & 255;

    int x = xf - floorx;
    int y = yf - floory;
    int z = zf - floorz;

    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    int A  = p[iX] + iY;
    int AA = p[A] + iZ;
    int AB = p[A+1] + iZ;

    int B  = p[iX + 1] + iY;
    int BA = p[B] + iZ;
    int BB = p[B+1] + iZ;

    float f = grad(p[BA], x-1, y, z);
    float g = grad(p[AA], x, y, z);
    float h = grad(p[BB], x-1, y-1, z);
    float j = grad(p[BB], x-1, y-1, z);
    float k = grad(p[AA+1], x, y, z-1);
    float l = grad(p[BA+1], x-1, y, z-1);
    float m = grad(p[AB+1], x, y-1, z-1);
    float n = grad(p[BB+1], x-1, y-1, z-1);

    float o = lerp(u, m, n);
    float q = lerp(u, k, l);
    float r = lerp(u, h, j);
    float s = lerp(u, f, g);
    float t = lerp(v, q, o);
    float e = lerp(v, s, r);
    float d = lerp(w, e, t);
    return d;
}

std::tuple<std::vector<Point3D>, std::vector<color4D>> noiseBatch(int num_points, float scale, int sp[]) {
    std::vector<Point3D> points;
    std::vector<color4D> colors;
    points.reserve(num_points);
    colors.reserve(num_points);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-scale, scale);
    
    for (int i = 0; i < num_points; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        float z = dis(gen);
        
        float noise1 = pnoise3d(sp, (x * 0.5f), (y * 0.5f), (z * 0.5f));
        float noise2 = pnoise3d(sp, (x * 0.3f), (y * 0.3f), (z * 0.3f));
        float noise3 = pnoise3d(sp, (x * 0.7f), (y * 0.7f), (z * 0.7f));
        float noise4 = pnoise3d(sp, (x * 0.7f), (y * 0.7f), (z * 0.7f));
        
        if (noise1 > 0.1f) {
            float rt = (noise1 + 1.0f) * 0.5f;
            float gt = (noise2 + 1.0f) * 0.5f;
            float bt = (noise3 + 1.0f) * 0.5f;
            float at = (noise4 + 1.0f) * 0.5f;

            float maxV = std::max({rt, gt, bt});
            if (maxV > 0) {
                float r = rt / maxV;
                float g = gt / maxV;
                float b = bt / maxV;
                float a = at / maxV;
                points.push_back({x, y, z});
                colors.push_back({r, g, b, a});
            }
        }
    }
    return std::make_tuple(points, colors);
}

std::tuple<std::vector<Point3D>, std::vector<color4D>> genPointCloud(int numP, float scale, int seed) {
    int permutation[256];
    for (int i = 0; i < 256; ++i) {
        permutation[i] = i;
    }
    std::mt19937 rng(seed);
    std::shuffle(permutation, permutation+256, rng);
    int p[512];
    for (int i = 0; i < 256; ++i) {
        p[i] = permutation[i];
        p[i + 256] = permutation[i];
    }
    return noiseBatch(numP, scale, p);
}

int main() {    
    std::cout << "Generating point cloud" << std::endl;
    auto [points, colors] = genPointCloud(150000, 5.0, 43);
    std::cout << "Generating done" << std::endl;
    
    
    VoxelGrid voxel_grid(0.2f);
    
    std::cout << "Adding points to voxel grid..." << std::endl;
    voxel_grid.addPoints(points, colors);
    
    voxel_grid.printStats();
    
    
    return 0;
}