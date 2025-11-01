#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include "timing_decorator.hpp"

const float EPSILON = 0.00000000001;

//classes and structs

class Vec3 {
public:
    double x, y, z;
    
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    inline double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    inline Vec3 normalize() const {
        double n = norm();
        return Vec3(x/n, y/n, z/n);
    }
    
    inline Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    inline double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    inline Vec3 operator+(float scalar) const {
        return Vec3(x + scalar, y + scalar, z + scalar);
    }

    inline Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    inline Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    inline Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    inline Vec3 operator*(const Vec3& scalar) const {
        return Vec3(x * scalar.x, y * scalar.y, z * scalar.z);
    }

    inline friend Vec3 operator*(float scalar, const Vec3& vec) {
        return Vec3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }

    inline Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }
    
    inline Vec3 operator/(const Vec3& scalar) const {
        return Vec3(x / scalar.x, y / scalar.y, z / scalar.z);
    }

    inline friend Vec3 operator/(float scalar, const Vec3& vec) {
        return Vec3(vec.x / scalar, vec.y / scalar, vec.z / scalar);
    }

    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    bool operator<(const Vec3& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    
    struct Hash {
        size_t operator()(const Vec3& v) const {
            size_t h1 = std::hash<double>()(std::round(v.x * 1000.0));
            size_t h2 = std::hash<double>()(std::round(v.y * 1000.0));
            size_t h3 = std::hash<double>()(std::round(v.z * 1000.0));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    Vec3& safe_inverse_dir(float epsilon = 1e-6f) {
        x = (std::abs(x) > epsilon) ? x : std::copysign(epsilon, x);
        y = (std::abs(y) > epsilon) ? y : std::copysign(epsilon, y);
        z = (std::abs(z) > epsilon) ? z : std::copysign(epsilon, z);
        return *this;
    }

    Vec3 sign() const {
        return Vec3(
            (x > 0) ? 1 : ((x < 0) ? -1 : 0),
            (y > 0) ? 1 : ((y < 0) ? -1 : 0),
            (z > 0) ? 1 : ((z < 0) ? -1 : 0)
        );
    }

    Vec3 abs() {
        return Vec3(std::abs(x), std::abs(y), std::abs(z));
    }
};

class Vec4 {
public:
    double x, y, z, w;
    
    Vec4(double x = 0, double y = 0, double z = 0, double w = 0) : x(x), y(y), z(z), w(w) {}
    
    inline double norm() const {
        return std::sqrt(x*x + y*y + z*z + w*w);
    }
    
    inline Vec4 normalize() const {
        double n = norm();
        return Vec4(x/n, y/n, z/n, w/n);
    }
    
    inline std::array<Vec4, 6> wedge(const Vec4& other) const {
        return {
            Vec4(0, x*other.y - y*other.x, 0, 0),           // xy-plane
            Vec4(0, 0, x*other.z - z*other.x, 0),           // xz-plane  
            Vec4(0, 0, 0, x*other.w - w*other.x),           // xw-plane
            Vec4(0, 0, y*other.z - z*other.y, 0),           // yz-plane
            Vec4(0, 0, 0, y*other.w - w*other.y),           // yw-plane
            Vec4(0, 0, 0, z*other.w - w*other.z)            // zw-plane
        };
    }
    inline double dot(const Vec4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
    
    inline Vec4 operator+(const Vec4& other) const {
        return Vec4(x + other.x, y + other.y, z + other.z, w + other.w);
    }
    
    inline Vec4 operator-(const Vec4& other) const {
        return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
    }
    
    inline Vec4 operator*(double scalar) const {
        return Vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }
    
    inline Vec4 operator/(double scalar) const {
        return Vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }
    
    bool operator==(const Vec4& other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
    
    bool operator<(const Vec4& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        if (z != other.z) return z < other.z;
        return w < other.w;
    }
    
    // Additional useful methods for 4D vectors
    inline Vec4 homogenize() const {
        if (w == 0) return *this;
        return Vec4(x/w, y/w, z/w, 1.0);
    }
    
    inline Vec3 xyz() const {
        return Vec3(x, y, z);
    }
    
    struct Hash {
        size_t operator()(const Vec4& v) const {
            size_t h1 = std::hash<double>()(std::round(v.x * 1000.0));
            size_t h2 = std::hash<double>()(std::round(v.y * 1000.0));
            size_t h3 = std::hash<double>()(std::round(v.z * 1000.0));
            size_t h4 = std::hash<double>()(std::round(v.w * 1000.0));
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };
};

namespace std {
    template<>
    struct hash<Vec3> {
        size_t operator()(const Vec3& p) const {
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
    std::unordered_map<Vec3, std::vector<std::pair<Vec3, color4D>>> grid;
    
    Vec3 getVoxelKey(const Vec3& point) const {
        int voxel_x = std::floor(point.x / voxel_size);
        int voxel_y = std::floor(point.y / voxel_size);
        int voxel_z = std::floor(point.z / voxel_size);
        return Vec3(voxel_x * voxel_size, voxel_y * voxel_size, voxel_z * voxel_size);
    }

public:
    VoxelGrid(float size = 0.1f) : voxel_size(size) {}
    void addPoint(const Vec3& point, const color4D& color) {
        Vec3 voxel_key = getVoxelKey(point);
        grid[voxel_key].push_back(std::make_pair(point, color));
    }
    void addPoints(const std::vector<Vec3>& points, const std::vector<color4D>& colors) {
        if (points.size() != colors.size()) {
            std::cerr << "Error: Points and colors vectors must have the same size" << std::endl;
            return;
        }
        
        for (size_t i = 0; i < points.size(); ++i) {
            addPoint(points[i], colors[i]);
        }
    }
    std::vector<std::pair<Vec3, color4D>> getPointsInVoxel(const Vec3& point) const {
        TIME_FUNCTION;
        Vec3 voxel_key = getVoxelKey(point);
        auto it = grid.find(voxel_key);
        if (it != grid.end()) {
            return it->second;
        }
        return {};
    }
    std::vector<Vec3> getVoxelKeys() const {
        TIME_FUNCTION;
        std::vector<Vec3> keys;
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
        TIME_FUNCTION;
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
    std::tuple<std::vector<Vec3>, std::vector<color4D>> getAllPoints() const {
        TIME_FUNCTION;
        std::vector<Vec3> points;
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
    bool hasVoxel(const Vec3& point) const {
        Vec3 voxel_key = getVoxelKey(point);
        return grid.find(voxel_key) != grid.end();
    }
    bool removePoint(const Vec3& point) {
        Vec3 voxel_key = getVoxelKey(point);
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
        TIME_FUNCTION;
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

struct Image {
    int width;
    int height;
    std::vector<uint8_t> data; // RGBA format
    
    Image(int w, int h) : width(w), height(h), data(w * h * 4) {}
    
    // Helper methods
    uint8_t* pixel(int x, int y) {
        return &data[(y * width + x) * 4];
    }
    
    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        uint8_t* p = pixel(x, y);
        p[0] = r; p[1] = g; p[2] = b; p[3] = a;
    }
};

//noise functions
float fade(const float& a) {
    TIME_FUNCTION;
    return a * a * a * (10 + a * (-15 + a * 6));
}

float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
    TIME_FUNCTION;
    if (x < lowerlimit) return lowerlimit;
    if (x > upperlimit) return upperlimit;
    return x;
}

float pascalTri(const float& a, const float& b) {
    TIME_FUNCTION;
    int result = 1;
    for (int i = 0; i < b; ++i){
        result *= (a - 1) / (i + 1);
    }
    return result;
}

float genSmooth(int N, float x) {
    TIME_FUNCTION;
    x = clamp(x, 0, 1);
    float result = 0;
    for (int n = 0; n <= N; ++n){
        result += pascalTri(-N - 1, n) * pascalTri(2 * N + 1, N-1) * pow(x, N + n + 1);
    }
    return result;
}

float inverse_smoothstep(float x) {
    TIME_FUNCTION;
    return 0.5 - sin(asin(1.0 - 2.0 * x) / 3.0);
}

float lerp(const float& t, const float& a, const float& b) {
    TIME_FUNCTION;
    return a + t * (b - a);
}

float grad(const int& hash, const float& b, const float& c, const float& d) {
    TIME_FUNCTION;
    int h = hash & 15;
    float u = (h < 8) ? c : b;
    float v = (h < 4) ? b : ((h == 12 || h == 14) ? c : d);
    return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
}

float pnoise3d(const int p[512], const float& xf, const float& yf, const float& zf) {
    TIME_FUNCTION;
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

std::tuple<std::vector<Vec3>, std::vector<color4D>> noiseBatch(int num_points, float scale, int sp[]) {
    TIME_FUNCTION;
    std::vector<Vec3> points;
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

std::tuple<std::vector<Vec3>, std::vector<color4D>> genPointCloud(int numP, float scale, int seed) {
    TIME_FUNCTION;
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


//voxel stuff

Image render(int height, int width, Vec3 forward, Vec3 right, Vec3 up, Vec3 rayOrigin, Vec3 vbound,
            float vsize, VoxelGrid grid, int dims) {
    Image img = Image(height, width);

    float max_t = 50.0;
    int max_steps = 123;
    float max_dist = 25.0;
    float screen_height = height;
    float screen_width = width;

    float inv_w = 1.0 / width;
    float inv_h = 1.0 / height;
    float scr_w_half = screen_width * 0.5;
    float scr_h_half = screen_height * 0.5;

    for (int y = 0; y < height; y++) {
        float sy = 1.0 - (2.0 * y * inv_h) * scr_h_half;
        for (int x = 0; x < width; x++) {
            float sx = ((2.0 * x * inv_w) - 1.0) * scr_h_half;
            Vec3 ray_dir = forward + sx * right + sy * up;
            ray_dir = ray_dir.normalize();
            Vec3 cv = ((rayOrigin - vbound) / vsize);
            Vec3 inv_dir = ray_dir.safe_inverse_dir();
            Vec3 step = ray_dir.sign();
            Vec3 next_voxel_bound = (cv + step) * vsize + vbound;
            Vec3 t_max = (next_voxel_bound + step) * vsize + vbound;
            Vec3 t_delta = vsize / inv_dir.abs();
            float t = 0.0f;

            Vec4 accumulatedColor = Vec4(0,0,0,0);
            for (int i; i < max_steps; i++) {
                if (max_t > t) break;
                if (accumulatedColor.z >= 1.0) break;
                if (cv.x >= 0 && cv.x < dims &&
                    cv.y >= 0 && cv.y < dims && 
                    cv.z >= 0 && cv.z < dims) {
                    
                    if (grid[cv.x][cv.y][cv.z]) {
                        // Get the color and alpha from the color array
                        Vec4 voxel_color = color_array[cv.x][cv.y][cv.z];
                        Vec3 color_rgb = Vec3(voxel_color.w, voxel_color.x, voxel_color.y);  // w=r, x=g, y=b
                        float alpha = voxel_color.z;  // z is alpha
                        
                        // Apply alpha compositing: front-to-back
                        if (alpha > 0) {
                            // Weight by current transparency
                            // Use accumulated_color.z for accumulated alpha
                            float weight = alpha * (1.0f - accumulated_color.z);
                            accumulated_color.x += color_rgb.x * weight;  // green
                            accumulated_color.y += color_rgb.y * weight;  // blue  
                            accumulated_color.w += color_rgb.z * weight;  // red (stored in w)
                            accumulated_color.z += weight;  // accumulate alpha in z component
                        }
                    }
                }
            }
        }
        
    }

    return img;
}

int main() {    
    std::cout << "Generating point cloud" << std::endl;
    auto [points, colors] = genPointCloud(150000, 5.0, 43);
    std::cout << "Generating done" << std::endl;
    
    
    VoxelGrid voxel_grid(0.2f);
    
    std::cout << "Adding points to voxel grid..." << std::endl;
    voxel_grid.addPoints(points, colors);
    
    voxel_grid.printStats();
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    
    return 0;
}
