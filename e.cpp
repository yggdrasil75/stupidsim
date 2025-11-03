#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <tuple>
#include "timing_decorator.hpp"
// #include "bmp_writer.hpp"

class Vec3;
class Vec4;

class Vec3 {
public:
    float x, y, z;
    
    // Constructors
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    // Arithmetic operations
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }
    
    // Dot product
    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // Cross product
    Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    // Length and normalization
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vec3 normalized() const {
        float len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }
    
    // Component-wise absolute value
    Vec3 abs() const {
        return Vec3(std::abs(x), std::abs(y), std::abs(z));
    }
    
    // Component-wise floor
    Vec3 floor() const {
        return Vec3(std::floor(x), std::floor(y), std::floor(z));
    }
    
    // Equality operator for use in hash maps
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    inline Vec3 operator*(const Vec3& scalar) const {
        return Vec3(x * scalar.x, y * scalar.y, z * scalar.z);
    }
    
    inline Vec3 operator/(const Vec3& scalar) const {
        return Vec3(x / scalar.x, y / scalar.y, z / scalar.z);
    }
};

// Hash function for Vec3 to use in unordered_map
namespace std {
    template<>
    struct hash<Vec3> {
        size_t operator()(const Vec3& v) const {
            return hash<float>()(v.x) ^ (hash<float>()(v.y) << 1) ^ (hash<float>()(v.z) << 2);
        }
    };
}

class Vec4 {
public:
    float r, g, b, a;
    
    // Constructors
    Vec4() : r(0), g(0), b(0), a(1.0f) {}
    Vec4(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}
    
    // Construct from Vec3 with alpha
    Vec4(const Vec3& rgb, float a = 1.0f) : r(rgb.x), g(rgb.y), b(rgb.z), a(a) {}
    
    // Arithmetic operations
    Vec4 operator+(const Vec4& other) const {
        return Vec4(r + other.r, g + other.g, b + other.b, a + other.a);
    }
    
    Vec4 operator-(const Vec4& other) const {
        return Vec4(r - other.r, g - other.g, b - other.b, a - other.a);
    }
    
    Vec4 operator*(float scalar) const {
        return Vec4(r * scalar, g * scalar, b * scalar, a * scalar);
    }
    
    Vec4 operator/(float scalar) const {
        return Vec4(r / scalar, g / scalar, b / scalar, a / scalar);
    }
    
    // Component-wise multiplication
    Vec4 operator*(const Vec4& other) const {
        return Vec4(r * other.r, g * other.g, b * other.b, a * other.a);
    }
    
    // Clamp values between 0 and 1
    Vec4 clamped() const {
        return Vec4(
            std::clamp(r, 0.0f, 1.0f),
            std::clamp(g, 0.0f, 1.0f),
            std::clamp(b, 0.0f, 1.0f),
            std::clamp(a, 0.0f, 1.0f)
        );
    }
    
    // Convert to Vec3 (ignoring alpha)
    Vec3 toVec3() const {
        return Vec3(r, g, b);
    }
    
    // Convert to 8-bit color values
    void toUint8(uint8_t& red, uint8_t& green, uint8_t& blue, uint8_t& alpha) const {
        red = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255);
        green = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255);
        alpha = static_cast<uint8_t>(std::clamp(a, 0.0f, 1.0f) * 255);
    }
    
    void toUint8(uint8_t& red, uint8_t& green, uint8_t& blue) const {
        red = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255);
        green = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255);
        blue = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255);
    }
};

class VoxelGrid {
private:
    std::unordered_map<Vec3, size_t> positionToIndex;
    std::vector<Vec3> positions;
    std::vector<Vec4> colors;
    
    Vec3 gridSize;

public:
    Vec3 voxelSize;
    VoxelGrid(const Vec3& size, const Vec3& voxelSize = Vec3(1, 1, 1)) 
        : gridSize(size), voxelSize(voxelSize) {}
    
    // Add a voxel at position with color
    void addVoxel(const Vec3& position, const Vec4& color) {
        Vec3 gridPos = worldToGrid(position);
        
        // Check if voxel already exists
        auto it = positionToIndex.find(gridPos);
        if (it == positionToIndex.end()) {
            // New voxel
            size_t index = positions.size();
            positions.push_back(gridPos);
            colors.push_back(color);
            positionToIndex[gridPos] = index;
        } else {
            // Update existing voxel (you might want to blend colors instead)
            colors[it->second] = color;
        }
    }
    
    // Get voxel color at position
    Vec4 getVoxel(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        auto it = positionToIndex.find(gridPos);
        if (it != positionToIndex.end()) {
            return colors[it->second];
        }
        return Vec4(0, 0, 0, 0); // Transparent black for empty voxels
    }
    
    // Check if position is occupied
    bool isOccupied(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        return positionToIndex.find(gridPos) != positionToIndex.end();
    }
    
    // Convert world coordinates to grid coordinates
    Vec3 worldToGrid(const Vec3& worldPos) const {
        return (worldPos / voxelSize).floor();
    }
    
    // Convert grid coordinates to world coordinates
    Vec3 gridToWorld(const Vec3& gridPos) const {
        return gridPos * voxelSize;
    }
    
    // Get all occupied positions
    const std::vector<Vec3>& getOccupiedPositions() const {
        return positions;
    }
    
    // Get all colors
    const std::vector<Vec4>& getColors() const {
        return colors;
    }
    
    // Get the mapping from position to index
    const std::unordered_map<Vec3, size_t>& getPositionToIndexMap() const {
        return positionToIndex;
    }
    
    // Get grid size
    const Vec3& getGridSize() const {
        return gridSize;
    }
    
    // Get voxel size
    const Vec3& getVoxelSize() const {
        return voxelSize;
    }
    
    // Clear the grid
    void clear() {
        positions.clear();
        colors.clear();
        positionToIndex.clear();
    }
};

class AmanatidesWooAlgorithm {
public:
    struct Ray {
        Vec3 origin;
        Vec3 direction;
        float tMax;
        
        Ray(const Vec3& origin, const Vec3& direction, float tMax = 1000.0f)
            : origin(origin), direction(direction.normalized()), tMax(tMax) {}
    };
    
    struct TraversalState {
        Vec3 currentVoxel;
        Vec3 tMax;
        Vec3 tDelta;
        Vec3 step;
        bool hit;
        float t;
        
        TraversalState() : hit(false), t(0) {}
    };
    
    // Initialize traversal state for a ray
    static TraversalState initTraversal(const Ray& ray, const Vec3& voxelSize) {
        TraversalState state;
        
        // Find the starting voxel
        Vec3 startPos = ray.origin / voxelSize;
        state.currentVoxel = startPos.floor();
        
        // Determine step directions and initialize tMax
        Vec3 rayDir = ray.direction;
        Vec3 invDir = Vec3(
            rayDir.x != 0 ? 1.0f / rayDir.x : std::numeric_limits<float>::max(),
            rayDir.y != 0 ? 1.0f / rayDir.y : std::numeric_limits<float>::max(),
            rayDir.z != 0 ? 1.0f / rayDir.z : std::numeric_limits<float>::max()
        );
        
        // Calculate step directions
        state.step = Vec3(
            rayDir.x > 0 ? 1 : (rayDir.x < 0 ? -1 : 0),
            rayDir.y > 0 ? 1 : (rayDir.y < 0 ? -1 : 0),
            rayDir.z > 0 ? 1 : (rayDir.z < 0 ? -1 : 0)
        );
        
        // Calculate tMax for each axis
        Vec3 nextVoxelBoundary = state.currentVoxel;
        if (state.step.x > 0) nextVoxelBoundary.x += 1;
        if (state.step.y > 0) nextVoxelBoundary.y += 1;
        if (state.step.z > 0) nextVoxelBoundary.z += 1;
        
        state.tMax = Vec3(
            (nextVoxelBoundary.x - startPos.x) * invDir.x,
            (nextVoxelBoundary.y - startPos.y) * invDir.y,
            (nextVoxelBoundary.z - startPos.z) * invDir.z
        );
        
        // Calculate tDelta
        state.tDelta = Vec3(
            state.step.x * invDir.x,
            state.step.y * invDir.y,
            state.step.z * invDir.z
        );
        
        state.hit = false;
        state.t = 0;
        
        return state;
    }
    
    // Traverse the grid along the ray
    static bool traverse(const Ray& ray, const VoxelGrid& grid, 
                        std::vector<Vec3>& hitVoxels, std::vector<float>& hitDistances,
                        int maxSteps = 1000) {
        TraversalState state = initTraversal(ray, grid.voxelSize);
        Vec3 voxelSize = grid.voxelSize;
        
        hitVoxels.clear();
        hitDistances.clear();
        
        for (int step = 0; step < maxSteps; step++) {
            // Check if current voxel is occupied
            Vec3 worldPos = grid.gridToWorld(state.currentVoxel);
            if (grid.isOccupied(worldPos)) {
                hitVoxels.push_back(state.currentVoxel);
                hitDistances.push_back(state.t);
            }
            
            // Find next voxel
            if (state.tMax.x < state.tMax.y) {
                if (state.tMax.x < state.tMax.z) {
                    state.currentVoxel.x += state.step.x;
                    state.t = state.tMax.x;
                    state.tMax.x += state.tDelta.x;
                } else {
                    state.currentVoxel.z += state.step.z;
                    state.t = state.tMax.z;
                    state.tMax.z += state.tDelta.z;
                }
            } else {
                if (state.tMax.y < state.tMax.z) {
                    state.currentVoxel.y += state.step.y;
                    state.t = state.tMax.y;
                    state.tMax.y += state.tDelta.y;
                } else {
                    state.currentVoxel.z += state.step.z;
                    state.t = state.tMax.z;
                    state.tMax.z += state.tDelta.z;
                }
            }
            
            // Check if we've exceeded maximum distance
            if (state.t > ray.tMax) {
                break;
            }
        }
        
        return !hitVoxels.empty();
    }
};

class BMPWriter {
private:
    #pragma pack(push, 1)
    struct BMPHeader {
        uint16_t signature = 0x4D42; // "BM"
        uint32_t fileSize;
        uint16_t reserved1 = 0;
        uint16_t reserved2 = 0;
        uint32_t dataOffset = 54;
    };
    
    struct BMPInfoHeader {
        uint32_t headerSize = 40;
        int32_t width;
        int32_t height;
        uint16_t planes = 1;
        uint16_t bitsPerPixel = 24;
        uint32_t compression = 0;
        uint32_t imageSize;
        int32_t xPixelsPerMeter = 0;
        int32_t yPixelsPerMeter = 0;
        uint32_t colorsUsed = 0;
        uint32_t importantColors = 0;
    };
    #pragma pack(pop)

public:
    static bool saveBMP(const std::string& filename, const std::vector<uint8_t>& pixels, int width, int height) {
        BMPHeader header;
        BMPInfoHeader infoHeader;
        
        int rowSize = (width * 3 + 3) & ~3; // 24-bit, padded to 4 bytes
        int imageSize = rowSize * height;
        
        header.fileSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + imageSize;
        infoHeader.width = width;
        infoHeader.height = height;
        infoHeader.imageSize = imageSize;
        
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
        
        // Write pixel data (BMP stores pixels bottom-to-top)
        std::vector<uint8_t> row(rowSize);
        for (int y = height - 1; y >= 0; --y) {
            const uint8_t* src = &pixels[y * width * 3];
            std::memcpy(row.data(), src, width * 3);
            file.write(reinterpret_cast<const char*>(row.data()), rowSize);
        }
        
        return true;
    }
    
    static bool saveVoxelGridSlice(const std::string& filename, const VoxelGrid& grid, int sliceZ) {
        Vec3 gridSize = grid.getGridSize();
        int width = static_cast<int>(gridSize.x);
        int height = static_cast<int>(gridSize.y);
        
        std::vector<uint8_t> pixels(width * height * 3, 0);
        
        // Render the slice
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec3 worldPos = grid.gridToWorld(Vec3(x, y, sliceZ));
                Vec4 color = grid.getVoxel(worldPos);
                
                int index = (y * width + x) * 3;
                color.toUint8(pixels[index + 2], pixels[index + 1], pixels[index]); // BMP is BGR
            }
        }
        
        return saveBMP(filename, pixels, width, height);
    }
    
    static bool saveRayTraceResults(const std::string& filename, const VoxelGrid& grid, 
                                   const std::vector<Vec3>& hitVoxels, 
                                   const AmanatidesWooAlgorithm::Ray& ray,
                                   int width = 800, int height = 600) {
        std::vector<uint8_t> pixels(width * height * 3, 0);
        
        // Background color (dark gray)
        for (int i = 0; i < width * height * 3; i += 3) {
            pixels[i] = 50;     // B
            pixels[i + 1] = 50; // G
            pixels[i + 2] = 50; // R
        }
        
        // Draw the grid bounds
        Vec3 gridSize = grid.getGridSize();
        drawGrid(pixels, width, height, gridSize);
        
        // Draw hit voxels
        for (const auto& voxel : hitVoxels) {
            drawVoxel(pixels, width, height, voxel, gridSize, Vec4(1, 0, 0, 1)); // Red for hit voxels
        }
        
        // Draw the ray
        drawRay(pixels, width, height, ray, gridSize);
        
        return saveBMP(filename, pixels, width, height);
    }
    
private:
    static void drawGrid(std::vector<uint8_t>& pixels, int width, int height, const Vec3& gridSize) {
        // Draw grid boundaries in white
        for (int x = 0; x <= gridSize.x; ++x) {
            int screenX = mapToScreenX(x, gridSize.x, width);
            for (int y = 0; y < height; ++y) {
                int index = (y * width + screenX) * 3;
                if (y % 5 == 0) { // Dotted line
                    pixels[index] = 255;     // B
                    pixels[index + 1] = 255; // G
                    pixels[index + 2] = 255; // R
                }
            }
        }
        
        for (int y = 0; y <= gridSize.y; ++y) {
            int screenY = mapToScreenY(y, gridSize.y, height);
            for (int x = 0; x < width; ++x) {
                int index = (screenY * width + x) * 3;
                if (x % 5 == 0) { // Dotted line
                    pixels[index] = 255;     // B
                    pixels[index + 1] = 255; // G
                    pixels[index + 2] = 255; // R
                }
            }
        }
    }
    
    static void drawVoxel(std::vector<uint8_t>& pixels, int width, int height, 
                         const Vec3& voxel, const Vec3& gridSize, const Vec4& color) {
        int screenX = mapToScreenX(voxel.x, gridSize.x, width);
        int screenY = mapToScreenY(voxel.y, gridSize.y, height);
        
        uint8_t r, g, b;
        color.toUint8(r, g, b);
        
        // Draw a 4x4 square for the voxel
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int px = screenX + dx;
                int py = screenY + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int index = (py * width + px) * 3;
                    pixels[index] = b;
                    pixels[index + 1] = g;
                    pixels[index + 2] = r;
                }
            }
        }
    }
    
    static void drawRay(std::vector<uint8_t>& pixels, int width, int height,
                       const AmanatidesWooAlgorithm::Ray& ray, const Vec3& gridSize) {
        // Draw ray origin and direction
        int originX = mapToScreenX(ray.origin.x, gridSize.x, width);
        int originY = mapToScreenY(ray.origin.y, gridSize.y, height);
        
        // Draw ray origin (green)
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                int px = originX + dx;
                int py = originY + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int index = (py * width + px) * 3;
                    pixels[index] = 0;       // B
                    pixels[index + 1] = 255; // G
                    pixels[index + 2] = 0;   // R
                }
            }
        }
        
        // Draw ray direction (yellow line)
        Vec3 endPoint = ray.origin + ray.direction * 10.0f; // Extend ray
        int endX = mapToScreenX(endPoint.x, gridSize.x, width);
        int endY = mapToScreenY(endPoint.y, gridSize.y, height);
        
        drawLine(pixels, width, height, originX, originY, endX, endY, Vec4(1, 1, 0, 1));
    }
    
    static void drawLine(std::vector<uint8_t>& pixels, int width, int height,
                        int x0, int y0, int x1, int y1, const Vec4& color) {
        uint8_t r, g, b;
        color.toUint8(r, g, b);
        
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        
        while (true) {
            if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                int index = (y0 * width + x0) * 3;
                pixels[index] = b;
                pixels[index + 1] = g;
                pixels[index + 2] = r;
            }
            
            if (x0 == x1 && y0 == y1) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
    
    static int mapToScreenX(float x, float gridWidth, int screenWidth) {
        return static_cast<int>((x / gridWidth) * screenWidth);
    }
    
    static int mapToScreenY(float y, float gridHeight, int screenHeight) {
        return static_cast<int>((y / gridHeight) * screenHeight);
    }
};

// Noise generation functions
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

    float x = xf - floorx;
    float y = yf - floory;
    float z = zf - floorz;

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

std::tuple<std::vector<Vec3>, std::vector<Vec4>> noiseBatch(int num_points, float scale, int sp[]) {
    TIME_FUNCTION;
    std::vector<Vec3> points;
    std::vector<Vec4> colors;
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

std::tuple<std::vector<Vec3>, std::vector<Vec4>> genPointCloud(int numP, float scale, int seed) {
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

// Function to populate voxel grid with point cloud data
void populateVoxelGridWithPointCloud(VoxelGrid& grid, 
                                   const std::vector<Vec3>& points, 
                                   const std::vector<Vec4>& colors) {
    TIME_FUNCTION;
    printf("Populating voxel grid with %zu points...\n", points.size());
    
    for (size_t i = 0; i < points.size(); i++) {
        grid.addVoxel(points[i], colors[i]);
    }
    
    printf("Voxel grid populated with %zu voxels\n", grid.getOccupiedPositions().size());
}

// Enhanced visualization function
void visualizePointCloud(const std::vector<Vec3>& points, const std::vector<Vec4>& colors, 
                        const std::string& filename, int width = 800, int height = 600) {
    TIME_FUNCTION;
    std::vector<uint8_t> pixels(width * height * 3, 0);
    
    // Background color (dark blue)
    for (int i = 0; i < width * height * 3; i += 3) {
        pixels[i] = 30;     // B
        pixels[i + 1] = 30; // G
        pixels[i + 2] = 50; // R
    }
    
    // Find bounds of point cloud
    Vec3 minPoint( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec3 maxPoint(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    
    for (const auto& point : points) {
        minPoint.x = std::min(minPoint.x, point.x);
        minPoint.y = std::min(minPoint.y, point.y);
        minPoint.z = std::min(minPoint.z, point.z);
        maxPoint.x = std::max(maxPoint.x, point.x);
        maxPoint.y = std::max(maxPoint.y, point.y);
        maxPoint.z = std::max(maxPoint.z, point.z);
    }
    
    Vec3 cloudSize = maxPoint - minPoint;
    float maxDim = std::max({cloudSize.x, cloudSize.y, cloudSize.z});
    
    // Draw points
    for (size_t i = 0; i < points.size(); i++) {
        const auto& point = points[i];
        const auto& color = colors[i];
        
        // Map 3D point to 2D screen coordinates (orthographic projection)
        int screenX = static_cast<int>(((point.x - minPoint.x) / maxDim) * (width - 20)) + 10;
        int screenY = static_cast<int>(((point.y - minPoint.y) / maxDim) * (height - 20)) + 10;
        
        if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
            uint8_t r, g, b;
            color.toUint8(r, g, b);
            
            // Draw a 2x2 pixel for each point
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = screenX + dx;
                    int py = screenY + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        int index = (py * width + px) * 3;
                        pixels[index] = b;
                        pixels[index + 1] = g;
                        pixels[index + 2] = r;
                    }
                }
            }
        }
    }
    
    BMPWriter::saveBMP(filename, pixels, width, height);
}

int main() {
    printf("=== Point Cloud Generation and Visualization ===\n\n");
    
    // Generate point cloud using noise function
    printf("Generating point cloud...\n");
    auto [points, colors] = genPointCloud(500000, 5.0f, 42);
    printf("Generated %zu points\n\n", points.size());
    
    // Create a voxel grid large enough to contain the point cloud
    Vec3 gridSize(25, 25, 25); // Adjusted to fit the scaled points
    Vec3 voxelSize(1, 1, 1);
    VoxelGrid grid(gridSize, voxelSize);
    
    // Populate voxel grid with point cloud
    populateVoxelGridWithPointCloud(grid, points, colors);
    
    // Visualize the point cloud
    printf("\nCreating visualizations...\n");
    visualizePointCloud(points, colors, "point_cloud_visualization.bmp");
    printf("Saved point cloud visualization to 'point_cloud_visualization.bmp'\n");
    
    // Save multiple slices of the voxel grid
    for (int z = 0; z < 5; z++) {
        std::string filename = "voxel_slice_z" + std::to_string(z) + ".bmp";
        if (BMPWriter::saveVoxelGridSlice(filename, grid, z)) {
            printf("Saved voxel grid slice to '%s'\n", filename.c_str());
        }
    }
    
    // Test ray tracing through the point cloud
    printf("\n=== Ray Tracing Test ===\n");
    AmanatidesWooAlgorithm::Ray ray(Vec3(-5, -5, -5), Vec3(1, 1, 1).normalized(), 50.0f);
    
    
    printf("\n=== Statistics ===\n");
    printf("Total points generated: %zu\n", points.size());
    printf("Voxels in grid: %zu\n", grid.getOccupiedPositions().size());
    printf("Grid size: (%.1f, %.1f, %.1f)\n", gridSize.x, gridSize.y, gridSize.z);
    printf("Voxel size: (%.1f, %.1f, %.1f)\n", voxelSize.x, voxelSize.y, voxelSize.z);

    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    return 0;
}