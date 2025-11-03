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
#include "timing_decorator.hpp"
#include <cstdint>
#include <string>


// Forward declarations
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

// Example usage
int main() {
    // Create a voxel grid
    VoxelGrid grid(Vec3(10, 10, 10), Vec3(1, 1, 1));
    
    // Add some voxels
    grid.addVoxel(Vec3(1, 1, 1), Vec4(1, 0, 0, 1)); // Red
    grid.addVoxel(Vec3(2, 2, 2), Vec4(0, 1, 0, 0.5f)); // Green with transparency
    grid.addVoxel(Vec3(3, 3, 3), Vec4(0, 0, 1, 1)); // Blue
    grid.addVoxel(Vec3(4, 4, 4), Vec4(1, 1, 0, 1)); // Yellow
    grid.addVoxel(Vec3(5, 5, 5), Vec4(1, 0, 1, 1)); // Magenta
    
    // Create a ray
    AmanatidesWooAlgorithm::Ray ray(Vec3(0, 0, 0), Vec3(1, 1, 1).normalized());
    
    // Traverse the grid
    std::vector<Vec3> hitVoxels;
    std::vector<float> hitDistances;
    bool hit = AmanatidesWooAlgorithm::traverse(ray, grid, hitVoxels, hitDistances);
    
    if (hit) {
        printf("Ray hit %zu voxels:\n", hitVoxels.size());
        for (size_t i = 0; i < hitVoxels.size(); i++) {
            Vec4 color = grid.getVoxel(grid.gridToWorld(hitVoxels[i]));
            printf("  Voxel at (%.1f, %.1f, %.1f), distance: %.2f, color: (%.1f, %.1f, %.1f, %.1f)\n",
                   hitVoxels[i].x, hitVoxels[i].y, hitVoxels[i].z,
                   hitDistances[i],
                   color.r, color.g, color.b, color.a);
        }
    }
    
    // Save results to BMP files
    printf("\nSaving results to BMP files...\n");
    
    // Save a slice of the voxel grid
    if (BMPWriter::saveVoxelGridSlice("voxel_grid_slice.bmp", grid, 1)) {
        printf("Saved voxel grid slice to 'voxel_grid_slice.bmp'\n");
    } else {
        printf("Failed to save voxel grid slice\n");
    }
    
    // Save ray tracing visualization
    if (BMPWriter::saveRayTraceResults("ray_trace_results.bmp", grid, hitVoxels, ray)) {
        printf("Saved ray trace results to 'ray_trace_results.bmp'\n");
    } else {
        printf("Failed to save ray trace results\n");
    }
    
    return 0;
}