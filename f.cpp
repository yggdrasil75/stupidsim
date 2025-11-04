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
#include "util/vec3.hpp"
#include "util/vec4.hpp"

class VoxelGrid {
private:
    std::unordered_map<Vec3, size_t> positionToIndex;
    std::vector<Vec3> positions;
    std::vector<Vec4> colors;
    
    Vec3 gridSize;

public:
    Vec3 voxelSize;
    VoxelGrid(const Vec3& size, const Vec3& voxelSize = Vec3(1, 1, 1)) : gridSize(size), voxelSize(voxelSize) {}
    
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
    
    static bool saveVoxelGridProjection(const std::string& filename, const VoxelGrid& grid, 
                                       int width = 800, int height = 600) {
        std::vector<uint8_t> pixels(width * height * 3, 0);
        
        // Background color (dark gray)
        for (int i = 0; i < width * height * 3; i += 3) {
            pixels[i] = 50;     // B
            pixels[i + 1] = 50; // G
            pixels[i + 2] = 50; // R
        }
        
        // Get all occupied positions
        const auto& positions = grid.getOccupiedPositions();
        const auto& colors = grid.getColors();
        
        if (positions.empty()) {
            return saveBMP(filename, pixels, width, height);
        }
        
        // Find bounds
        Vec3 minPos = positions[0];
        Vec3 maxPos = positions[0];
        for (const auto& pos : positions) {
            minPos.x = std::min(minPos.x, pos.x);
            minPos.y = std::min(minPos.y, pos.y);
            minPos.z = std::min(minPos.z, pos.z);
            maxPos.x = std::max(maxPos.x, pos.x);
            maxPos.y = std::max(maxPos.y, pos.y);
            maxPos.z = std::max(maxPos.z, pos.z);
        }
        
        Vec3 size = maxPos - minPos;
        float scale = std::min((width - 40) / size.x, (height - 40) / size.y);
        
        // Draw voxels
        for (size_t i = 0; i < positions.size(); i++) {
            const auto& pos = positions[i];
            const auto& color = colors[i];
            
            // Project to 2D (orthographic - ignoring Z for now)
            int screenX = static_cast<int>((pos.x - minPos.x) * scale) + 20;
            int screenY = static_cast<int>((pos.y - minPos.y) * scale) + 20;
            
            if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
                uint8_t r, g, b;
                color.toUint8(r, g, b);
                
                // Draw a 3x3 square for each voxel
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
        
        return saveBMP(filename, pixels, width, height);
    }
};

// Sobol sequence generator
class SobolSequence {
private:
    uint32_t n;
    
public:
    SobolSequence() : n(0) {}
    
    void reset() {
        n = 0;
    }
    
    double next() {
        uint32_t value = n;
        n++;
        
        // Gray code and convert to [0,1)
        value ^= value >> 1;
        value ^= value >> 2;
        value ^= value >> 4;
        value ^= value >> 8;
        value ^= value >> 16;
        
        return static_cast<double>(value) / static_cast<double>(0xFFFFFFFF);
    }
    
    Vec3 nextVec3() {
        return Vec3(next(), next(), next());
    }
};

// Halton sequence generator
class HaltonSequence {
private:
    int base2, base3, base5;
    int index;
    
    double halton(int base) {
        double result = 0.0;
        double f = 1.0 / base;
        int i = index;
        
        while (i > 0) {
            result += f * (i % base);
            i = i / base;
            f /= base;
        }
        
        return result;
    }
    
public:
    HaltonSequence() : base2(2), base3(3), base5(5), index(1) {}
    
    void reset() {
        index = 1;
    }
    
    Vec3 next() {
        Vec3 result(halton(base2), halton(base3), halton(base5));
        index++;
        return result;
    }
};

// Generate sphere points using Fibonacci spiral
std::vector<Vec3> generateFibonacciSphere(int numPoints, float radius = 1.0f) {
    std::vector<Vec3> points;
    points.reserve(numPoints);
    
    const float goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    
    for (int i = 0; i < numPoints; ++i) {
        float y = 1.0f - (2.0f * i) / (numPoints - 1.0f);
        float radiusAtY = std::sqrt(1.0f - y * y);
        
        float theta = 2.0f * M_PI * i / goldenRatio;
        
        float x = std::cos(theta) * radiusAtY;
        float z = std::sin(theta) * radiusAtY;
        
        points.emplace_back(x * radius, y * radius, z * radius);
    }
    
    return points;
}

// Generate sphere using Sobol sequence
std::vector<Vec3> generateSobolSphere(int numPoints, float radius = 1.0f) {
    std::vector<Vec3> points;
    points.reserve(numPoints);
    
    SobolSequence sobol;
    
    for (int i = 0; i < numPoints; ++i) {
        Vec3 rand = sobol.nextVec3();
        
        // Convert to spherical coordinates
        float theta = 2.0f * M_PI * rand.x;  // azimuthal angle [0, 2π]
        float phi = std::acos(1.0f - 2.0f * rand.y);  // polar angle [0, π]
        
        // Convert to Cartesian coordinates
        float x = radius * std::sin(phi) * std::cos(theta);
        float y = radius * std::sin(phi) * std::sin(theta);
        float z = radius * std::cos(phi);
        
        points.emplace_back(x, y, z);
    }
    
    return points;
}

// Generate sphere using Halton sequence
std::vector<Vec3> generateHaltonSphere(int numPoints, float radius = 1.0f) {
    std::vector<Vec3> points;
    points.reserve(numPoints);
    
    HaltonSequence halton;
    
    for (int i = 0; i < numPoints; ++i) {
        Vec3 rand = halton.next();
        
        // Convert to spherical coordinates
        float theta = 2.0f * M_PI * rand.x;  // azimuthal angle
        float phi = std::acos(1.0f - 2.0f * rand.y);  // polar angle
        
        // Convert to Cartesian coordinates
        float x = radius * std::sin(phi) * std::cos(theta);
        float y = radius * std::sin(phi) * std::sin(theta);
        float z = radius * std::cos(phi);
        
        points.emplace_back(x, y, z);
    }
    
    return points;
}

// Generate sphere using normalized random points
std::vector<Vec3> generateNormalizedSphere(int numPoints, float radius = 1.0f) {
    std::vector<Vec3> points;
    points.reserve(numPoints);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < numPoints; ++i) {
        // Generate random point with normal distribution
        Vec3 point(dist(gen), dist(gen), dist(gen));
        
        // Normalize and scale to sphere surface
        float len = point.length();
        if (len > 0.001f) {
            point = point * (radius / len);
            points.push_back(point);
        }
    }
    
    return points;
}

// Function to populate voxel grid with sphere points
void populateVoxelGridWithSphere(VoxelGrid& grid, const std::vector<Vec3>& points, const Vec4& color) {
    printf("Populating voxel grid with %zu sphere points...\n", points.size());
    
    for (const auto& point : points) {
        grid.addVoxel(point, color);
    }
    
    printf("Voxel grid populated with %zu voxels\n", grid.getOccupiedPositions().size());
}

// Function to visualize sphere in voxel grid
void visualizeSphere(const std::vector<Vec3>& points, const std::string& filename, 
                    const Vec4& color = Vec4(0, 0, 1, 1),  // Default blue
                    int width = 800, int height = 600) {
    // Create a voxel grid large enough to contain the sphere
    Vec3 gridSize(10, 10, 10); // Adjusted for sphere radius of 2.0
    Vec3 voxelSize(0.1f, 0.1f, 0.1f); // Smaller voxels for better sphere representation
    VoxelGrid grid(gridSize, voxelSize);
    
    // Populate with sphere points
    populateVoxelGridWithSphere(grid, points, color);
    
    // Save projection
    BMPWriter::saveVoxelGridProjection(filename, grid, width, height);
    printf("Saved sphere visualization to '%s'\n", filename.c_str());
    
    // Also save a few slices
    for (int z = 4; z <= 6; z++) {
        std::string sliceFilename = filename.substr(0, filename.length() - 4) + 
                                   "_slice_z" + std::to_string(z) + ".bmp";
        BMPWriter::saveVoxelGridSlice(sliceFilename, grid, z);
        printf("Saved slice to '%s'\n", sliceFilename.c_str());
    }
}

int main() {
    printf("=== Sphere Generation and Visualization ===\n\n");
    
    const int numPoints = 1000;
    const float radius = 2.0f;
    const Vec4 blueColor(0.2f, 0.4f, 1.0f, 1.0f); // Nice blue color
    
    printf("Generating spheres with %d points each, radius %.1f\n\n", numPoints, radius);
    
    // Generate and visualize Fibonacci sphere
    printf("1. Generating Fibonacci sphere...\n");
    auto fibSphere = generateFibonacciSphere(numPoints, radius);
    visualizeSphere(fibSphere, "output/fibonacci_sphere.bmp", blueColor);
    printf("   Generated %zu points\n\n", fibSphere.size());
    
    // Generate and visualize Sobol sphere
    printf("2. Generating Sobol sphere...\n");
    auto sobolSphere = generateSobolSphere(numPoints, radius);
    visualizeSphere(sobolSphere, "output/sobol_sphere.bmp", blueColor);
    printf("   Generated %zu points\n\n", sobolSphere.size());
    
    // Generate and visualize Halton sphere
    printf("3. Generating Halton sphere...\n");
    auto haltonSphere = generateHaltonSphere(numPoints, radius);
    visualizeSphere(haltonSphere, "output/halton_sphere.bmp", blueColor);
    printf("   Generated %zu points\n\n", haltonSphere.size());
    
    // Generate and visualize normalized sphere
    printf("4. Generating normalized sphere...\n");
    auto normSphere = generateNormalizedSphere(numPoints, radius);
    visualizeSphere(normSphere, "output/normalized_sphere.bmp", blueColor);
    printf("   Generated %zu points\n\n", normSphere.size());
    
    printf("=== All spheres generated successfully ===\n");
    printf("Files created:\n");
    printf("  - fibonacci_sphere.bmp (and slices)\n");
    printf("  - sobol_sphere.bmp (and slices)\n");
    printf("  - halton_sphere.bmp (and slices)\n");
    printf("  - normalized_sphere.bmp (and slices)\n");
    
    return 0;
}