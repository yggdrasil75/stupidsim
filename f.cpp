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
#include "util/bmpwriter.hpp"

class VoxelGrid {
private:
    std::unordered_map<Vec3, size_t> positionToIndex;
    std::vector<Vec3> positions;
    std::vector<Vec4> colors;
    std::vector<int> layers; // New: stores layer information for each voxel
    
    Vec3 gridSize;

public:
    Vec3 voxelSize;
    
    // Layer types
    enum LayerType {
        ATMOSPHERE = 0,
        CRUST = 1,
        MANTLE = 2,
        OUTER_CORE = 3,
        INNER_CORE = 4,
        EMPTY = -1
    };
    
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
            layers.push_back(EMPTY); // Initialize with empty layer
            positionToIndex[gridPos] = index;
        } else {
            colors[it->second] = color;
        }
    }
    
    // Add a voxel with specified layer
    void addVoxelWithLayer(const Vec3& position, const Vec4& color, int layer) {
        Vec3 gridPos = worldToGrid(position);
        
        auto it = positionToIndex.find(gridPos);
        if (it == positionToIndex.end()) {
            size_t index = positions.size();
            positions.push_back(gridPos);
            colors.push_back(color);
            layers.push_back(layer);
            positionToIndex[gridPos] = index;
        } else {
            colors[it->second] = color;
            layers[it->second] = layer;
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
    
    // Get voxel layer at position
    int getVoxelLayer(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        auto it = positionToIndex.find(gridPos);
        if (it != positionToIndex.end()) {
            return layers[it->second];
        }
        return EMPTY;
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
    
    // Get all layers
    const std::vector<int>& getLayers() const {
        return layers;
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
        layers.clear();
        positionToIndex.clear();
    }
    
    // New method: Assign layers based on distance from center
    void assignPlanetaryLayers(const Vec3& center = Vec3(0, 0, 0)) {
        printf("Assigning planetary layers...\n");
        
        // Layer percentages (must sum to 1.0)
        const float atmospherePercent = 0.05f;  // Outer 5%
        const float crustPercent = 0.01f;       // Next 1%
        const float mantlePercent = 0.10f;      // Next 10%
        const float outerCorePercent = 0.42f;   // 50% of remaining 84%
        const float innerCorePercent = 0.42f;   // 50% of remaining 84%
        
        // Calculate maximum distance from center
        float maxDistance = 0.0f;
        for (const auto& pos : positions) {
            Vec3 worldPos = gridToWorld(pos);
            float distance = (worldPos - center).length();
            maxDistance = std::max(maxDistance, distance);
        }
        
        printf("Maximum distance from center: %.2f\n", maxDistance);
        
        // Calculate layer boundaries
        const float atmosphereStart = maxDistance * (1.0f - atmospherePercent);
        const float crustStart = maxDistance * (1.0f - atmospherePercent - crustPercent);
        const float mantleStart = maxDistance * (1.0f - atmospherePercent - crustPercent - mantlePercent);
        const float outerCoreStart = maxDistance * (1.0f - atmospherePercent - crustPercent - mantlePercent - outerCorePercent);
        
        printf("Layer boundaries:\n");
        printf("  Atmosphere: %.2f to %.2f\n", atmosphereStart, maxDistance);
        printf("  Crust: %.2f to %.2f\n", crustStart, atmosphereStart);
        printf("  Mantle: %.2f to %.2f\n", mantleStart, crustStart);
        printf("  Outer Core: %.2f to %.2f\n", outerCoreStart, mantleStart);
        printf("  Inner Core: 0.00 to %.2f\n", outerCoreStart);
        
        // Assign layers and colors based on distance
        int atmosphereCount = 0, crustCount = 0, mantleCount = 0, outerCoreCount = 0, innerCoreCount = 0;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            Vec3 worldPos = gridToWorld(positions[i]);
            float distance = (worldPos - center).length();
            
            // Define layer colors
            Vec4 layerColor;
            int layerType;
            
            if (distance >= atmosphereStart) {
                // Atmosphere - transparent blue
                layerColor = Vec4(0.2f, 0.4f, 1.0f, 0.3f); // Semi-transparent blue
                layerType = ATMOSPHERE;
                atmosphereCount++;
            } else if (distance >= crustStart) {
                // Crust - light brown
                layerColor = Vec4(0.8f, 0.7f, 0.5f, 1.0f); // Light brown
                layerType = CRUST;
                crustCount++;
            } else if (distance >= mantleStart) {
                // Mantle - reddish brown
                layerColor = Vec4(0.7f, 0.3f, 0.2f, 1.0f); // Reddish brown
                layerType = MANTLE;
                mantleCount++;
            } else if (distance >= outerCoreStart) {
                // Outer Core - orange/yellow
                layerColor = Vec4(1.0f, 0.6f, 0.2f, 1.0f); // Orange
                layerType = OUTER_CORE;
                outerCoreCount++;
            } else {
                // Inner Core - bright yellow
                layerColor = Vec4(1.0f, 0.9f, 0.1f, 1.0f); // Bright yellow
                layerType = INNER_CORE;
                innerCoreCount++;
            }
            
            colors[i] = layerColor;
            layers[i] = layerType;
        }
        
        printf("Layer distribution:\n");
        printf("  Atmosphere: %d voxels (%.1f%%)\n", atmosphereCount, (atmosphereCount * 100.0f) / positions.size());
        printf("  Crust: %d voxels (%.1f%%)\n", crustCount, (crustCount * 100.0f) / positions.size());
        printf("  Mantle: %d voxels (%.1f%%)\n", mantleCount, (mantleCount * 100.0f) / positions.size());
        printf("  Outer Core: %d voxels (%.1f%%)\n", outerCoreCount, (outerCoreCount * 100.0f) / positions.size());
        printf("  Inner Core: %d voxels (%.1f%%)\n", innerCoreCount, (innerCoreCount * 100.0f) / positions.size());
    }
};


// Generate sphere using Halton sequence
std::vector<Vec3> generateSphere(int numPoints, float radius = 1.0f) {
    
    printf("Generating sphere with %d points using grid method...\n", numPoints);
    
    std::vector<Vec3> points;
    
    // Calculate grid resolution based on desired number of points
    // For a sphere, we need to account that only about 52% of points in a cube will be inside the sphere
    int gridRes = static_cast<int>(std::cbrt(numPoints / 0.52f)) + 1;
    
    printf("Using grid resolution: %d x %d x %d\n", gridRes, gridRes, gridRes);
    
    // Generate points in a cube from -1 to 1 in all dimensions
    for (int x = 0; x < gridRes; ++x) {
        for (int y = 0; y < gridRes; ++y) {
            for (int z = 0; z < gridRes; ++z) {
                // Convert grid coordinates to normalized cube coordinates [-1, 1]
                Vec3 point(
                    (2.0f * x / (gridRes - 1)) - 1.0f,
                    (2.0f * y / (gridRes - 1)) - 1.0f,
                    (2.0f * z / (gridRes - 1)) - 1.0f
                );
                
                // Check if point is inside the unit sphere
                if (point.lengthSquared() <= 1.0f) {
                    points.push_back(point * radius); // Scale by radius
                }
            }
        }
    }
    
    printf("Generated %zu points inside sphere\n", points.size());
    
    // If we have too many points, randomly sample down to the desired number
    if (points.size() > static_cast<size_t>(numPoints)) {
        printf("Sampling down from %zu to %d points...\n", points.size(), numPoints);
        
        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Shuffle and resize
        std::shuffle(points.begin(), points.end(), gen);
        points.resize(numPoints);
    }
    // If we have too few points, we'll use what we have
    else if (points.size() < static_cast<size_t>(numPoints)) {
        printf("Warning: Only generated %zu points (requested %d)\n", points.size(), numPoints);
    }
    
    return points;
}


// Modified function to populate voxel grid with sphere points and assign layers
void populateVoxelGridWithLayeredSphere(VoxelGrid& grid, const std::vector<Vec3>& points) {
    printf("Populating voxel grid with %zu sphere points...\n", points.size());
    
    // First add all voxels with a default color
    Vec4 defaultColor(1.0f, 1.0f, 1.0f, 1.0f); // Temporary color
    for (const auto& point : points) {
        grid.addVoxel(point, defaultColor);
    }
    
    printf("Voxel grid populated with %zu voxels\n", grid.getOccupiedPositions().size());
    
    // Now assign the planetary layers
    grid.assignPlanetaryLayers();
}

// void visualizeSphere(const std::vector<Vec3>& points, const std::string& filename, 
//                     const Vec4& color = Vec4(0, 0, 1, 1),
//                     int width = 800, int height = 600) {
    
//     // Create a 2D vector for the image
//     std::vector<std::vector<Vec3>> image(height, std::vector<Vec3>(width, Vec3(0.2f, 0.2f, 0.2f))); // Dark gray background
    
//     if (points.empty()) {
//         BMPWriter::saveBMP(filename, image);
//         return;
//     }
    
//     // Find bounds of the projected points
//     float minX = points[0].x, maxX = points[0].x;
//     float minY = points[0].y, maxY = points[0].y;
    
//     for (const auto& point : points) {
//         minX = std::min(minX, point.x);
//         maxX = std::max(maxX, point.x);
//         minY = std::min(minY, point.y);
//         maxY = std::max(maxY, point.y);
//     }
    
//     // Calculate scale and offset to fit points in image
//     float scaleX = (width - 40) / (maxX - minX);
//     float scaleY = (height - 40) / (maxY - minY);
//     float scale = std::min(scaleX, scaleY);
    
//     float offsetX = 20 - minX * scale;
//     float offsetY = 20 - minY * scale;
    
//     // Convert Vec4 color to Vec3 (RGB)
//     Vec3 rgbColor(color.r, color.g, color.b);
    
//     // Draw points
//     for (const auto& point : points) {
//         int screenX = static_cast<int>(point.x * scale + offsetX);
//         int screenY = static_cast<int>(point.y * scale + offsetY);
        
//         if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
//             // Draw a 3x3 square for each point
//             for (int dy = -1; dy <= 1; dy++) {
//                 for (int dx = -1; dx <= 1; dx++) {
//                     int px = screenX + dx;
//                     int py = screenY + dy;
//                     if (px >= 0 && px < width && py >= 0 && py < height) {
//                         image[py][px] = rgbColor;
//                     }
//                 }
//             }
//         }
//     }
    
//     // Save using the independent BMPWriter
//     BMPWriter::saveBMP(filename, image);
//     printf("Saved sphere visualization to '%s'\n", filename.c_str());
// }


void visualizePointCloud(const std::vector<Vec3>& points, const std::vector<Vec4>& colors, 
                        const std::string& filename, int width = 1000, int height = 1000) {
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
    printf("=== Layered Sphere Generation and Visualization ===\n\n");
    
    const int numPoints = 10000000;
    const float radius = 2.0f;
    
    printf("Generating layered spheres with %d points each, radius %.1f\n\n", numPoints, radius);
    
    // Create voxel grid
    VoxelGrid grid(Vec3(10, 10, 10), Vec3(0.1f, 0.1f, 0.1f));
    
    // Generate and visualize Fibonacci sphere with layers
    printf("1. Generating sphere...\n");
    auto fibSphere = generateSphere(numPoints, radius);
    populateVoxelGridWithLayeredSphere(grid, fibSphere);
    
    // Extract positions and colors for visualization
    std::vector<Vec3> occupiedPositions = grid.getOccupiedPositions();
    std::vector<Vec4> layerColors = grid.getColors();
    
    // Convert to world coordinates for visualization
    std::vector<Vec3> worldPositions;
    for (const auto& gridPos : occupiedPositions) {
        worldPositions.push_back(grid.gridToWorld(gridPos));
    }
    
    // Create a simple visualization using the layer colors
    visualizePointCloud(grid.getOccupiedPositions(), grid.getColors(), "output/sphere.bmp",  800, 600);
    
    printf("=== sphere generated successfully ===\n");
    printf("Files created:\n");
    printf("  - sphere.bmp\n");
    
    return 0;
}