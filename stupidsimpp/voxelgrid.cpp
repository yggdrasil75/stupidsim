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
#include "classes.cpp"

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
    
};