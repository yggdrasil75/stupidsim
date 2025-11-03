#include <vector>
#include <cstdio>
#include "voxel_grid.hpp"
#include "ray_tracer.hpp"
#include "bmp_writer.hpp"
#include "noise_generator.hpp"
#include "timing_decorator.hpp"

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

int main() {
    printf("=== Point Cloud Generation and Visualization ===\n\n");
    
    // Generate point cloud using noise function
    printf("Generating point cloud...\n");
    auto [points, colors] = NoiseGenerator::genPointCloud(500000, 5.0f, 42);
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