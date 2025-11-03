#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include "timing_decorator.hpp"
#include <cstdint>
#include <fstream>
#include <string>

const float EPSILON = 0.00000000001;

//classes and structs

struct Bool3 {
    bool x, y, z;
};

class Vec3 {
public:
    double x, y, z;
    
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    //Vec3(const VoxelIndex& idx) : x(static_cast<double>(idx.x)), y(static_cast<double>(idx.y)), z(static_cast<double>(idx.z)) {}
    
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

    Vec3 operator+(const Bool3& other) const {
        return {
            x + (other.x ? 1.0f : 0.0f),
            y + (other.y ? 1.0f : 0.0f),
            z + (other.z ? 1.0f : 0.0f)
        };
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

    const float& operator[](int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        if (index == 2) return z;
        throw std::out_of_range("Index out of range");
    }

    double& operator[](int index) {
        if (index == 0) return x;
        if (index == 1) return y;
        if (index == 2) return z;
        throw std::out_of_range("Index out of range");
    }
    
    struct Hash {
        size_t operator()(const Vec3& v) const {
            size_t h1 = std::hash<double>()(std::round(v.x * 1000.0));
            size_t h2 = std::hash<double>()(std::round(v.y * 1000.0));
            size_t h3 = std::hash<double>()(std::round(v.z * 1000.0));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    std::string toString() const {
        return "Vec3(" + std::to_string(x) + ", " + 
                             std::to_string(y) + ", " + 
                             std::to_string(z) + ")";
    }

    Vec3& safe_inverse_dir(float epsilon = 1e-6f) {
        x = (std::abs(x) > epsilon) ? x : std::copysign(epsilon, -x);
        y = (std::abs(y) > epsilon) ? y : std::copysign(epsilon, -y);
        z = (std::abs(z) > epsilon) ? z : std::copysign(epsilon, -z);
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

struct VoxelIndex {
    int x, y, z;

    // Constructor
    VoxelIndex(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}

    // Array-like access
    int& operator[](size_t index) {
        switch(index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("Index out of range");
        }
    }

    const int& operator[](size_t index) const {
        switch(index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("Index out of range");
        }
    }

    inline VoxelIndex operator+(const Vec3& other) const {
        return VoxelIndex(std::floor(x + other.x), std::floor(y + other.y), std::floor(z + other.z));
    }

    inline VoxelIndex operator-(const Vec3& other) const {
        return VoxelIndex(std::floor(x - other.x), std::floor(y - other.y), std::floor(z - other.z));
    }

    inline VoxelIndex operator+(float scalar) const {
        return VoxelIndex(std::floor(x + scalar), std::floor(y + scalar), std::floor(z + scalar));
    }
    
    VoxelIndex operator+(const Bool3& other) const {
        return {
            x + (other.x ? 1 : 0),
            y + (other.y ? 1 : 0),
            z + (other.z ? 1 : 0)
        };
    }

    inline VoxelIndex operator*(const Vec3& other) const {
        return VoxelIndex(std::floor(x * other.x), std::floor(y * other.y), std::floor(z * other.z));
    }
    inline VoxelIndex operator*(float scalar) const {
        return VoxelIndex(std::floor(x * scalar), std::floor(y * scalar), std::floor(z * scalar));
    }

    operator Vec3() const {
        return Vec3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
    }

    Vec3 toVec3() const {
        return Vec3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
    }

    // Hash function
    size_t hash() const {
        return std::hash<int>{}(x) ^ 
               (std::hash<int>{}(y) << 1) ^ 
               (std::hash<int>{}(z) << 2);
    }

    // Convert to string
    std::string toString() const {
        return "VoxelIndex(" + std::to_string(x) + ", " + 
                             std::to_string(y) + ", " + 
                             std::to_string(z) + ")";
    }

    // Comparison operators for completeness
    bool operator==(const VoxelIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const VoxelIndex& other) const {
        return !(*this == other);
    }
};

namespace std {
    template<>
    struct hash<Vec3> {
        size_t operator()(const Vec3& p) const {
            return hash<float>()(p.x) ^ hash<float>()(p.y) ^ hash<float>()(p.z);
        }
    };
    template<>
    struct hash<VoxelIndex> {
        size_t operator()(const VoxelIndex& idx) const {
            return idx.hash();
        }
    };
}

class VoxelGrid {
public:
    // New structure - using position-based indexing
    std::unordered_map<Vec3, size_t> pos_index_map;  // Maps voxel center -> index in positions/colors
    std::vector<Vec3> positions;                     // Voxel center positions
    std::vector<Vec4> colors;                        // Average colors per voxel
    float gridsize;                                  // Voxel size
    
    // Old structure - to be removed/replaced
    float voxel_size;
    Vec3 min_bounds;
    Vec3 max_bounds;
        
    // Grid dimensions
    int dim_x, dim_y, dim_z;
    
    // Grid arrays for fast access (like Python)
    std::vector<bool> grid_array;
    std::vector<Vec4> color_array;
    std::vector<int> count_array;
    
    bool arrays_initialized = false;

    void initializeArrays() {
        if (arrays_initialized) return;
        
        size_t total_size = dim_x * dim_y * dim_z;
        grid_array.resize(total_size, false);
        color_array.resize(total_size, Vec4{0, 0, 0, 0});
        count_array.resize(total_size, 0);
        
        // Populate arrays from the new data structure
        for (const auto& [voxel_pos, index] : pos_index_map) {
            VoxelIndex voxel_idx = getVoxelIndex(voxel_pos);
            if (isValidIndex(voxel_idx)) {
                size_t array_idx = getArrayIndex(voxel_idx);
                grid_array[array_idx] = true;
                color_array[array_idx] = colors[index];
                count_array[array_idx] = 1; // Each entry represents one voxel
            }
        }
        
        arrays_initialized = true;
    }

    size_t getArrayIndex(const VoxelIndex& index) const {
        return index[0] + dim_x * (index[1] + dim_y * index[2]);
    }

    bool isValidIndex(const VoxelIndex& idx) const {
        return idx[0] >= 0 && idx[0] < dim_x &&
               idx[1] >= 0 && idx[1] < dim_y &&
               idx[2] >= 0 && idx[2] < dim_z;
    }

    VoxelGrid(float size = 0.1f) : gridsize(size), voxel_size(size), arrays_initialized(false) {}
    
    void addPoints(const std::vector<Vec3>& points, const std::vector<Vec4>& input_colors) {
        if (points.size() != input_colors.size()) {
            std::cerr << "Error: Points and colors vectors must have the same size" << std::endl;
            return;
        }
        
        if (points.empty()) return;
        
        // Calculate bounds
        min_bounds = points[0];
        max_bounds = points[0];
        
        for (const auto& point : points) {
            min_bounds.x = std::min(min_bounds.x, point.x);
            min_bounds.y = std::min(min_bounds.y, point.y);
            min_bounds.z = std::min(min_bounds.z, point.z);
            
            max_bounds.x = std::max(max_bounds.x, point.x);
            max_bounds.y = std::max(max_bounds.y, point.y);
            max_bounds.z = std::max(max_bounds.z, point.z);
        }
        
        // Calculate grid dimensions
        Vec3 range = max_bounds - min_bounds;
        dim_x = static_cast<int>(std::ceil(range.x / gridsize)) + 1;
        dim_y = static_cast<int>(std::ceil(range.y / gridsize)) + 1;
        dim_z = static_cast<int>(std::ceil(range.z / gridsize)) + 1;
        
        // Temporary storage for averaging colors per voxel
        std::unordered_map<Vec3, std::vector<Vec4>, Vec3::Hash> voxel_color_map;
        
        // Group points by voxel and collect colors
        for (size_t i = 0; i < points.size(); ++i) {
            Vec3 voxel_center = getVoxelCenter(getVoxelIndex(points[i]));
            voxel_color_map[voxel_center].push_back(input_colors[i]);
        }
        
        // Convert to the new structure - store average colors per voxel
        positions.clear();
        colors.clear();
        pos_index_map.clear();
        
        for (auto& [voxel_pos, color_list] : voxel_color_map) {
            // Calculate average color
            Vec4 avg_color{0, 0, 0, 0};
            for (const auto& color : color_list) {
                avg_color.w += color.w;
                avg_color.x += color.x;
                avg_color.y += color.y;
                avg_color.z += color.z;
            }
            avg_color.w /= color_list.size();
            avg_color.x /= color_list.size();
            avg_color.y /= color_list.size();
            avg_color.z /= color_list.size();
            
            // Store in new structure
            size_t index = positions.size();
            positions.push_back(voxel_pos);
            colors.push_back(avg_color);
            pos_index_map[voxel_pos] = index;
        }
        
        // Initialize arrays for fast access
        initializeArrays();
    }
    
    // Get voxel data using voxel center position
    bool hasVoxelAt(const Vec3& voxel_center) const {
        return pos_index_map.find(voxel_center) != pos_index_map.end();
    }
    
    Vec4 getVoxelColor(const Vec3& voxel_center) const {
        auto it = pos_index_map.find(voxel_center);
        if (it != pos_index_map.end()) {
            return colors[it->second];
        }
        return Vec4{0, 0, 0, 0};
    }
    
    // Fast array access using VoxelIndex (compatibility with existing code)
    bool getVoxelOccupied(const VoxelIndex& voxel_idx) const {
        if (!arrays_initialized || !isValidIndex(voxel_idx)) return false;
        return grid_array[getArrayIndex(voxel_idx)];
    }
    
    Vec4 getVoxelColor(const VoxelIndex& voxel_idx) const {
        if (!arrays_initialized || !isValidIndex(voxel_idx)) 
            return Vec4{0, 0, 0, 0};
        return color_array[getArrayIndex(voxel_idx)];
    }
    
    int getVoxelPointCount(const VoxelIndex& voxel_idx) const {
        if (!arrays_initialized || !isValidIndex(voxel_idx)) return 0;
        return count_array[getArrayIndex(voxel_idx)];
    }
    
    std::vector<VoxelIndex> getVoxelIndices() const {
        std::vector<VoxelIndex> indices;
        for (const auto& voxel_pos : positions) {
            indices.push_back(getVoxelIndex(voxel_pos));
        }
        return indices;
    }
    
    std::vector<Vec3> getVoxelCenters() const {
        return positions;
    }
    
    size_t getNumVoxels() const { return positions.size(); }
    size_t getNumPoints() const {
        // Since we're storing averaged voxels, each position represents multiple original points
        // For simplicity, return number of voxels
        return positions.size();
    }
    
    Vec3 getMinBounds() const { return min_bounds; }
    Vec3 getMaxBounds() const { return max_bounds; }
    std::array<int, 3> getDimensions() const { return {dim_x, dim_y, dim_z}; }
    float getVoxelSize() const { return gridsize; }
    
    VoxelIndex getVoxelIndex(const Vec3& point) const {
        Vec3 normalized = point - min_bounds;
        return VoxelIndex{
            static_cast<int>(std::floor(normalized.x / gridsize)),
            static_cast<int>(std::floor(normalized.y / gridsize)),
            static_cast<int>(std::floor(normalized.z / gridsize))
        };
    }

    Vec3 getVoxelCenter(const VoxelIndex& voxel_idx) const {
        return Vec3(
            min_bounds.x + (voxel_idx[0] + 0.5f) * gridsize,
            min_bounds.y + (voxel_idx[1] + 0.5f) * gridsize,
            min_bounds.z + (voxel_idx[2] + 0.5f) * gridsize
        );
    }

    void clear() {
        pos_index_map.clear();
        positions.clear();
        colors.clear();
        grid_array.clear();
        color_array.clear();
        count_array.clear();
        arrays_initialized = false;
    }
    
    void setVoxelSize(float size) {
        gridsize = size;
        voxel_size = size;
        clear();
    }
    
    void printStats() const {
        std::cout << "Voxel Grid Statistics:" << std::endl;
        std::cout << "  Voxel size: " << gridsize << std::endl;
        std::cout << "  Grid dimensions: " << dim_x << " x " << dim_y << " x " << dim_z << std::endl;
        std::cout << "  Number of voxels: " << getNumVoxels() << std::endl;
        std::cout << "  Using new position-based storage" << std::endl;
    }
};

struct Image {
    int width;
    int height;
    std::vector<uint8_t> data; // RGBA format
    
    Image(int w, int h) : width(w), height(h), data(w * h * 4) {
        // Initialize to white background
        for (int i = 0; i < w * h * 4; i += 4) {
            data[i] = 255;     // R
            data[i + 1] = 255; // G  
            data[i + 2] = 255; // B
            data[i + 3] = 255; // A
        }
    }    

    // Helper methods
    uint8_t* pixel(int x, int y) {
        return &data[(y * width + x) * 4];
    }
    
    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        uint8_t* p = pixel(x, y);
        p[0] = r; p[1] = g; p[2] = b; p[3] = a;
    }

    bool saveAsBMP(const std::string& filename) {
        #pragma pack(push, 1)

        // BMP file header (14 bytes)
        struct BMPFileHeader {
            uint16_t file_type{0x4D42}; // "BM"
            uint32_t file_size{0};
            uint16_t reserved1{0};
            uint16_t reserved2{0};
            uint32_t offset_data{0};
        } file_header;
        
        // BMP info header (40 bytes)
        struct BMPInfoHeader {
            uint32_t size{40};
            int32_t width{0};
            int32_t height{0};
            uint16_t planes{1};
            uint16_t bit_count{32};
            uint32_t compression{0}; // BI_RGB
            uint32_t size_image{0};
            int32_t x_pixels_per_meter{0};
            int32_t y_pixels_per_meter{0};
            uint32_t colors_used{0};
            uint32_t colors_important{0};
        } info_header;
        
        // Calculate sizes
        uint32_t row_stride = width * 4;
        uint32_t padding_size = (4 - (row_stride % 4)) % 4;
        uint32_t data_size = (row_stride + padding_size) * height;
        
        file_header.file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + data_size;
        file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        
        info_header.width = width;
        info_header.height = -height; // Negative for top-down bitmap (no flipping needed)
        info_header.size_image = data_size;
        
        // Open file
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // Write headers
        file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
        file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
        
        // Write pixel data (BMP stores as BGR, we have RGB)
        std::vector<uint8_t> row_buffer(row_stride + padding_size);
        
        for (int y = 0; y < height; ++y) {
            uint8_t* row_start = &data[y * row_stride];
            
            // Convert RGB to BGR and copy to buffer
            for (int x = 0; x < width; ++x) {
                uint8_t* src_pixel = row_start + (x * 4);
                uint8_t* dst_pixel = row_buffer.data() + (x * 4);
                
                // Swap R and B channels (RGBA -> BGRA)
                dst_pixel[0] = src_pixel[2]; // B
                dst_pixel[1] = src_pixel[1]; // G
                dst_pixel[2] = src_pixel[0]; // R
                //dst_pixel[3] = src_pixel[3]; // A
            }
            
            // Write row with padding
            file.write(reinterpret_cast<const char*>(row_buffer.data()), row_stride + padding_size);
        }
        
        return file.good();
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


//voxel stuff
Bool3 greaterThanZero(const Vec3& v) {
    return {v.x > 0, v.y > 0, v.z > 0};
}

Image render(int height, int width, Vec3 forward, Vec3 right, Vec3 up, Vec3 rayOrigin, Vec3 vbound,
            float vsize, VoxelGrid grid, int dims) {
    TIME_FUNCTION;

    Image img = Image(height, width);
    
    float max_t = 50.0f;
    int max_steps = 123;
    float max_dist = 25.0f;
    float screen_height = height;
    float screen_width = width;
    
    float inv_w = 1.0f / width;
    float inv_h = 1.0f / height;
    float scr_w_half = screen_width * 0.5f;
    float scr_h_half = screen_height * 0.5f;
    
    std::cout << height << std::endl;
    std::cout << width << std::endl;
    for (int y = 0; y < height; y++) {
        int sy = std::ceil(1.0 - ((2.0 * y) * inv_h) * scr_h_half);
        for (int x = 0; x < width; x++) {
            int sx = std::ceil((((2.0 * x) * inv_w) - 1.0) * scr_w_half);
            std::cout << "working in: " << x << ", " << y << std::endl;
            Vec3 ray_dir = (forward + (sx * right) + (sy * up)).normalize();
            std::cout << "current ray direction: " << ray_dir.toString() << std::endl;
            Vec3 cv1 = ((rayOrigin - vbound) / vsize);
            VoxelIndex cv = VoxelIndex(static_cast<int>(cv1.x), static_cast<int>(cv1.y), static_cast<int>(cv1.z));
            std::cout << "cell at: " << cv.toString() << std::endl;
            Vec3 inv_dir =  Vec3(
                ray_dir.x != 0 ? 1.0f / ray_dir.x : std::numeric_limits<float>::max(),
                ray_dir.y != 0 ? 1.0f / ray_dir.y : std::numeric_limits<float>::max(),
                ray_dir.z != 0 ? 1.0f / ray_dir.z : std::numeric_limits<float>::max()
            );
            std::cout << "inverse of the current ray: " << inv_dir.toString() << std::endl;
            Vec3 step = ray_dir.sign();
            std::cout << "current ray signs: " << step.toString() << std::endl;
            Bool3 step_mask = greaterThanZero(step);
            VoxelIndex next_voxel_bound = (cv + step_mask) * vsize + vbound;
            std::cout << "next cell at: " << next_voxel_bound.toString() << std::endl;
            Vec3 t_max = (Vec3(next_voxel_bound) - rayOrigin) * inv_dir;
            std::cout << "t_max at: " << t_max.toString() << std::endl;
            Vec3 t_delta = vsize / inv_dir.abs();
            std::cout << "t_delta: " << t_delta.toString() << std::endl;
            float t = 0.0f;

            Vec4 accumulatedColor = Vec4(0,0,0,0);
            //std::cout << x << "," << y << std::endl;
            for (int iter = 0; iter < max_steps; iter++) {
                if (max_t < t) {
                    std::cout << "t failed " << std::endl;
                    break;
                }
                if (accumulatedColor.z >= 1.0) {
                    std::cout << "z failed " << std::endl;
                    break;
                }

                if (cv.x >= 0 && cv.x < dims &&
                    cv.y >= 0 && cv.y < dims && 
                    cv.z >= 0 && cv.z < dims) {
                    std::cout << "found cell at: " << cv.toString() << std::endl;
                    if (grid.getVoxelOccupied(cv)) {
                        std::cout << "found occupied cell at: " << cv.toString() << std::endl;
                        Vec4 voxel_color = grid.getVoxelColor(cv);
                        
                        float weight = voxel_color.z * (1.0f - accumulatedColor.z);
                        accumulatedColor.w += voxel_color.w * weight;
                        accumulatedColor.x += voxel_color.x * weight;
                        accumulatedColor.y += voxel_color.y * weight;
                        accumulatedColor.z += voxel_color.z * weight;
                    }

                    int minAxis = 0;
                    if (t_max.y < t_max.x) minAxis = 1;
                    if (t_max.z < t_max[minAxis]) minAxis = 2;
                    cv[minAxis] += step[minAxis];
                    t = t_max[minAxis];
                    t_max[minAxis] += t_delta[minAxis];
                }
            }
            if (accumulatedColor.z > 0) {
                std::cout << "setting a color at " << x << " and " << y << std::endl;
                float r = accumulatedColor.w + (1.0f - accumulatedColor.z) * 1.0f;
                float g = accumulatedColor.x + (1.0f - accumulatedColor.z) * 1.0f;
                float b = accumulatedColor.y + (1.0f - accumulatedColor.z) * 1.0f;
                
                img.setPixel(x, y, 
                    static_cast<uint8_t>(r * 255),
                    static_cast<uint8_t>(g * 255), 
                    static_cast<uint8_t>(b * 255),
                    255);
            }
            
        }
    }
    return img;
}

int main() {
    std::cout << "Generating point cloud" << std::endl;
    auto [points, colors] = genPointCloud(150000, 10.0, 43);
    std::cout << "Generating done" << std::endl;
    
    
    VoxelGrid voxel_grid(0.2f);
    
    std::cout << "Adding points to voxel grid..." << std::endl;
    voxel_grid.addPoints(points, colors);
    voxel_grid.printStats();
    
    // Use the voxel grid's actual bounds
    Vec3 min_bounds = voxel_grid.getMinBounds();
    Vec3 max_bounds = voxel_grid.getMaxBounds();
    Vec3 grid_center = (min_bounds + max_bounds) * 0.5;

    // Proper camera setup
    Vec3 rayOrigin(0, 0, 15);
    Vec3 forward = (grid_center - rayOrigin).normalize();
    Vec3 up(0, 1, 0);
    Vec3 right = forward.cross(up).normalize();
    auto dims = voxel_grid.getDimensions();
    int max_dim = std::max({dims[0], dims[1], dims[2]});

    Image img = render(50, 50, forward, right, up, rayOrigin, min_bounds, 
                    voxel_grid.getVoxelSize(), voxel_grid, max_dim);
    img.saveAsBMP("cpp_voxel_render.bmp");
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}
