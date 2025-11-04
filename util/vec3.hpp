#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <algorithm>

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
    
    Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }
    
    Vec3 operator/(const Vec3& other) const {
        return Vec3(x / other.x, y / other.y, z / other.z);
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
    
    float lengthSquared() const {
        return x*x + y*y + z*z;  // Fast - no sqrt
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

#endif