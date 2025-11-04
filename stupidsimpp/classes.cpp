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
