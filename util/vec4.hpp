#ifndef VEC4_HPP
#define VEC4_HPP

#include "vec3.hpp"
#include <algorithm>
#include <cstdint>

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

#endif