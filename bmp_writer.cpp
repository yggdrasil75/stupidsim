#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "vec_math.hpp"

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
