#ifndef BMP_WRITER_HPP
#define BMP_WRITER_HPP

#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include "vec3.hpp"

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
    // Save a 2D vector of Vec3 (RGB) colors as BMP
    // Vec3 components: x = red, y = green, z = blue (values in range [0,1])
    static bool saveBMP(const std::string& filename, const std::vector<std::vector<Vec3>>& pixels) {
        if (pixels.empty() || pixels[0].empty()) {
            return false;
        }
        
        int height = static_cast<int>(pixels.size());
        int width = static_cast<int>(pixels[0].size());
        
        // Validate that all rows have the same width
        for (const auto& row : pixels) {
            if (row.size() != width) {
                return false;
            }
        }
        
        return saveBMP(filename, pixels, width, height);
    }
    
    // Alternative interface with width/height and flat vector (row-major order)
    static bool saveBMP(const std::string& filename, const std::vector<Vec3>& pixels, int width, int height) {
        if (pixels.size() != width * height) {
            return false;
        }
        
        // Convert to 2D vector format
        std::vector<std::vector<Vec3>> pixels2D(height, std::vector<Vec3>(width));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels2D[y][x] = pixels[y * width + x];
            }
        }
        
        return saveBMP(filename, pixels2D, width, height);
    }
    
    // Save from 1D vector of uint8_t pixels (BGR order: pixels[i]=b, pixels[i+1]=g, pixels[i+2]=r)
    static bool saveBMP(const std::string& filename, const std::vector<uint8_t>& pixels, int width, int height) {
        if (pixels.size() != width * height * 3) {
            return false;
        }
        
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
        std::vector<uint8_t> row(rowSize, 0);
        for (int y = height - 1; y >= 0; --y) {
            const uint8_t* srcRow = pixels.data() + (y * width * 3);
            
            // Copy and rearrange if necessary (input is already in BGR order)
            for (int x = 0; x < width; ++x) {
                int srcOffset = x * 3;
                int dstOffset = x * 3;
                
                // Input is already BGR: pixels[i]=b, pixels[i+1]=g, pixels[i+2]=r
                // So we can copy directly
                row[dstOffset] = srcRow[srcOffset];     // B
                row[dstOffset + 1] = srcRow[srcOffset + 1]; // G
                row[dstOffset + 2] = srcRow[srcOffset + 2]; // R
            }
            file.write(reinterpret_cast<const char*>(row.data()), rowSize);
        }
        
        return true;
    }
    
private:
    static bool saveBMP(const std::string& filename, const std::vector<std::vector<Vec3>>& pixels, int width, int height) {
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
        std::vector<uint8_t> row(rowSize, 0);
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                const Vec3& color = pixels[y][x];
                
                // Convert from [0,1] float to [0,255] uint8_t
                uint8_t r = static_cast<uint8_t>(std::clamp(color.x * 255.0f, 0.0f, 255.0f));
                uint8_t g = static_cast<uint8_t>(std::clamp(color.y * 255.0f, 0.0f, 255.0f));
                uint8_t b = static_cast<uint8_t>(std::clamp(color.z * 255.0f, 0.0f, 255.0f));
                
                // BMP is BGR order
                int pixelOffset = x * 3;
                row[pixelOffset] = b;
                row[pixelOffset + 1] = g;
                row[pixelOffset + 2] = r;
            }
            file.write(reinterpret_cast<const char*>(row.data()), rowSize);
        }
        
        return true;
    }
};

#endif