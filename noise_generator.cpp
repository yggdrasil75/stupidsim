#include "noise_generator.hpp"
#include "vec_math.hpp"
#include "timing_decorator.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace NoiseGenerator {

float fade(float t) {
    TIME_FUNCTION;
    return t * t * t * (10 + t * (-15 + t * 6));
}

float clamp(float x, float lowerlimit, float upperlimit) {
    TIME_FUNCTION;
    if (x < lowerlimit) return lowerlimit;
    if (x > upperlimit) return upperlimit;
    return x;
}

float lerp(float t, float a, float b) {
    TIME_FUNCTION;
    return a + t * (b - a);
}

float grad(int hash, float x, float y, float z) {
    TIME_FUNCTION;
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);
    return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
}

float pnoise3d(const int p[512], float xf, float yf, float zf) {
    TIME_FUNCTION;
    int floorx = static_cast<int>(std::floor(xf));
    int floory = static_cast<int>(std::floor(yf));
    int floorz = static_cast<int>(std::floor(zf));
    
    int iX = floorx & 255;
    int iY = floory & 255;
    int iZ = floorz & 255;

    float x = xf - floorx;
    float y = yf - floory;
    float z = zf - floorz;

    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    int A  = p[iX] + iY;
    int AA = p[A] + iZ;
    int AB = p[A + 1] + iZ;

    int B  = p[iX + 1] + iY;
    int BA = p[B] + iZ;
    int BB = p[B + 1] + iZ;

    float x1 = lerp(u, grad(p[BA], x-1, y, z),   grad(p[AA], x, y, z));
    float x2 = lerp(u, grad(p[BB], x-1, y-1, z), grad(p[AB], x, y-1, z));
    float y1 = lerp(v, x1, x2);

    float x3 = lerp(u, grad(p[BA + 1], x-1, y, z-1),   grad(p[AA + 1], x, y, z-1));
    float x4 = lerp(u, grad(p[BB + 1], x-1, y-1, z-1), grad(p[AB + 1], x, y-1, z-1));
    float y2 = lerp(v, x3, x4);

    return lerp(w, y1, y2);
}

float pascalTri(float a, float b) {
    TIME_FUNCTION;
    float result = 1.0f;
    int b_int = static_cast<int>(b);
    for (int i = 0; i < b_int; ++i) {
        result *= (a - 1) / (i + 1);
    }
    return result;
}

float genSmooth(int N, float x) {
    TIME_FUNCTION;
    x = clamp(x, 0.0f, 1.0f);
    float result = 0.0f;
    for (int n = 0; n <= N; ++n) {
        result += pascalTri(-N - 1, n) * pascalTri(2 * N + 1, N - n) * std::pow(x, N + n + 1);
    }
    return result;
}

float inverse_smoothstep(float x) {
    TIME_FUNCTION;
    return 0.5f - std::sin(std::asin(1.0f - 2.0f * x) / 3.0f);
}

std::tuple<std::vector<Vec3>, std::vector<Vec4>> noiseBatch(int num_points, float scale, int p[512]) {
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
        
        float noise1 = pnoise3d(p, x * 0.5f, y * 0.5f, z * 0.5f);
        float noise2 = pnoise3d(p, x * 0.3f, y * 0.3f, z * 0.3f);
        float noise3 = pnoise3d(p, x * 0.7f, y * 0.7f, z * 0.7f);
        float noise4 = pnoise3d(p, x * 0.7f, y * 0.7f, z * 0.7f);
        
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
                points.emplace_back(x, y, z);
                colors.emplace_back(r, g, b, a);
            }
        }
    }
    return std::make_tuple(points, colors);
}

std::tuple<std::vector<Vec3>, std::vector<Vec4>> genPointCloud(int num_points, float scale, int seed) {
    TIME_FUNCTION;
    int permutation[256];
    for (int i = 0; i < 256; ++i) {
        permutation[i] = i;
    }
    std::mt19937 rng(seed);
    std::shuffle(permutation, permutation + 256, rng);
    
    int p[512];
    for (int i = 0; i < 256; ++i) {
        p[i] = permutation[i];
        p[i + 256] = permutation[i];
    }
    
    return noiseBatch(num_points, scale, p);
}

} // namespace NoiseGenerator