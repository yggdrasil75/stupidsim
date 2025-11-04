#include "../stupidsimpp/timing_decorator.hpp"
#include "../stupidsimpp/classes.cpp"

// Noise generation functions
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

    float x = xf - floorx;
    float y = yf - floory;
    float z = zf - floorz;

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
