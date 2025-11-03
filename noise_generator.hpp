#ifndef NOISE_GENERATOR_HPP
#define NOISE_GENERATOR_HPP

#include <vector>
#include <random>
#include <tuple>
#include "vec_math.hpp"  // We'll need to extract Vec3 and Vec4 too

// Forward declarations
class Vec3;
class Vec4;

namespace NoiseGenerator {

// Core noise functions
float fade(float t);
float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f);
float lerp(float t, float a, float b);
float grad(int hash, float x, float y, float z);
float pnoise3d(const int p[512], float x, float y, float z);

// Smoothing functions
float pascalTri(float a, float b);
float genSmooth(int N, float x);
float inverse_smoothstep(float x);

// Point cloud generation
std::tuple<std::vector<Vec3>, std::vector<Vec4>> noiseBatch(int num_points, float scale, int permutation[512]);
std::tuple<std::vector<Vec3>, std::vector<Vec4>> genPointCloud(int num_points, float scale, int seed);

} // namespace NoiseGenerator

#endif // NOISE_GENERATOR_HPP