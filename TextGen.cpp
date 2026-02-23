#define _CRT_SECURE_NO_WARNINGS
#define TEXTUREGENERATORCORE_EXPORTS
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include <string>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// Helper function for clamping values
template<typename T>
T clamp(T value, T min, T max) {
    return (value < min) ? min : (value > max) ? max : value;
}

// RGB Color struct
struct Color {
    float r, g, b;

    Color(float red = 0, float green = 0, float blue = 0)
        : r(red), g(green), b(blue) {
    }

    Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }

    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }
};

// Linear interpolation for colors
Color lerpColor(const Color& a, const Color& b, float t) {
    t = clamp(t, 0.0f, 1.0f);
    return Color(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t
    );
}

// ===========================
// IMPROVED NOISE GENERATOR WITH TILEABLE SUPPORT
// ===========================
class NoiseGenerator {
private:
    int width, height;
    float scale;
    int octaves;
    unsigned int seed;
    vector<int> permutation;
    bool tileable;

    void initPermutation() {
        permutation.resize(512);
        vector<int> p(256);

        for (int i = 0; i < 256; i++) {
            p[i] = i;
        }

        mt19937 rng(seed);
        shuffle(p.begin(), p.end(), rng);

        for (int i = 0; i < 256; i++) {
            permutation[i] = p[i];
            permutation[256 + i] = p[i];
        }
    }

    float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    float lerp(float t, float a, float b) {
        return a + t * (b - a);
    }

    float grad(int hash, float x, float y) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : 0);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    float perlin(float x, float y) {
        int X = (int)floor(x) & 255;
        int Y = (int)floor(y) & 255;

        x -= floor(x);
        y -= floor(y);

        float u = fade(x);
        float v = fade(y);

        int a = permutation[X] + Y;
        int aa = permutation[a];
        int ab = permutation[a + 1];
        int b = permutation[X + 1] + Y;
        int ba = permutation[b];
        int bb = permutation[b + 1];

        float res = lerp(v,
            lerp(u, grad(permutation[aa], x, y), grad(permutation[ba], x - 1, y)),
            lerp(u, grad(permutation[ab], x, y - 1), grad(permutation[bb], x - 1, y - 1))
        );

        return (res + 1.0f) / 2.0f;
    }

    // Tileable perlin noise using domain wrapping
    float perlinTileable(float x, float y, float wrapX, float wrapY) {
        // Get the integer parts
        int xi0 = ((int)floor(x)) % (int)wrapX;
        int yi0 = ((int)floor(y)) % (int)wrapY;
        int xi1 = (xi0 + 1) % (int)wrapX;
        int yi1 = (yi0 + 1) % (int)wrapY;

        // Handle negative wrapping
        if (xi0 < 0) xi0 += (int)wrapX;
        if (yi0 < 0) yi0 += (int)wrapY;
        if (xi1 < 0) xi1 += (int)wrapX;
        if (yi1 < 0) yi1 += (int)wrapY;

        // Get fractional parts
        float xf = x - floor(x);
        float yf = y - floor(y);

        // Fade curves
        float u = fade(xf);
        float v = fade(yf);

        // Hash coordinates
        int aa = permutation[permutation[xi0 & 255] + (yi0 & 255)];
        int ab = permutation[permutation[xi0 & 255] + (yi1 & 255)];
        int ba = permutation[permutation[xi1 & 255] + (yi0 & 255)];
        int bb = permutation[permutation[xi1 & 255] + (yi1 & 255)];

        // Calculate gradients
        float x00 = lerp(u, grad(aa, xf, yf), grad(ba, xf - 1, yf));
        float x10 = lerp(u, grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1));

        float result = lerp(v, x00, x10);

        return (result + 1.0f) / 2.0f;
    }

public:
    NoiseGenerator(int w, int h, float s, int oct, unsigned int sd, bool tile = false)
        : width(w), height(h), scale(s), octaves(oct), seed(sd), tileable(tile) {
        initPermutation();
    }

    vector<float> generateFBM(float persistence, float lacunarity) {
        vector<float> data(width * height);
        float minVal = 1e10f, maxVal = -1e10f;

        // Calculate wrap dimensions for tileable noise based on texture dimensions
        float wrapX = (float)width;
        float wrapY = (float)height;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float amplitude = 1.0f;
                float frequency = 1.0f;
                float noiseValue = 0.0f;

                for (int oct = 0; oct < octaves; oct++) {
                    float sampleX = (x / scale) * frequency;
                    float sampleY = (y / scale) * frequency;

                    float octaveNoise;
                    if (tileable) {
                        // Wrap dimensions scale with frequency
                        float octaveWrapX = (wrapX / scale) * frequency;
                        float octaveWrapY = (wrapY / scale) * frequency;
                        octaveNoise = perlinTileable(sampleX, sampleY, octaveWrapX, octaveWrapY);
                    }
                    else {
                        octaveNoise = perlin(sampleX, sampleY);
                    }

                    noiseValue += octaveNoise * amplitude;

                    amplitude *= persistence;
                    frequency *= lacunarity;
                }

                data[y * width + x] = noiseValue;
                minVal = min(minVal, noiseValue);
                maxVal = max(maxVal, noiseValue);
            }
        }

        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (data[i] - minVal) / (maxVal - minVal);
        }

        return data;
    }

    vector<float> generatePerlin() {
        vector<float> data(width * height);
        float minVal = 1e10f, maxVal = -1e10f;

        float wrapX = (float)width;
        float wrapY = (float)height;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sampleX = x / scale;
                float sampleY = y / scale;

                float noiseValue;
                if (tileable) {
                    float octaveWrapX = wrapX / scale;
                    float octaveWrapY = wrapY / scale;
                    noiseValue = perlinTileable(sampleX, sampleY, octaveWrapX, octaveWrapY);
                }
                else {
                    noiseValue = perlin(sampleX, sampleY);
                }

                data[y * width + x] = noiseValue;
                minVal = min(minVal, noiseValue);
                maxVal = max(maxVal, noiseValue);
            }
        }

        // Normalize to 0-1 range
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (data[i] - minVal) / (maxVal - minVal);
        }

        return data;
    }
};

// ===========================
// OVERLAY GENERATOR
// ===========================
class OverlayGenerator {
public:
    static vector<float> generateMarbleVeins(int width, int height, float veinScale, unsigned int seed, bool tileable = false) {
        vector<float> overlay(width * height, 0.0f);

        NoiseGenerator noiseGen(width, height, 80.0f, 4, seed + 1000, tileable);
        vector<float> noiseData = noiseGen.generateFBM(0.5f, 2.0f);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                float vein = x / veinScale + noiseData[idx] * 7.0f;
                vein = abs(sin(vein));

                overlay[idx] = 1.0f - vein;
            }
        }

        return overlay;
    }

    static vector<float> generateBrickPattern(int width, int height, float brickWidth, float brickHeight) {
        vector<float> overlay(width * height, 0.0f);
        float mortarWidth = 3.0f;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                int row = (int)(y / brickHeight);
                float xOffset = (row % 2) * (brickWidth * 0.5f);

                float localX = fmod(x + xOffset, brickWidth);
                float localY = fmod((float)y, brickHeight);

                bool isMortar = (localX < mortarWidth) ||
                    (localX > brickWidth - mortarWidth) ||
                    (localY < mortarWidth) ||
                    (localY > brickHeight - mortarWidth);

                overlay[idx] = isMortar ? 0.3f : 0.0f;
            }
        }

        return overlay;
    }

    static vector<float> generatePlankPattern(int width, int height, float plankWidth, unsigned int seed, bool uniqueGrain) {
        vector<float> overlay(width * height, 0.0f);
        float gapWidth = 2.0f;
        float plankHeight = plankWidth / 5.0f;

        // Base noise for grain if uniqueGrain is enabled
        NoiseGenerator grainNoise(width, height, 50.0f, 3, seed + 7000);
        vector<float> grainData;
        if (uniqueGrain) {
            grainData = grainNoise.generateFBM(0.5f, 2.0f);
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                int plankRow = (int)(y / plankHeight);
                float rowShift = (plankRow % 2) * (plankWidth * 0.5f);
                float shiftedX = x + rowShift;

                float posX = fmod(shiftedX, plankWidth);
                float posY = fmod((float)y, plankHeight);

                bool verticalGap = (posX < gapWidth);
                bool horizontalGap = (posY < gapWidth);

                if (verticalGap || horizontalGap) {
                    overlay[idx] = 0.6f; // Dark gap
                }
                else {
                    // Calculate which plank we're in
                    int plankCol = (int)(shiftedX / plankWidth);

                    if (uniqueGrain) {
                        // Each plank gets unique grain pattern
                        mt19937 plankRng(seed + plankRow * 1000 + plankCol);
                        uniform_real_distribution<float> grainOffset(0.0f, 100.0f);
                        uniform_real_distribution<float> colorShift(-0.1f, 0.1f);

                        float offset = grainOffset(plankRng);
                        float shift = colorShift(plankRng);

                        // Grain running lengthwise (horizontal)
                        float grain = sin((posY + offset) / 3.0f) * 0.15f;

                        // Add base noise with plank-specific shift
                        float plankValue = grainData[idx] * 0.2f + grain + shift;

                        overlay[idx] = plankValue;
                    }
                    else {
                        // Original - just color variation per plank
                        mt19937 plankRng(seed + plankRow * 1000 + plankCol);
                        uniform_real_distribution<float> plankColor(0.0f, 0.3f);

                        overlay[idx] = plankColor(plankRng);
                    }
                }
            }
        }

        return overlay;
    }
    static vector<float> generateScratches(int width, int height, int numScratches, unsigned int seed) {
        vector<float> overlay(width * height, 0.0f);

        mt19937 rng(seed + 999);
        uniform_real_distribution<float> distPos(0.0f, 1.0f);
        uniform_real_distribution<float> distAngle(0.0f, 3.14159f * 2.0f);
        uniform_real_distribution<float> distLength(50.0f, 250.0f);
        uniform_real_distribution<float> distWidth(0.8f, 2.0f);

        for (int s = 0; s < numScratches; s++) {
            float startX = distPos(rng) * width;
            float startY = distPos(rng) * height;
            float angle = distAngle(rng);
            float length = distLength(rng);
            float scratchWidth = distWidth(rng);

            float dx = cos(angle);
            float dy = sin(angle);

            for (float t = 0; t < length; t += 0.3f) {
                int centerX = (int)(startX + dx * t);
                int centerY = (int)(startY + dy * t);

                float widthVar = sin(t / length * 3.14159f) * scratchWidth;

                for (int w = -(int)widthVar; w <= (int)widthVar; w++) {
                    int x = centerX + (int)(dy * w);
                    int y = centerY - (int)(dx * w);

                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = y * width + x;
                        float intensity = 1.0f - abs(w) / (widthVar + 0.1f);
                        overlay[idx] = max(overlay[idx], intensity * 0.6f);
                    }
                }
            }
        }

        return overlay;
    }

    static void applyOverlay(vector<float>& base, const vector<float>& overlay, float intensity) {
        for (size_t i = 0; i < base.size(); i++) {
            base[i] = clamp(base[i] - overlay[i] * intensity, 0.0f, 1.0f);
        }
    }
};

// ===========================
// PATTERN GENERATOR
// ===========================
class PatternGenerator {
public:
    static vector<float> generateWood(const vector<float>& noiseData, int width, int height, float ringScale, bool tileable = false) {
        vector<float> pattern(width * height);

        if (tileable) {
            // For tileable wood, use a distance field that wraps
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;

                    // Use wrapped distance calculation
                    float dx = min(abs(x - width / 2.0f), width - abs(x - width / 2.0f));
                    float dy = min(abs(y - height / 2.0f), height - abs(y - height / 2.0f));
                    float distance = sqrt(dx * dx + dy * dy);

                    float grain = distance / ringScale;
                    grain += noiseData[idx] * 3.0f;
                    float rings = sin(grain * 3.14159f * 2.0f) * 0.5f + 0.5f;
                    float detail = noiseData[idx] * 0.4f + 0.3f;
                    pattern[idx] = clamp(rings * 0.5f + detail * 0.5f, 0.0f, 1.0f);
                }
            }
        }
        else {
            // Original non-tileable wood
            float centerX = width / 2.0f;
            float centerY = height / 2.0f;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;
                    float dx = x - centerX;
                    float dy = y - centerY;
                    float distance = sqrt(dx * dx + dy * dy);
                    float grain = distance / ringScale;
                    grain += noiseData[idx] * 3.0f;
                    float rings = sin(grain * 3.14159f * 2.0f) * 0.5f + 0.5f;
                    float detail = noiseData[idx] * 0.4f + 0.3f;
                    pattern[idx] = clamp(rings * 0.5f + detail * 0.5f, 0.0f, 1.0f);
                }
            }
        }
        return pattern;
    }

    static vector<float> generateMarble(const vector<float>& noiseData, int width, int height, float veinScale, bool tileable = false) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Marble veins work fine with tiling since they're based on x+y
                float vein = (x + y * 0.6f) / veinScale + noiseData[idx] * 8.0f;
                vein = sin(vein);
                vein = abs(vein);
                vein = pow(vein, 1.5f);
                vein = 1.0f - vein;
                float base = noiseData[idx] * 0.15f + 0.85f;
                float result = base - vein * 0.4f;
                pattern[idx] = clamp(result, 0.0f, 1.0f);
            }
        }
        return pattern;
    }

    static vector<float> generateClouds(const vector<float>& noiseData, int width, int height, float density) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float cloud = noiseData[idx];

                density = clamp(density, 0.5f, 3.0f);
                cloud = pow(cloud, 1.0f / density);
                cloud = cloud * cloud * (3.0f - 2.0f * cloud);

                pattern[idx] = clamp(cloud, 0.0f, 1.0f);
            }
        }

        return pattern;
    }

    static vector<float> generateStone(const vector<float>& noiseData, int width, int height, float roughness) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float stone = noiseData[idx];
                float cracks = abs(stone - 0.5f) * 2.0f;
                cracks = pow(cracks, roughness);
                float surface = stone * 0.7f + 0.3f;
                pattern[idx] = clamp(surface * (1.0f - cracks * 0.3f), 0.0f, 1.0f);
            }
        }
        return pattern;
    }

    static vector<float> generateVoronoi(const vector<float>& noiseData, int width, int height, float cellSize, unsigned int seed) {
        vector<float> pattern(width * height);

        // Generate random cell points
        int numCells = max(10, (int)((width * height) / (cellSize * cellSize * 2)));
        vector<pair<float, float>> points;

        mt19937 rng(seed + 500);
        uniform_real_distribution<float> distX(0, (float)width);
        uniform_real_distribution<float> distY(0, (float)height);

        for (int i = 0; i < numCells; i++) {
            points.push_back({ distX(rng), distY(rng) });
        }

        // Calculate Voronoi cells with edge detection
        float maxDist = sqrt((float)(width * width + height * height));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Find two closest points
                float minDist1 = maxDist;
                float minDist2 = maxDist;

                for (const auto& point : points) {
                    float dx = x - point.first;
                    float dy = y - point.second;
                    float dist = sqrt(dx * dx + dy * dy);

                    if (dist < minDist1) {
                        minDist2 = minDist1;
                        minDist1 = dist;
                    }
                    else if (dist < minDist2) {
                        minDist2 = dist;
                    }
                }

                // Cell edge intensity (distance between closest and second-closest)
                float edgeValue = (minDist2 - minDist1) / cellSize;
                edgeValue = clamp(edgeValue * 2.0f, 0.0f, 1.0f);

                // Invert for darker cells, lighter edges
                float value = 1.0f - edgeValue;

                // Add noise for organic variation
                value = value * 0.8f + noiseData[idx] * 0.2f;

                pattern[idx] = clamp(value, 0.0f, 1.0f);
            }
        }

        return pattern;
    }


    static vector<float> generateScales(const vector<float>& noiseData, int width, int height, float scaleSize) {
        vector<float> pattern(width * height, 0.3f); // Dark base

        float scaleRadius = scaleSize / 2.0f;
        float rowHeight = scaleSize * 0.7f; // Vertical spacing

        // Add extra rows/cols to avoid cutoff
        int startRow = -2;
        int endRow = (int)(height / rowHeight) + 2;
        int startCol = -2;
        int endCol = (int)(width / scaleSize) + 2;

        for (int row = startRow; row <= endRow; row++) {
            for (int col = startCol; col <= endCol; col++) {
                // Scale center position
                float centerX = col * scaleSize;
                float centerY = row * rowHeight;

                // Offset every other row for scale overlap
                if (row % 2 != 0) {
                    centerX += scaleSize / 2.0f;
                }

                // Draw each scale
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int idx = y * width + x;

                        float dx = x - centerX;
                        float dy = y - centerY;
                        float dist = sqrt(dx * dx + dy * dy);

                        if (dist < scaleRadius) {
                            // Scale body - brighten toward center
                            float normalizedDist = dist / scaleRadius;
                            float value = 1.0f - normalizedDist * 0.5f;

                            // Add scale edge highlight
                            if (normalizedDist > 0.85f) {
                                value += (normalizedDist - 0.85f) / 0.15f * 0.3f;
                            }

                            // Add subtle noise
                            value += noiseData[idx] * 0.1f;

                            pattern[idx] = max(pattern[idx], clamp(value, 0.0f, 1.0f));
                        }
                    }
                }
            }
        }

        return pattern;
    }

    

    static vector<float> generateCheckerboard(int width, int height, float tileSize) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                int tileX = (int)(x / tileSize);
                int tileY = (int)(y / tileSize);

                bool isWhite = ((tileX + tileY) % 2) == 0;

                pattern[idx] = isWhite ? 1.0f : 0.0f;
            }
        }

        return pattern;
    }

    

    static vector<float> generateLeather(const vector<float>& noiseData, int width, int height, float grainSize) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Multiple scales of noise for leather texture
                float n1 = noiseData[idx];

                // Grain pattern
                float grain = sin(x / grainSize) * sin(y / grainSize);
                grain = abs(grain) * 0.3f;

                // Combine
                float value = n1 * 0.5f + grain + 0.3f;
                value = pow(value, 1.2f); // Darken

                pattern[idx] = clamp(value, 0.0f, 1.0f);
            }
        }

        return pattern;
    }
    static vector<float> generateWoodGrain(const vector<float>& noiseData, int width, int height, float grainScale) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Base wood color
                float baseWood = 0.45f + noiseData[idx] * 0.2f;

                // Grain lines running horizontally
                float grainLine = sin(y / grainScale + noiseData[idx] * 3.0f) * 0.15f;

                // Fine grain detail
                float fineGrain = sin(y / (grainScale * 0.2f)) * 0.03f;

                // Occasional knots/irregularities
                float knot = 0.0f;
                float knotDist = sqrt(
                    pow(fmod((float)x, grainScale * 3.0f) - grainScale * 1.5f, 2) +
                    pow(fmod((float)y, grainScale * 5.0f) - grainScale * 2.5f, 2)
                );
                if (knotDist < grainScale * 0.5f) {
                    knot = (1.0f - knotDist / (grainScale * 0.5f)) * -0.2f;
                }

                // Color bands (growth rings running lengthwise)
                float bands = sin(x / (grainScale * 2.0f)) * 0.08f;

                float value = baseWood + grainLine + fineGrain + knot + bands;
                pattern[idx] = clamp(value, 0.0f, 1.0f);
            }
        }

        return pattern;
    }
    static vector<float> generateBrushedMetal(const vector<float>& noiseData, int width, int height, float scratchDensity, unsigned int seed) {
        vector<float> pattern(width * height);

        // Initialize with base metal color + fine noise
        for (int i = 0; i < width * height; i++) {
            pattern[i] = 0.55f + noiseData[i] * 0.08f;
        }

        // Add random scratches
        int numScratches = (int)(width * height * scratchDensity / 10000.0f);
        numScratches = max(50, min(500, numScratches)); // Clamp between 50-500 scratches

        mt19937 rng(seed + 3000);
        uniform_real_distribution<float> distX(0.0f, (float)width);
        uniform_real_distribution<float> distY(0.0f, (float)height);
        uniform_real_distribution<float> distAngle(-0.3f, 0.3f); // Mostly horizontal
        uniform_real_distribution<float> distLength(20.0f, 200.0f);
        uniform_real_distribution<float> distIntensity(0.05f, 0.15f);

        for (int s = 0; s < numScratches; s++) {
            float startX = distX(rng);
            float startY = distY(rng);
            float angle = distAngle(rng); // Mostly horizontal scratches
            float length = distLength(rng);
            float intensity = distIntensity(rng);

            float dx = cos(angle);
            float dy = sin(angle);

            // Draw scratch
            for (float t = 0; t < length; t += 0.5f) {
                int x = (int)(startX + dx * t);
                int y = (int)(startY + dy * t);

                if (x >= 0 && x < width && y >= 0 && y < height) {
                    int idx = y * width + x;

                    // Lighter scratches on metal
                    float fade = 1.0f - (t / length); // Fade toward end
                    pattern[idx] += intensity * fade;
                }
            }
        }

        // Add some subtle directional brushing (very fine)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Very subtle horizontal brush texture
                float microBrush = sin(y * 2.0f + noiseData[idx] * 10.0f) * 0.01f;
                pattern[idx] += microBrush;

                pattern[idx] = clamp(pattern[idx], 0.0f, 1.0f);
            }
        }

        return pattern;
    }
    static vector<float> generateDiamondPlate(const vector<float>& noiseData, int width, int height, float plateSize) {
        vector<float> pattern(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                // Diamond pattern offset
                float offsetX = x + y * 0.5f;

                // Create diamond grid
                float gridX = fmod(offsetX, plateSize);
                float gridY = fmod((float)y, plateSize);

                // Distance from diamond center
                float centerX = plateSize / 2.0f;
                float centerY = plateSize / 2.0f;
                float dx = abs(gridX - centerX);
                float dy = abs(gridY - centerY);

                // Diamond shape (45-degree rotated square)
                float diamondDist = dx + dy;

                // Raised diamond pattern
                float raised = 0.0f;
                if (diamondDist < plateSize * 0.4f) {
                    raised = (1.0f - diamondDist / (plateSize * 0.4f)) * 0.15f;
                }

                // Base metal with noise
                float metalValue = 0.55f + noiseData[idx] * 0.1f + raised;

                pattern[idx] = clamp(metalValue, 0.0f, 1.0f);
            }
        }

        return pattern;
    }
};

// ===========================
// TEXTURE PROCESSOR
// ===========================
class TextureProcessor {
public:
    static void adjustContrast(vector<float>& data, float contrast) {
        for (size_t i = 0; i < data.size(); i++) {
            float value = data[i];
            value = (value - 0.5f) * contrast + 0.5f;
            data[i] = clamp(value, 0.0f, 1.0f);
        }
    }

    static void adjustBrightness(vector<float>& data, float brightness) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = clamp(data[i] + brightness, 0.0f, 1.0f);
        }
    }

    static void applyPowerCurve(vector<float>& data, float power) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = clamp(pow(data[i], power), 0.0f, 1.0f);
        }
    }

    static void invert(vector<float>& data) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = 1.0f - data[i];
        }
    }
};

// ===========================
// COLOR MAPPER
// ===========================
class ColorMapper {
public:
    static vector<Color> applyGradient(const vector<float>& data, const Color& color1, const Color& color2) {
        vector<Color> colorData(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            colorData[i] = lerpColor(color1, color2, data[i]);
        }
        return colorData;
    }

    static vector<Color> applyTriGradient(const vector<float>& data, const Color& color1, const Color& color2, const Color& color3) {
        vector<Color> colorData(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            float value = data[i];
            if (value < 0.5f) {
                colorData[i] = lerpColor(color1, color2, value * 2.0f);
            }
            else {
                colorData[i] = lerpColor(color2, color3, (value - 0.5f) * 2.0f);
            }
        }
        return colorData;
    }

    static vector<Color> applyPreset(const vector<float>& data, int presetType) {
        switch (presetType) {
        case 2:
            return applyTriGradient(data,
                Color(0.25f, 0.15f, 0.05f),
                Color(0.55f, 0.35f, 0.15f),
                Color(0.75f, 0.55f, 0.25f));
        case 3:
            return applyTriGradient(data,
                Color(0.3f, 0.3f, 0.35f),
                Color(0.9f, 0.9f, 0.92f),
                Color(1.0f, 1.0f, 1.0f));
        case 4:
            return applyTriGradient(data,
                Color(0.25f, 0.25f, 0.28f),
                Color(0.5f, 0.5f, 0.53f),
                Color(0.75f, 0.75f, 0.78f));
        case 5:
            return applyTriGradient(data,
                Color(0.15f, 0.25f, 0.08f),
                Color(0.35f, 0.55f, 0.18f),
                Color(0.55f, 0.75f, 0.35f));
        case 6:
            return applyTriGradient(data,
                Color(0.05f, 0.15f, 0.35f),
                Color(0.15f, 0.35f, 0.65f),
                Color(0.35f, 0.65f, 0.95f));
        case 7:
            return applyTriGradient(data,
                Color(0.15f, 0.0f, 0.0f),
                Color(0.9f, 0.35f, 0.0f),
                Color(1.0f, 0.95f, 0.3f));
        default:
            return applyGradient(data, Color(0, 0, 0), Color(1, 1, 1));
        }
    }
};

// ===========================
// TEXTURE EXPORTER
// ===========================
class TextureExporter {
public:
    static bool exportGrayscalePNG(const vector<float>& data, int width, int height, const string& filename) {
        vector<unsigned char> imageData(width * height);
        for (size_t i = 0; i < data.size(); i++) {
            imageData[i] = static_cast<unsigned char>(clamp(data[i], 0.0f, 1.0f) * 255.0f);
        }
        int result = stbi_write_png(filename.c_str(), width, height, 1, imageData.data(), width);
        return result != 0;
    }

    static bool exportColorPNG(const vector<Color>& data, int width, int height, const string& filename) {
        vector<unsigned char> imageData(width * height * 3);
        for (size_t i = 0; i < data.size(); i++) {
            imageData[i * 3 + 0] = static_cast<unsigned char>(clamp(data[i].r, 0.0f, 1.0f) * 255.0f);
            imageData[i * 3 + 1] = static_cast<unsigned char>(clamp(data[i].g, 0.0f, 1.0f) * 255.0f);
            imageData[i * 3 + 2] = static_cast<unsigned char>(clamp(data[i].b, 0.0f, 1.0f) * 255.0f);
        }
        int result = stbi_write_png(filename.c_str(), width, height, 3, imageData.data(), width * 3);
        return result != 0;
    }
};

// DLL Export declarations
#ifdef TEXTUREGENERATORCORE_EXPORTS
#define TEXTUREAPI __declspec(dllexport)
#else
#define TEXTUREAPI __declspec(dllimport)
#endif

extern "C" {
    TEXTUREAPI int __stdcall TestDLL() {
        return 42;
    }

    TEXTUREAPI int __stdcall GenerateTexture(
        int width, int height,
        int noiseType, float scale, int octaves, float persistence, float lacunarity,
        int colorMode, int patternType, float patternParam,
        int overlayType, float overlayParam1, float overlayParam2,
        unsigned int seed,
        float contrast, float brightness, int invert,
        int tileable,
        float normalStrength,    // NEW: Normal map strength (1.0-10.0)
        float specularSmoothness, // NEW: Specular smoothness (0.0-1.0)
        int exportRoughness,     // NEW: Export roughness map?
        int exportMetallic,      // NEW: Export metallic map?
        const wchar_t* outputPath
    )
    {
        try {
            // Convert wchar_t to string (same as original working code)
            char filename[512];
            size_t converted;
            wcstombs_s(&converted, filename, 512, outputPath, 512);
            string filenameStr(filename);

            // Generate base texture with tileable support
            NoiseGenerator generator(width, height, scale, octaves, seed, tileable == 1);

            vector<float> textureData;
            if (noiseType == 1) {
                textureData = generator.generatePerlin();
            }
            else {
                textureData = generator.generateFBM(persistence, lacunarity);
            }

            // Apply pattern
            if (patternType == 2) {
                textureData = PatternGenerator::generateWood(textureData, width, height, patternParam, tileable == 1);
            }
            else if (patternType == 3) {
                textureData = PatternGenerator::generateMarble(textureData, width, height, patternParam, tileable == 1);
            }
            else if (patternType == 4) {
                textureData = PatternGenerator::generateClouds(textureData, width, height, patternParam);
            }
            else if (patternType == 5) {
                textureData = PatternGenerator::generateStone(textureData, width, height, patternParam);
            }
            else if (patternType == 6) {
                textureData = PatternGenerator::generateVoronoi(textureData, width, height, patternParam, seed);
            }
            else if (patternType == 7) {
                textureData = PatternGenerator::generateBrushedMetal(textureData, width, height, patternParam, seed);
            }
            else if (patternType == 8) {
                textureData = PatternGenerator::generateScales(textureData, width, height, patternParam);
            }
            else if (patternType == 9) {
                textureData = PatternGenerator::generateCheckerboard(width, height, patternParam);
            }
            else if (patternType == 10) {
                textureData = PatternGenerator::generateWoodGrain(textureData, width, height, patternParam);
            }
            else if (patternType == 11) {
                textureData = PatternGenerator::generateDiamondPlate(textureData, width, height, patternParam);
            }
            else if (patternType == 12) {
                textureData = PatternGenerator::generateLeather(textureData, width, height, patternParam);
            }

            // Apply overlay
            if (overlayType == 1) {
                vector<float> veins = OverlayGenerator::generateMarbleVeins(width, height, overlayParam1, seed, tileable == 1);
                OverlayGenerator::applyOverlay(textureData, veins, overlayParam2);
            }
            else if (overlayType == 2) {
                float brickWidth = overlayParam1;
                float brickHeight = overlayParam2;
                vector<float> bricks = OverlayGenerator::generateBrickPattern(width, height, brickWidth, brickHeight);
                OverlayGenerator::applyOverlay(textureData, bricks, 1.0f);
            }
            else if (overlayType == 3) {
                bool uniqueGrain = (overlayParam2 > 0.5f); // Use overlayParam2 as toggle
                vector<float> planks = OverlayGenerator::generatePlankPattern(width, height, overlayParam1, seed, uniqueGrain);
                OverlayGenerator::applyOverlay(textureData, planks, 1.0f);
            }
            else if (overlayType == 4) {
                vector<float> scratches = OverlayGenerator::generateScratches(width, height, (int)overlayParam1, seed);
                OverlayGenerator::applyOverlay(textureData, scratches, overlayParam2);
            }

            vector<float> heightMap = textureData;

            // Apply advanced effects
            if (contrast != 1.0f) {
                TextureProcessor::adjustContrast(textureData, contrast);
            }
            if (brightness != 0.0f) {
                TextureProcessor::adjustBrightness(textureData, brightness);
            }
            if (invert == 1) {
                TextureProcessor::invert(textureData);
            }

            // Export diffuse
            string diffPath = filenameStr;
            size_t extPos = diffPath.rfind(".png");
            if (extPos != string::npos) {
                diffPath = diffPath.substr(0, extPos) + "_diff.png";
            }

            bool success = false;
            if (colorMode == 1 || colorMode == 8) {
                success = TextureExporter::exportGrayscalePNG(textureData, width, height, diffPath);
            }
            else {
                vector<Color> colorData = ColorMapper::applyPreset(textureData, colorMode);
                success = TextureExporter::exportColorPNG(colorData, width, height, diffPath);
            }

            if (!success) return 0;

            // === GENERATE AND EXPORT NORMAL MAP (_nmap.png) ===
            string nmapPath = filenameStr;
            extPos = nmapPath.rfind(".png");
            if (extPos != string::npos) {
                nmapPath = nmapPath.substr(0, extPos) + "_nmap.png";
            }

            vector<Color> normalMap(width * height);
            float strength = normalStrength; // Use user-defined strength

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;

                    int xLeft = (x > 0) ? x - 1 : (tileable ? width - 1 : 0);
                    int xRight = (x < width - 1) ? x + 1 : (tileable ? 0 : width - 1);
                    int yUp = (y > 0) ? y - 1 : (tileable ? height - 1 : 0);
                    int yDown = (y < height - 1) ? y + 1 : (tileable ? 0 : height - 1);

                    float hL = heightMap[y * width + xLeft];
                    float hR = heightMap[y * width + xRight];
                    float hU = heightMap[yUp * width + x];
                    float hD = heightMap[yDown * width + x];

                    float dx = (hR - hL) * strength;
                    float dy = (hD - hU) * strength;

                    float nx = -dx;
                    float ny = -dy;
                    float nz = 1.0f;

                    float length = sqrt(nx * nx + ny * ny + nz * nz);
                    nx /= length;
                    ny /= length;
                    nz /= length;

                    normalMap[idx].r = (nx + 1.0f) * 0.5f;
                    normalMap[idx].g = (ny + 1.0f) * 0.5f;
                    normalMap[idx].b = (nz + 1.0f) * 0.5f;
                }
            }

            success = TextureExporter::exportColorPNG(normalMap, width, height, nmapPath);
            if (!success) return 0;

            // Generate and export specular map
            string specPath = filenameStr;
            extPos = specPath.rfind(".png");
            if (extPos != string::npos) {
                specPath = specPath.substr(0, extPos) + "_spec.png";
            }

            vector<float> specularMap(width * height);
            for (int i = 0; i < width * height; i++) {
                specularMap[i] = 1.0f - (heightMap[i] * 0.7f);
                specularMap[i] = clamp(specularMap[i], 0.0f, 1.0f);
            }

            success = TextureExporter::exportGrayscalePNG(specularMap, width, height, specPath);

            // === GENERATE AND EXPORT ROUGHNESS MAP (_rough.png) - OPTIONAL ===
            if (exportRoughness == 1) {
                string roughPath = filenameStr;
                extPos = roughPath.rfind(".png");
                if (extPos != string::npos) {
                    roughPath = roughPath.substr(0, extPos) + "_rough.png";
                }

                // Roughness is inverse of smoothness
                vector<float> roughnessMap(width * height);
                for (int i = 0; i < width * height; i++) {
                    // Base roughness from height variation
                    float baseRough = heightMap[i] * 0.7f;
                    // Apply inverse smoothness
                    roughnessMap[i] = baseRough + (1.0f - specularSmoothness) * 0.3f;
                    roughnessMap[i] = clamp(roughnessMap[i], 0.0f, 1.0f);
                }

                TextureExporter::exportGrayscalePNG(roughnessMap, width, height, roughPath);
            }

            // === GENERATE AND EXPORT METALLIC MAP (_metal.png) - OPTIONAL ===
            if (exportMetallic == 1) {
                string metalPath = filenameStr;
                extPos = metalPath.rfind(".png");
                if (extPos != string::npos) {
                    metalPath = metalPath.substr(0, extPos) + "_metal.png";
                }

                // Metallic based on smooth areas (inverted height variation)
                vector<float> metallicMap(width * height);
                for (int i = 0; i < width * height; i++) {
                    // Smoother areas = more metallic
                    metallicMap[i] = (1.0f - heightMap[i]) * 0.5f;
                    metallicMap[i] = clamp(metallicMap[i], 0.0f, 1.0f);
                }

                TextureExporter::exportGrayscalePNG(metallicMap, width, height, metalPath);
            }

            return success ? 1 : 0;

        }
        catch (...) {
            return 0;
        }
    }
}