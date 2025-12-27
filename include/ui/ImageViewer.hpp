#pragma once

#include <SFML/Graphics.hpp>
#include <limits>
#include <string>
#include <vector>

class ImageViewer {
public:
    explicit ImageViewer(const std::string& texturePath);

    bool isReady() const;
    const std::string& getError() const;
    sf::Vector2u getTextureSize() const;

    // scale is relative to the texture size (1.0 = original pixels).
    void addTile(float centerX, float centerY, float scale);
    void addTile(float centerX, float centerY, float scaleX, float scaleY);
    void clearTiles();

    // viewCenterX/Y shift the layout center; NaN keeps it centered.
    int run(
        float viewCenterX = std::numeric_limits<float>::quiet_NaN(),
        float viewCenterY = std::numeric_limits<float>::quiet_NaN());

private:
    struct TileSpec {
        sf::Vector2f center;
        float scaleX;
        float scaleY;
    };

    bool computeBounds(sf::Vector2f& min, sf::Vector2f& max) const;
    sf::Vector2u computeWindowSize() const;
    sf::Vector2f computeOffset(
        const sf::Vector2u& windowSize,
        float viewCenterX,
        float viewCenterY) const;
    static float resolveCenter(float value, float fallback);

    sf::Texture texture_;
    sf::Vector2u textureSize_{0, 0};
    std::vector<TileSpec> tiles_;
    std::string error_;
    bool ready_ = false;
};

// Convenience wrapper for a single tile; scale is relative to texture size.
int runImageViewer(
    const std::string& texturePath,
    float scale,
    float centerX = std::numeric_limits<float>::quiet_NaN(),
    float centerY = std::numeric_limits<float>::quiet_NaN());
