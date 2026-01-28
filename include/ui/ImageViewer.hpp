#pragma once

#include <SFML/Graphics.hpp>
#include <limits>
#include <string>
#include <vector>

/**
 * Utility viewer that lays out tiles and renders them in a window.
 *
 * Owns a texture and a list of tile placements.
 */
class ImageViewer {
public:
    /// Loads the texture from disk.
    explicit ImageViewer(const std::string& texturePath);

    /// Returns true if the texture loaded successfully.
    bool isReady() const;
    /// Returns the last error message, if any.
    const std::string& getError() const;
    /// Returns the loaded texture size in pixels.
    sf::Vector2u getTextureSize() const;

    // scale is relative to the texture size (1.0 = original pixels).
    /// Adds a tile with uniform scale centered at (centerX, centerY).
    void addTile(float centerX, float centerY, float scale);
    /// Adds a tile with non-uniform scale centered at (centerX, centerY).
    void addTile(float centerX, float centerY, float scaleX, float scaleY);
    /// Clears all queued tiles.
    void clearTiles();

    // viewCenterX/Y shift the layout center; NaN keeps it centered.
    /// Renders the tiles and returns an exit code.
    int run(
        float viewCenterX = std::numeric_limits<float>::quiet_NaN(),
        float viewCenterY = std::numeric_limits<float>::quiet_NaN());

private:
    /**
     *Tile placement specification for the viewer.
     */
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
/// Runs a viewer with a single tiled layout and returns an exit code.
int runImageViewer(
    const std::string& texturePath,
    float scale,
    float centerX = std::numeric_limits<float>::quiet_NaN(),
    float centerY = std::numeric_limits<float>::quiet_NaN());
