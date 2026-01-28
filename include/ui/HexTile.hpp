#pragma once

#include <SFML/Graphics.hpp>

/**
 *  Drawable hex tile wrapper around an SFML sprite.
 *
 * Owns the sprite and cached scale for layout queries.
 */
class HexTile {
public:
    /// Creates a tile using the provided texture.
    explicit HexTile(const sf::Texture& texture);

    /// Sets the sprite position in pixels.
    void setPosition(float x, float y);
    /// Sets the sprite rotation in degrees.
    void setRotation(float degrees);
    /// Sets a uniform scale factor.
    void setScale(float factor);
    /// Sets a non-uniform scale.
    void setScale(float scaleX, float scaleY);
    /// Sets the sprite tint color.
    void setColor(const sf::Color& color);

    /// Returns the sprite position in pixels.
    sf::Vector2f getPosition() const;
    /// Returns the sprite rotation in degrees.
    float getRotation() const;
    /// Returns the cached scale factors.
    sf::Vector2f getScale() const;

    /// Draws the sprite to the render target.
    void draw(sf::RenderTarget& target) const;

private:
    sf::Sprite sprite_;
    sf::Vector2f scale_{1.0f, 1.0f};
};
