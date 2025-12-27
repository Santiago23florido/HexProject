#pragma once

#include <SFML/Graphics.hpp>

class HexTile {
public:
    explicit HexTile(const sf::Texture& texture);

    void setPosition(float x, float y);
    void setRotation(float degrees);
    void setScale(float factor);
    void setScale(float scaleX, float scaleY);
    void setColor(const sf::Color& color);

    sf::Vector2f getPosition() const;
    float getRotation() const;
    sf::Vector2f getScale() const;

    void draw(sf::RenderTarget& target) const;

private:
    sf::Sprite sprite_;
    sf::Vector2f scale_{1.0f, 1.0f};
};
