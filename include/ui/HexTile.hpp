#pragma once

#include <SFML/Graphics.hpp>

class HexTile {
public:
    explicit HexTile(const sf::Texture& texture);

    void setPosition(float x, float y);
    void setRotation(float degrees);
    void setScale(float factor);

    sf::Vector2f getPosition() const;
    float getRotation() const;
    float getScale() const;

    void draw(sf::RenderTarget& target) const;

private:
    sf::Sprite sprite_;
    float scale_ = 1.0f;
};
