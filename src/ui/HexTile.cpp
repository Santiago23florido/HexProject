#include "ui/HexTile.hpp"

// Implements a lightweight drawable hex tile wrapper around an SFML sprite.

HexTile::HexTile(const sf::Texture& texture) : sprite_(texture) {
    sf::FloatRect bounds = sprite_.getLocalBounds();
    sprite_.setOrigin(bounds.width / 2.0f, bounds.height / 2.0f); // enable rotation around center
}

void HexTile::setPosition(float x, float y) {
    sprite_.setPosition(x, y);
}

void HexTile::setRotation(float degrees) {
    sprite_.setRotation(degrees);
}

void HexTile::setScale(float factor) {
    setScale(factor, factor);
}

void HexTile::setScale(float scaleX, float scaleY) {
    scale_ = sf::Vector2f(scaleX, scaleY);
    sprite_.setScale(scale_.x, scale_.y);
}

void HexTile::setColor(const sf::Color& color) {
    sprite_.setColor(color);
}

sf::Vector2f HexTile::getPosition() const {
    return sprite_.getPosition();
}

float HexTile::getRotation() const {
    return sprite_.getRotation();
}

sf::Vector2f HexTile::getScale() const {
    return scale_;
}

void HexTile::draw(sf::RenderTarget& target) const {
    target.draw(sprite_);
}
