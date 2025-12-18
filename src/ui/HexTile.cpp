#include "ui/HexTile.hpp"

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
    scale_ = factor;
    sprite_.setScale(scale_, scale_);
}

sf::Vector2f HexTile::getPosition() const {
    return sprite_.getPosition();
}

float HexTile::getRotation() const {
    return sprite_.getRotation();
}

float HexTile::getScale() const {
    return scale_;
}

void HexTile::draw(sf::RenderTarget& target) const {
    target.draw(sprite_);
}
