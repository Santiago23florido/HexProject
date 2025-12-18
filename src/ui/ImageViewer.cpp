#include "ui/ImageViewer.hpp"
#include "ui/HexTile.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>
#include <stdexcept>

namespace {

sf::Texture loadTexture(const std::string& path) {
    sf::Texture tex;
    if (!tex.loadFromFile(path)) {
        throw std::runtime_error("Failed to load texture: " + path);
    }
    return tex;
}

sf::RenderWindow createWindowForTexture(const sf::Texture& tex) {
    sf::Vector2u texSize = tex.getSize();
    return sf::RenderWindow(sf::VideoMode(texSize.x, texSize.y), "Hex UI - Image");
}

HexTile makeCenteredTile(const sf::Texture& tex, const sf::RenderWindow& window) {
    HexTile tile(tex);
    sf::Vector2u texSize = tex.getSize();
    sf::Vector2u winSize = window.getSize();
    // Center tile if window is bigger than the image.
    tile.setPosition(
        static_cast<float>(winSize.x) / 2.0f,
        static_cast<float>(winSize.y) / 2.0f);
    return tile;
}

} // namespace

int runImageViewer(const std::string& texturePath) {
    sf::Texture texture;
    try {
        texture = loadTexture(texturePath);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    sf::RenderWindow window = createWindowForTexture(texture);
    HexTile tile = makeCenteredTile(texture, window);
    tile.setRotation(0.0f);
    tile.setScale(1.0f);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        window.clear(sf::Color(30, 30, 40));
        tile.draw(window);
        window.display();
    }
    return 0;
}
