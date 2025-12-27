#include "ui/ImageViewer.hpp"
#include "ui/HexTile.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

constexpr float kWindowMargin = 24.0f;

ImageViewer::ImageViewer(const std::string& texturePath) {
    if (!texture_.loadFromFile(texturePath)) {
        error_ = "Failed to load texture: " + texturePath;
        return;
    }

    textureSize_ = texture_.getSize();
    ready_ = true;
}

bool ImageViewer::isReady() const {
    return ready_;
}

const std::string& ImageViewer::getError() const {
    return error_;
}

sf::Vector2u ImageViewer::getTextureSize() const {
    return textureSize_;
}

void ImageViewer::addTile(float centerX, float centerY, float scale) {
    addTile(centerX, centerY, scale, scale);
}

void ImageViewer::addTile(float centerX, float centerY, float scaleX, float scaleY) {
    if (scaleX <= 0.0f || scaleY <= 0.0f) {
        if (error_.empty()) {
            error_ = "Tile scale must be positive.";
        }
        return;
    }

    tiles_.push_back({sf::Vector2f(centerX, centerY), scaleX, scaleY});
}

void ImageViewer::clearTiles() {
    tiles_.clear();
}

float ImageViewer::resolveCenter(float value, float fallback) {
    return std::isnan(value) ? fallback : value;
}

bool ImageViewer::computeBounds(sf::Vector2f& min, sf::Vector2f& max) const {
    if (tiles_.empty()) {
        return false;
    }

    const TileSpec& first = tiles_.front();
    float firstHalfW = static_cast<float>(textureSize_.x) * first.scaleX / 2.0f;
    float firstHalfH = static_cast<float>(textureSize_.y) * first.scaleY / 2.0f;
    min.x = first.center.x - firstHalfW;
    max.x = first.center.x + firstHalfW;
    min.y = first.center.y - firstHalfH;
    max.y = first.center.y + firstHalfH;

    for (const auto& tile : tiles_) {
        float halfW = static_cast<float>(textureSize_.x) * tile.scaleX / 2.0f;
        float halfH = static_cast<float>(textureSize_.y) * tile.scaleY / 2.0f;
        float left = tile.center.x - halfW;
        float right = tile.center.x + halfW;
        float top = tile.center.y - halfH;
        float bottom = tile.center.y + halfH;

        min.x = std::min(min.x, left);
        max.x = std::max(max.x, right);
        min.y = std::min(min.y, top);
        max.y = std::max(max.y, bottom);
    }
    return true;
}

sf::Vector2u ImageViewer::computeWindowSize() const {
    sf::Vector2f min;
    sf::Vector2f max;
    if (!computeBounds(min, max)) {
        return sf::Vector2u(0, 0);
    }

    float boardWidth = max.x - min.x;
    float boardHeight = max.y - min.y;

    return sf::Vector2u(
        static_cast<unsigned int>(std::ceil(boardWidth + 2.0f * kWindowMargin)),
        static_cast<unsigned int>(std::ceil(boardHeight + 2.0f * kWindowMargin)));
}

sf::Vector2f ImageViewer::computeOffset(
    const sf::Vector2u& windowSize,
    float viewCenterX,
    float viewCenterY) const {
    sf::Vector2f min;
    sf::Vector2f max;
    computeBounds(min, max);

    sf::Vector2f layoutCenter((min.x + max.x) / 2.0f, (min.y + max.y) / 2.0f);
    float windowCenterX = static_cast<float>(windowSize.x) / 2.0f;
    float windowCenterY = static_cast<float>(windowSize.y) / 2.0f;
    float resolvedCenterX = resolveCenter(viewCenterX, windowCenterX);
    float resolvedCenterY = resolveCenter(viewCenterY, windowCenterY);

    return sf::Vector2f(
        resolvedCenterX - layoutCenter.x,
        resolvedCenterY - layoutCenter.y);
}

int ImageViewer::run(float viewCenterX, float viewCenterY) {
    if (!ready_) {
        if (!error_.empty()) {
            std::cerr << error_ << "\n";
        } else {
            std::cerr << "Viewer is not ready.\n";
        }
        return 1;
    }

    if (!error_.empty()) {
        std::cerr << error_ << "\n";
        return 1;
    }

    if (tiles_.empty()) {
        std::cerr << "No tiles to render.\n";
        return 1;
    }

    sf::Vector2u windowSize = computeWindowSize();
    if (windowSize.x == 0 || windowSize.y == 0) {
        std::cerr << "Invalid window size.\n";
        return 1;
    }

    sf::RenderWindow window(
        sf::VideoMode(windowSize.x, windowSize.y),
        "Hex UI - Viewer");
    window.setFramerateLimit(60);

    sf::Vector2f offset = computeOffset(windowSize, viewCenterX, viewCenterY);

    std::vector<HexTile> tiles;
    tiles.reserve(tiles_.size());
    for (const auto& tileSpec : tiles_) {
        HexTile tile(texture_);
        tile.setScale(tileSpec.scaleX, tileSpec.scaleY);
        tile.setPosition(tileSpec.center.x + offset.x, tileSpec.center.y + offset.y);
        tiles.push_back(tile);
    }

    std::sort(tiles.begin(), tiles.end(), [](const HexTile& a, const HexTile& b) {
        sf::Vector2f pa = a.getPosition();
        sf::Vector2f pb = b.getPosition();
        if (pa.y == pb.y) {
            return pa.x < pb.x;
        }
        return pa.y < pb.y;
    });

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        window.clear(sf::Color(30, 30, 40));
        for (const auto& tile : tiles) {
            tile.draw(window);
        }
        window.display();
    }
    return 0;
}

int runImageViewer(
    const std::string& texturePath,
    float scale,
    float centerX,
    float centerY) {

    if (scale <= 0.0f) {
        std::cerr << "Tile scale must be positive.\n";
        return 1;
    }

    ImageViewer viewer(texturePath);
    if (!viewer.isReady()) {
        std::cerr << viewer.getError() << "\n";
        return 1;
    }

    const sf::Vector2u tex = viewer.getTextureSize();

    const float sx = scale;
    const float sy = scale;

    const float halfW = (static_cast<float>(tex.x) * sx) * 0.5f;
    const float halfH = (static_cast<float>(tex.y) * sy) * 0.5f;

    const float dxCol = 2.0f * halfW * 0.498269896f;
    const float dyCol = -2.0f * halfH * 0.532818532f;

    const float oddRowXFactor  = 1.0f;
    const float oddRowYFactor  = 0.0f;

    const float evenRowXFactor = 0.498269896f;
    const float evenRowYFactor = 0.525096525f;

    const float dxRowOdd  = 2.0f * halfW * oddRowXFactor;
    const float dyRowOdd  = 2.0f * halfH * oddRowYFactor;

    const float dxRowEven = 2.0f * halfW * evenRowXFactor;
    const float dyRowEven = 2.0f * halfH * evenRowYFactor;

    sf::Vector2f rowStart(0.0f, 0.0f);

    for (int row = 0; row < 7; ++row) {
        sf::Vector2f c = rowStart;

        for (int col = 0; col < 7; ++col) {
            viewer.addTile(c.x, c.y, sx, sy);
            c.x += dxCol;
            c.y += dyCol;
        }

        if (row == 6) break;

        const int nextRow = row + 1;
        if (nextRow % 2 == 1) {
            rowStart.x += dxRowOdd;
            rowStart.y += dyRowOdd;
        } else {
            rowStart.x += dxRowEven;
            rowStart.y += dyRowEven;
        }
    }

    return viewer.run(centerX, centerY);
}

