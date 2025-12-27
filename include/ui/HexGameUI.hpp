#pragma once

#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "core/Board.hpp"
#include "core/Player.hpp"
#include "ui/HexTile.hpp"

class HexGameUI {
public:
    HexGameUI(
        const std::string& texturePath,
        int boardSize,
        float tileScale,
        bool useGnnAi,
        const std::string& modelPath);

    int run();

private:
    struct Tile {
        HexTile sprite;
        sf::Vector2f center;
        int index;

        Tile(const sf::Texture& texture, const sf::Vector2f& centerPos, int idx, float scale);
    };

    bool loadTexture();
    void buildLayout();
    void updateTileColors();
    bool applyMove(int moveIdx);
    int pickTileIndex(const sf::Vector2f& pos) const;
    bool pointInHex(const sf::Vector2f& pos, const sf::Vector2f& center) const;
    void updateWindowTitle(sf::RenderWindow& window) const;
    void printBoardStatus() const;
    void resetGame();

    std::string texturePath_;
    std::string modelPath_;
    int boardSize_ = 0;
    float tileScale_ = 1.0f;
    bool useGnnAi_ = false;

    Board board_;
    AIPlayer heuristicAI_;
    AIPlayer gnnAI_;
    int currentPlayerId_ = 1;
    bool gameOver_ = false;
    int winnerId_ = 0;

    sf::Texture texture_;
    sf::Vector2u textureSize_{0, 0};
    float tileWidth_ = 0.0f;
    float tileHeight_ = 0.0f;
    sf::Vector2u windowSize_{0, 0};

    std::vector<Tile> tiles_;
    std::string error_;
};
