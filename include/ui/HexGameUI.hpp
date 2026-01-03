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
        const std::string& backgroundPath,
        const std::string& player1Path,
        const std::string& player2Path,
        const std::string& startPagePath,
        const std::string& startButtonPath,
        const std::string& startTitlePath,
        const std::string& player1WinPath,
        const std::string& player2WinPath,
        int boardSize,
        float tileScale,
        bool useGnnAi,
        const std::string& modelPath,
        bool preferCuda);

    int run();

private:
    struct Tile {
        HexTile sprite;
        sf::Vector2f center;
        int index;

        Tile(const sf::Texture& texture, const sf::Vector2f& centerPos, int idx, float scale);
    };

    bool loadTexture();
    bool loadBackgroundTexture();
    bool loadPlayerTextures();
    bool loadStartScreenTextures();
    bool loadVictoryTextures();
    void buildLayout();
    void updateTileColors();
    bool applyMove(int moveIdx);
    int pickTileIndex(const sf::Vector2f& pos) const;
    bool pointInHex(const sf::Vector2f& pos, const sf::Vector2f& center) const;
    void updateWindowTitle(sf::RenderWindow& window) const;
    void updateHover(const sf::RenderWindow& window);
    void printBoardStatus() const;
    void resetGame();

    std::string texturePath_;
    std::string backgroundPath_;
    std::string player1Path_;
    std::string player2Path_;
    std::string startPagePath_;
    std::string startButtonPath_;
    std::string startTitlePath_;
    std::string player1WinPath_;
    std::string player2WinPath_;
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
    int hoveredIndex_ = -1;

    sf::Texture texture_;
    sf::Texture backgroundTexture_;
    sf::Sprite backgroundSprite_;
    sf::Texture player1Texture_;
    sf::Texture player2Texture_;
    sf::Sprite player1Sprite_;
    sf::Sprite player2Sprite_;
    sf::Texture startPageTexture_;
    sf::Texture startButtonTexture_;
    sf::Sprite startPageSprite_;
    sf::Sprite startButtonSprite_;
    sf::Texture startTitleTexture_;
    sf::Sprite startTitleSprite_;
    sf::Font startFont_;
    sf::Text startHintText_;
    sf::RectangleShape startHintBox_;

    //Menu configuration
    sf::RectangleShape aiConfigBox_;
    sf::Text aiConfigText_;
    sf::Text hardwareInfoText_;
    bool isAiButtonHovered_ = false;

    sf::Clock startScreenClock_;
    sf::Vector2f startHintBoxBasePos_{0.0f, 0.0f};
    sf::Vector2f startHintTextBasePos_{0.0f, 0.0f};
    sf::Texture player1WinTexture_;
    sf::Texture player2WinTexture_;
    sf::Sprite player1WinSprite_;
    sf::Sprite player2WinSprite_;
    sf::RectangleShape victoryOverlay_;
    sf::Clock victoryClock_;
    sf::Vector2u textureSize_{0, 0};
    float tileWidth_ = 0.0f;
    float tileHeight_ = 0.0f;
    sf::Vector2u windowSize_{0, 0};

    std::vector<Tile> tiles_;
    std::string error_;
    bool showStartScreen_ = true;
    bool startFontLoaded_ = false;
    bool victoryAnimationActive_ = false;
};
