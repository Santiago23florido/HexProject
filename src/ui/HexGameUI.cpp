#include "ui/HexGameUI.hpp"

#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

constexpr float kWindowMargin = 24.0f;
constexpr sf::Uint8 kHoverAlpha = 180;
constexpr float kPlayerIconMargin = 12.0f;
constexpr sf::Uint8 kInactivePlayerAlpha = 90;
constexpr float kStartButtonWidthRatio = 0.25f;
constexpr float kStartTitleWidthRatio = 0.45f;
constexpr float kStartTitleTopMargin = -40.0f;
constexpr float kStartHintBoxWidthRatio = 0.60f;
constexpr float kStartHintBoxHeightRatio = 0.09f;
constexpr float kStartHintTopMargin = 12.0f;
constexpr float kStartHintVibrateAmplitude = 2.0f;
constexpr float kStartHintVibrateSpeed = 18.0f;
constexpr float kVictoryFadeDuration = 0.9f;
constexpr float kVictoryOverlayMaxAlpha = 190.0f;
constexpr float kVictoryImageWidthRatio = 0.55f;
constexpr float kVictoryImageScaleStart = 0.92f;
constexpr float kVictoryImageScaleEnd = 1.0f;

HexGameUI::Tile::Tile(
    const sf::Texture& texture,
    const sf::Vector2f& centerPos,
    int idx,
    float scale)
    : sprite(texture), center(centerPos), index(idx) {
    sprite.setScale(scale);
    sprite.setPosition(center.x, center.y);
}

HexGameUI::HexGameUI(
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
    const std::string& modelPath)
    : texturePath_(texturePath),
      backgroundPath_(backgroundPath),
      player1Path_(player1Path),
      player2Path_(player2Path),
      startPagePath_(startPagePath),
      startButtonPath_(startButtonPath),
      startTitlePath_(startTitlePath),
      player1WinPath_(player1WinPath),
      player2WinPath_(player2WinPath),
      modelPath_(modelPath),
      boardSize_(boardSize),
      tileScale_(tileScale),
      useGnnAi_(useGnnAi),
      board_(boardSize),
      heuristicAI_(2, std::make_unique<NegamaxHeuristicStrategy>(3, 2000)),
      gnnAI_(2, std::make_unique<NegamaxGnnStrategy>(20, 10000, modelPath)) {
    if (!loadTexture()) {
        return;
    }
    if (!loadBackgroundTexture()) {
        return;
    }
    if (!loadPlayerTextures()) {
        return;
    }
    if (!loadStartScreenTextures()) {
        return;
    }
    if (!loadVictoryTextures()) {
        return;
    }

    buildLayout();
    updateTileColors();
}

bool HexGameUI::loadTexture() {
    if (boardSize_ <= 0) {
        error_ = "Board size must be positive.";
        return false;
    }
    if (tileScale_ <= 0.0f) {
        error_ = "Tile scale must be positive.";
        return false;
    }
    if (!texture_.loadFromFile(texturePath_)) {
        error_ = "Failed to load texture: " + texturePath_;
        return false;
    }
    textureSize_ = texture_.getSize();
    if (textureSize_.x == 0 || textureSize_.y == 0) {
        error_ = "Invalid texture size.";
        return false;
    }
    tileWidth_ = static_cast<float>(textureSize_.x) * tileScale_;
    tileHeight_ = static_cast<float>(textureSize_.y) * tileScale_;
    return true;
}

bool HexGameUI::loadBackgroundTexture() {
    if (backgroundPath_.empty()) {
        return true;
    }
    if (!backgroundTexture_.loadFromFile(backgroundPath_)) {
        error_ = "Failed to load background texture: " + backgroundPath_;
        return false;
    }
    const sf::Vector2u backgroundSize = backgroundTexture_.getSize();
    if (backgroundSize.x == 0 || backgroundSize.y == 0) {
        error_ = "Invalid background texture size.";
        return false;
    }
    backgroundSprite_.setTexture(backgroundTexture_);
    return true;
}

bool HexGameUI::loadPlayerTextures() {
    if (player1Path_.empty() || player2Path_.empty()) {
        return true;
    }
    if (!player1Texture_.loadFromFile(player1Path_)) {
        error_ = "Failed to load player 1 texture: " + player1Path_;
        return false;
    }
    if (!player2Texture_.loadFromFile(player2Path_)) {
        error_ = "Failed to load player 2 texture: " + player2Path_;
        return false;
    }
    const sf::Vector2u size1 = player1Texture_.getSize();
    const sf::Vector2u size2 = player2Texture_.getSize();
    if (size1.x == 0 || size1.y == 0 || size2.x == 0 || size2.y == 0) {
        error_ = "Invalid player texture size.";
        return false;
    }
    player1Sprite_.setTexture(player1Texture_);
    player2Sprite_.setTexture(player2Texture_);
    return true;
}

bool HexGameUI::loadStartScreenTextures() {
    if (startPagePath_.empty() || startButtonPath_.empty() || startTitlePath_.empty()) {
        showStartScreen_ = false;
        return true;
    }
    if (!startPageTexture_.loadFromFile(startPagePath_)) {
        error_ = "Failed to load start page texture: " + startPagePath_;
        return false;
    }
    if (!startButtonTexture_.loadFromFile(startButtonPath_)) {
        error_ = "Failed to load start button texture: " + startButtonPath_;
        return false;
    }
    if (!startTitleTexture_.loadFromFile(startTitlePath_)) {
        error_ = "Failed to load start title texture: " + startTitlePath_;
        return false;
    }
    const sf::Vector2u pageSize = startPageTexture_.getSize();
    const sf::Vector2u buttonSize = startButtonTexture_.getSize();
    const sf::Vector2u titleSize = startTitleTexture_.getSize();
    if (pageSize.x == 0 || pageSize.y == 0 ||
        buttonSize.x == 0 || buttonSize.y == 0 ||
        titleSize.x == 0 || titleSize.y == 0) {
        error_ = "Invalid start screen texture size.";
        return false;
    }
    startPageSprite_.setTexture(startPageTexture_);
    startButtonSprite_.setTexture(startButtonTexture_);
    startTitleSprite_.setTexture(startTitleTexture_);

    startFontLoaded_ = startFont_.loadFromFile("../assets/DejaVuSans.ttf");
    if (!startFontLoaded_) {
        startFontLoaded_ =
            startFont_.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    }
    if (!startFontLoaded_) {
        error_ = "Failed to load start screen font.";
        return false;
    }
    startHintText_.setFont(startFont_);
    startHintText_.setString("Press Start to Play");
    startHintText_.setFillColor(sf::Color::White);

    showStartScreen_ = true;
    return true;
}

bool HexGameUI::loadVictoryTextures() {
    if (player1WinPath_.empty() || player2WinPath_.empty()) {
        return true;
    }
    if (!player1WinTexture_.loadFromFile(player1WinPath_)) {
        error_ = "Failed to load player 1 win texture: " + player1WinPath_;
        return false;
    }
    if (!player2WinTexture_.loadFromFile(player2WinPath_)) {
        error_ = "Failed to load player 2 win texture: " + player2WinPath_;
        return false;
    }
    const sf::Vector2u size1 = player1WinTexture_.getSize();
    const sf::Vector2u size2 = player2WinTexture_.getSize();
    if (size1.x == 0 || size1.y == 0 || size2.x == 0 || size2.y == 0) {
        error_ = "Invalid victory texture size.";
        return false;
    }
    player1WinSprite_.setTexture(player1WinTexture_);
    player2WinSprite_.setTexture(player2WinTexture_);
    return true;
}

void HexGameUI::buildLayout() {
    tiles_.clear();

    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();

    const float halfW = tileWidth_ * 0.5f;
    const float halfH = tileHeight_ * 0.5f;

    const float dxCol = 2.0f * halfW * 0.498269896f;
    const float dyCol = -2.0f * halfH * 0.532818532f;

    const float dxRowOdd = 2.0f * halfW * 1.0f;
    const float dyRowOdd = 2.0f * halfH * 0.0f;

    const float dxRowEven = 2.0f * halfW * 0.498269896f;
    const float dyRowEven = 2.0f * halfH * 0.525096525f;

    std::vector<sf::Vector2f> centers;
    centers.reserve(static_cast<std::size_t>(boardSize_ * boardSize_));

    sf::Vector2f rowStart(0.0f, 0.0f);
    for (int row = 0; row < boardSize_; ++row) {
        sf::Vector2f c = rowStart;
        for (int col = 0; col < boardSize_; ++col) {
            centers.emplace_back(c);
            minX = std::min(minX, c.x);
            maxX = std::max(maxX, c.x);
            minY = std::min(minY, c.y);
            maxY = std::max(maxY, c.y);
            c.x += dxCol;
            c.y += dyCol;
        }

        if (row + 1 >= boardSize_) {
            break;
        }
        if ((row + 1) % 2 == 1) {
            rowStart.x += dxRowOdd;
            rowStart.y += dyRowOdd;
        } else {
            rowStart.x += dxRowEven;
            rowStart.y += dyRowEven;
        }
    }

    float boardWidth = (maxX - minX) + tileWidth_;
    float boardHeight = (maxY - minY) + tileHeight_;

    windowSize_ = sf::Vector2u(
        static_cast<unsigned int>(std::ceil(boardWidth + 2.0f * kWindowMargin)),
        static_cast<unsigned int>(std::ceil(boardHeight + 2.0f * kWindowMargin)));

    sf::Vector2f offset(
        kWindowMargin + tileWidth_ / 2.0f - minX,
        kWindowMargin + tileHeight_ / 2.0f - minY + 20.0f);

    tiles_.reserve(static_cast<std::size_t>(boardSize_ * boardSize_));
    for (int row = 0; row < boardSize_; ++row) {
        for (int col = 0; col < boardSize_; ++col) {
            int idx = row * boardSize_ + col;
            const sf::Vector2f& baseCenter = centers[static_cast<std::size_t>(idx)];
            sf::Vector2f center(baseCenter.x + offset.x, baseCenter.y + offset.y);
            tiles_.emplace_back(texture_, center, idx, tileScale_);
        }
    }

    std::sort(tiles_.begin(), tiles_.end(), [](const Tile& a, const Tile& b) {
        if (a.center.y == b.center.y) {
            return a.center.x < b.center.x;
        }
        return a.center.y < b.center.y;
    });
}

void HexGameUI::updateTileColors() {
    const sf::Color emptyColor(210, 210, 220);
    const sf::Color playerXColor(210, 70, 70);
    const sf::Color playerOColor(70, 120, 210);

    for (auto& tile : tiles_) {
        int row = tile.index / boardSize_;
        int col = tile.index % boardSize_;
        int value = board_.board[row][col];
        sf::Color color = emptyColor;
        if (value == 1) {
            color = playerXColor;
        } else if (value == 2) {
            color = playerOColor;
        }
        color.a = (tile.index == hoveredIndex_) ? kHoverAlpha : 255;
        tile.sprite.setColor(color);
    }
}

bool HexGameUI::applyMove(int moveIdx) {
    if (moveIdx < 0 || moveIdx >= boardSize_ * boardSize_) {
        return false;
    }

    if (!board_.place(moveIdx, currentPlayerId_)) {
        return false;
    }

    GameState state(board_, currentPlayerId_);
    int winner = state.Winner();
    if (winner != 0) {
        gameOver_ = true;
        winnerId_ = winner;
        victoryAnimationActive_ = true;
        victoryClock_.restart();
    } else {
        currentPlayerId_ = (currentPlayerId_ == 1) ? 2 : 1;
    }

    updateTileColors();
    printBoardStatus();
    return true;
}

int HexGameUI::pickTileIndex(const sf::Vector2f& pos) const {
    int bestIndex = -1;
    float bestDist2 = std::numeric_limits<float>::max();

    for (const auto& tile : tiles_) {
        if (!pointInHex(pos, tile.center)) {
            continue;
        }
        float dx = pos.x - tile.center.x;
        float dy = pos.y - tile.center.y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < bestDist2) {
            bestDist2 = dist2;
            bestIndex = tile.index;
        }
    }
    return bestIndex;
}

bool HexGameUI::pointInHex(const sf::Vector2f& pos, const sf::Vector2f& center) const {
    float dx = std::fabs(pos.x - center.x);
    float dy = std::fabs(pos.y - center.y);

    float halfW = tileWidth_ / 2.0f;
    float halfH = tileHeight_ / 2.0f;
    if (dx > halfW || dy > halfH) {
        return false;
    }

    float inner = tileWidth_ / 4.0f;
    if (dx <= inner) {
        return true;
    }

    float maxY = tileHeight_ - (2.0f * tileHeight_ / tileWidth_) * dx;
    return dy <= maxY;
}

void HexGameUI::updateWindowTitle(sf::RenderWindow& window) const {
    if (gameOver_) {
        if (winnerId_ == 1) {
            window.setTitle("Hex UI - Winner X");
        } else if (winnerId_ == 2) {
            window.setTitle("Hex UI - Winner O");
        } else {
            window.setTitle("Hex UI - Game Over");
        }
        return;
    }

    if (currentPlayerId_ == 1) {
        window.setTitle("Hex UI - Turn X");
    } else {
        window.setTitle("Hex UI - Turn O");
    }
}

void HexGameUI::updateHover(const sf::RenderWindow& window) {
    sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
    if (pixelPos.x < 0 || pixelPos.y < 0 ||
        pixelPos.x >= static_cast<int>(window.getSize().x) ||
        pixelPos.y >= static_cast<int>(window.getSize().y)) {
        if (hoveredIndex_ != -1) {
            hoveredIndex_ = -1;
            updateTileColors();
        }
        return;
    }

    sf::Vector2f pos = window.mapPixelToCoords(pixelPos);
    int idx = pickTileIndex(pos);
    if (idx != hoveredIndex_) {
        hoveredIndex_ = idx;
        updateTileColors();
    }
}

void HexGameUI::printBoardStatus() const {
    board_.print();
    if (gameOver_) {
        if (winnerId_ == 1) {
            std::cout << "\nPlayer X wins!\n";
        } else if (winnerId_ == 2) {
            std::cout << "\nPlayer O wins!\n";
        } else {
            std::cout << "\nGame over.\n";
        }
        return;
    }
    std::cout << "\nPlayer " << (currentPlayerId_ == 1 ? "X" : "O") << " turn\n";
}

void HexGameUI::resetGame() {
    board_ = Board(boardSize_);
    currentPlayerId_ = 1;
    gameOver_ = false;
    winnerId_ = 0;
    victoryAnimationActive_ = false;
    victoryOverlay_.setFillColor(sf::Color(0, 0, 0, 0));
    updateTileColors();
    printBoardStatus();
}

int HexGameUI::run() {
    if (!error_.empty()) {
        std::cerr << error_ << "\n";
        return 1;
    }
    if (windowSize_.x == 0 || windowSize_.y == 0) {
        std::cerr << "Invalid window size.\n";
        return 1;
    }

    sf::RenderWindow window(
        sf::VideoMode(windowSize_.x, windowSize_.y),
        "Hex UI - Viewer");
    window.setFramerateLimit(60);
    victoryOverlay_.setSize(sf::Vector2f(windowSize_.x, windowSize_.y));
    victoryOverlay_.setPosition(0.0f, 0.0f);
    victoryOverlay_.setFillColor(sf::Color(0, 0, 0, 0));
    if (backgroundTexture_.getSize().x != 0 && backgroundTexture_.getSize().y != 0) {
        const sf::Vector2u backgroundSize = backgroundTexture_.getSize();
        const float scaleX =
            (static_cast<float>(windowSize_.x) / backgroundSize.x) * 1.5f;
        const float scaleY =
            (static_cast<float>(windowSize_.y) / backgroundSize.y) * 1.05f;
        backgroundSprite_.setScale(scaleX, scaleY);
        const float scaledWidth = backgroundSize.x * scaleX;
        const float offsetX = (static_cast<float>(windowSize_.x) - scaledWidth) / 2.0f;
        backgroundSprite_.setPosition(offsetX, 0.0f);
    }
    if (player1Texture_.getSize().x != 0 && player2Texture_.getSize().x != 0) {
        const float desiredHeight = tileHeight_ * 2.0f;
        const float scale1 =
            desiredHeight / static_cast<float>(player1Texture_.getSize().y);
        const float scale2 =
            desiredHeight / static_cast<float>(player2Texture_.getSize().y);
        player1Sprite_.setScale(scale1, scale1);
        player2Sprite_.setScale(scale2, scale2);
        player1Sprite_.setPosition(kPlayerIconMargin, kPlayerIconMargin);
        const float player2Width =
            static_cast<float>(player2Texture_.getSize().x) * scale2;
        player2Sprite_.setPosition(
            static_cast<float>(windowSize_.x) - player2Width - kPlayerIconMargin,
            kPlayerIconMargin);
    }
    if (showStartScreen_) {
        const sf::Vector2u pageSize = startPageTexture_.getSize();
        const sf::Vector2u buttonSize = startButtonTexture_.getSize();
        const sf::Vector2u titleSize = startTitleTexture_.getSize();
        const float pageScaleX = static_cast<float>(windowSize_.x) / pageSize.x;
        const float pageScaleY = static_cast<float>(windowSize_.y) / pageSize.y;
        startPageSprite_.setScale(pageScaleX, pageScaleY);
        startPageSprite_.setPosition(0.0f, 0.0f);
        const float desiredWidth =
            static_cast<float>(windowSize_.x) * kStartButtonWidthRatio;
        const float buttonScale = desiredWidth / buttonSize.x;
        startButtonSprite_.setScale(buttonScale, buttonScale);
        const float scaledButtonWidth = buttonSize.x * buttonScale;
        const float scaledButtonHeight = buttonSize.y * buttonScale;
        const float buttonX =
            (static_cast<float>(windowSize_.x) - scaledButtonWidth) / 2.0f;
        const float buttonY =
            (static_cast<float>(windowSize_.y) - scaledButtonHeight) / 2.0f;
        startButtonSprite_.setPosition(buttonX, buttonY);

        const float desiredTitleWidth =
            static_cast<float>(windowSize_.x) * kStartTitleWidthRatio;
        const float titleScale = desiredTitleWidth / titleSize.x;
        startTitleSprite_.setScale(titleScale, titleScale);
        const float scaledTitleWidth = titleSize.x * titleScale;
        startTitleSprite_.setPosition(
            (static_cast<float>(windowSize_.x) - scaledTitleWidth) / 2.0f,
            kStartTitleTopMargin);

        if (startFontLoaded_) {
            const float hintBoxWidth =
                static_cast<float>(windowSize_.x) * kStartHintBoxWidthRatio;
            const float hintBoxHeight = std::max(
                32.0f, static_cast<float>(windowSize_.y) * kStartHintBoxHeightRatio);
            const float hintBoxX =
                (static_cast<float>(windowSize_.x) - hintBoxWidth) / 2.0f;
            const float hintBoxY = buttonY + scaledButtonHeight + kStartHintTopMargin;
            startHintBox_.setSize(sf::Vector2f(hintBoxWidth, hintBoxHeight));
            startHintBox_.setPosition(hintBoxX, hintBoxY);
            startHintBox_.setFillColor(sf::Color(0, 0, 0, 160));
            startHintBox_.setOutlineColor(sf::Color(255, 255, 255, 200));
            startHintBox_.setOutlineThickness(2.0f);
            startHintBoxBasePos_ = sf::Vector2f(hintBoxX, hintBoxY);

            const unsigned int textSize =
                static_cast<unsigned int>(std::max(12.0f, hintBoxHeight * 0.45f));
            startHintText_.setCharacterSize(textSize);
            const sf::FloatRect textBounds = startHintText_.getLocalBounds();
            startHintText_.setPosition(
                hintBoxX + (hintBoxWidth - textBounds.width) / 2.0f - textBounds.left,
                hintBoxY + (hintBoxHeight - textBounds.height) / 2.0f - textBounds.top);
            startHintTextBasePos_ = startHintText_.getPosition();
        }

        startScreenClock_.restart();
        window.setTitle("Hex UI - Start");
    } else {
        updateWindowTitle(window);
        printBoardStatus();
    }

    while (window.isOpen()) {
        bool humanMovedThisFrame = false;
        sf::Event event;
        if (showStartScreen_ && startFontLoaded_) {
            const float t = startScreenClock_.getElapsedTime().asSeconds();
            const float offsetX =
                std::sin(t * kStartHintVibrateSpeed) * kStartHintVibrateAmplitude;
            const float offsetY =
                std::cos(t * kStartHintVibrateSpeed) * kStartHintVibrateAmplitude;
            startHintBox_.setPosition(
                startHintBoxBasePos_.x + offsetX,
                startHintBoxBasePos_.y + offsetY);
            startHintText_.setPosition(
                startHintTextBasePos_.x + offsetX,
                startHintTextBasePos_.y + offsetY);
        }
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
            if (showStartScreen_) {
                if (event.type == sf::Event::MouseButtonPressed &&
                    event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2f pos = window.mapPixelToCoords(
                        sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
                    if (startButtonSprite_.getGlobalBounds().contains(pos)) {
                        showStartScreen_ = false;
                        updateWindowTitle(window);
                        printBoardStatus();
                    }
                }
                continue;
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::R) {
                resetGame();
                updateWindowTitle(window);
            }
            if (!gameOver_ &&
                event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f pos = window.mapPixelToCoords(
                    sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
                int moveIdx = pickTileIndex(pos);
                if (applyMove(moveIdx)) {
                    updateWindowTitle(window);
                    humanMovedThisFrame = true;
                }
            }
        }

        if (showStartScreen_) {
            window.clear(sf::Color(30, 30, 40));
            if (startPageTexture_.getSize().x != 0 && startPageTexture_.getSize().y != 0) {
                window.draw(startPageSprite_);
            }
            if (startTitleTexture_.getSize().x != 0 && startTitleTexture_.getSize().y != 0) {
                window.draw(startTitleSprite_);
            }
            if (startButtonTexture_.getSize().x != 0 &&
                startButtonTexture_.getSize().y != 0) {
                window.draw(startButtonSprite_);
            }
            if (startFontLoaded_) {
                window.draw(startHintBox_);
                window.draw(startHintText_);
            }
            window.display();
            continue;
        }

        updateHover(window);

        if (!gameOver_ && currentPlayerId_ == 2 && !humanMovedThisFrame) {
            GameState state(board_, currentPlayerId_);
            int moveIdx = useGnnAi_ ? gnnAI_.ChooseMove(state)
                                    : heuristicAI_.ChooseMove(state);
            if (!applyMove(moveIdx)) {
                const auto fallback = state.GetAvailableMoves();
                for (int idx : fallback) {
                    if (applyMove(idx)) {
                        break;
                    }
                }
            }
            updateWindowTitle(window);
        }

        if (player1Texture_.getSize().x != 0 && player2Texture_.getSize().x != 0) {
            const sf::Uint8 player1Alpha =
                (currentPlayerId_ == 1) ? 255 : kInactivePlayerAlpha;
            const sf::Uint8 player2Alpha =
                (currentPlayerId_ == 2) ? 255 : kInactivePlayerAlpha;
            player1Sprite_.setColor(sf::Color(255, 255, 255, player1Alpha));
            player2Sprite_.setColor(sf::Color(255, 255, 255, player2Alpha));
        }

        window.clear(sf::Color(30, 30, 40));
        if (backgroundTexture_.getSize().x != 0 && backgroundTexture_.getSize().y != 0) {
            window.draw(backgroundSprite_);
        }
        for (const auto& tile : tiles_) {
            tile.sprite.draw(window);
        }
        if (player1Texture_.getSize().x != 0 && player2Texture_.getSize().x != 0) {
            window.draw(player1Sprite_);
            window.draw(player2Sprite_);
        }
        if (gameOver_ && (winnerId_ == 1 || winnerId_ == 2) &&
            player1WinTexture_.getSize().x != 0 && player2WinTexture_.getSize().x != 0) {
            float progress = 1.0f;
            if (victoryAnimationActive_) {
                const float t = victoryClock_.getElapsedTime().asSeconds();
                progress = std::min(t / kVictoryFadeDuration, 1.0f);
                if (progress >= 1.0f) {
                    victoryAnimationActive_ = false;
                }
            }

            const sf::Uint8 overlayAlpha = static_cast<sf::Uint8>(
                std::min(kVictoryOverlayMaxAlpha, kVictoryOverlayMaxAlpha * progress));
            victoryOverlay_.setFillColor(sf::Color(0, 0, 0, overlayAlpha));
            window.draw(victoryOverlay_);

            sf::Sprite& winSprite = (winnerId_ == 1) ? player1WinSprite_ : player2WinSprite_;
            const sf::Vector2u winSize = (winnerId_ == 1)
                                             ? player1WinTexture_.getSize()
                                             : player2WinTexture_.getSize();
            const float baseScale =
                (static_cast<float>(windowSize_.x) * kVictoryImageWidthRatio) / winSize.x;
            const float animScale = kVictoryImageScaleStart +
                                    (kVictoryImageScaleEnd - kVictoryImageScaleStart) * progress;
            const float scale = baseScale * animScale;
            winSprite.setScale(scale, scale);
            const float scaledWidth = winSize.x * scale;
            const float scaledHeight = winSize.y * scale;
            winSprite.setPosition(
                (static_cast<float>(windowSize_.x) - scaledWidth) / 2.0f,
                (static_cast<float>(windowSize_.y) - scaledHeight) / 2.0f);
            const sf::Uint8 winAlpha = static_cast<sf::Uint8>(255.0f * progress);
            winSprite.setColor(sf::Color(255, 255, 255, winAlpha));
            window.draw(winSprite);
        }
        window.display();
    }
    return 0;
}
