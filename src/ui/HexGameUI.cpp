#include "ui/HexGameUI.hpp"

#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <torch/torch.h>
#include <cuda_runtime.h>

constexpr float kWindowMargin = 24.0f;
constexpr sf::Uint8 kHoverAlpha = 180;
constexpr float kPlayerIconMargin = 12.0f;
constexpr sf::Uint8 kInactivePlayerAlpha = 90;
constexpr float kStartButtonWidthRatio = 0.25f;
constexpr float kPlayerStartButtonWidthRatio = 0.35f;
constexpr float kPlayerNextButtonWidthRatio = 0.20f;
constexpr float kPlayerHumanLabelWidthRatio = 0.25f;
constexpr float kPlayerSelectButtonGap = 18.0f;
constexpr float kBoardSizeButtonWidthRatio = 0.10f;
constexpr float kBoardSizeButtonGap = 10.0f;
constexpr float kStartTitleWidthRatio = 0.60f;
constexpr float kStartTitleGap = 16.0f;
constexpr float kStartHintBoxWidthRatio = 0.60f;
constexpr float kStartHintBoxHeightRatio = 0.09f;
constexpr float kStartHintTopMargin = 12.0f;
constexpr float kStartHintVibrateAmplitude = 2.0f;
constexpr float kStartHintVibrateSpeed = 18.0f;
constexpr float kStartButtonPulseSpeed = 2.0f;
constexpr float kVictoryFadeDuration = 0.9f;
constexpr float kVictoryOverlayMaxAlpha = 190.0f;
constexpr float kVictoryImageWidthRatio = 0.55f;
constexpr float kVictoryImageScaleStart = 0.92f;
constexpr float kVictoryImageScaleEnd = 1.0f;
constexpr float kMusicVolume = 50.0f;
constexpr float kSfxVolume = 80.0f;

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
    const std::string& playerSelectPagePath,
    const std::string& playerStartButtonPath,
    const std::string& nextTypeButtonPath,
    const std::string& plusNButtonPath,
    const std::string& minusNButtonPath,
    const std::string& humanLabelPath,
    const std::string& player2HumanLabelPath,
    const std::string& player2GnnLabelPath,
    const std::string& player2HeuristicLabelPath,
    const std::string& player1WinPath,
    const std::string& player2WinPath,
    int boardSize,
    float tileScale,
    bool useGnnAi,
    const std::string& modelPath,
    bool preferCuda)
    : texturePath_(texturePath),
      backgroundPath_(backgroundPath),
      player1Path_(player1Path),
      player2Path_(player2Path),
      startPagePath_(startPagePath),
      startButtonPath_(startButtonPath),
      startTitlePath_(startTitlePath),
      playerSelectPagePath_(playerSelectPagePath),
      playerStartButtonPath_(playerStartButtonPath),
      nextTypeButtonPath_(nextTypeButtonPath),
      plusNButtonPath_(plusNButtonPath),
      minusNButtonPath_(minusNButtonPath),
      humanLabelPath_(humanLabelPath),
      player2HumanLabelPath_(player2HumanLabelPath),
      player2GnnLabelPath_(player2GnnLabelPath),
      player2HeuristicLabelPath_(player2HeuristicLabelPath),
      player1WinPath_(player1WinPath),
      player2WinPath_(player2WinPath),
      modelPath_(modelPath),
      boardSize_(boardSize),
      tileScale_(tileScale),
      useGnnAi_(useGnnAi),
      preferCuda_(preferCuda),
      board_(boardSize),
      heuristicAI_(2, std::make_unique<NegamaxHeuristicStrategy>(4, 4000)),
      gnnAI_(2, std::make_unique<NegamaxGnnStrategy>(4, 4000, modelPath, preferCuda)) {
    tileScale_ *= scaleFactor_;
    tileScale_ *= 0.95f;
    baseTileScale_ = tileScale_;
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
    if (!loadPlayerSelectTextures()) {
        return;
    }
    if (!loadVictoryTextures()) {
        return;
    }
    if (!loadPauseTextures()) {
        return;
    }

    buildLayout();
    updateTileColors();
    if (baseWindowSize_.x == 0 || baseWindowSize_.y == 0) {
        baseWindowSize_ = windowSize_;
    }
    aiRng_.seed(static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count() ^
        static_cast<unsigned int>(std::random_device{}())));
    const int baseDifficulty = std::min(5, std::max(1, aiDifficulty_));
    int initialIndex = 0;
    if (player2IsHuman_) {
        initialIndex = 10;
    } else if (useGnnAi_) {
        initialIndex = baseDifficulty - 1;
    } else {
        initialIndex = 5 + (baseDifficulty - 1);
    }
    setPlayer2ModeIndex(initialIndex);
}

HexGameUI::~HexGameUI() {
    menuMusic_.stop();
    gameMusic_.stop();
    gameOverSound_.stop();
    gameClickSound_.stop();
}

void HexGameUI::switchMusic(bool playGameMusic) {
    if (playGameMusic) {
        menuMusic_.stop();
        gameMusic_.play();
    } else {
        gameMusic_.stop();
        menuMusic_.play();
    }
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
        screen_ = UIScreen::Game;
        switchMusic(false);
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

    // Load settings button texture
    if (!settingsButtonTexture_.loadFromFile("../assets/settings_button.png")) {
        error_ = "Failed to load settings button texture.";
        return false;
    }
    settingsButtonSprite_.setTexture(settingsButtonTexture_);

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

    menuBackground_.setSize(sf::Vector2f(400.0f, 500.0f)); 
    menuBackground_.setFillColor(sf::Color(45, 45, 48)); 
    menuBackground_.setOutlineThickness(2.0f);
    menuBackground_.setOutlineColor(sf::Color::Cyan);

    menuOverlay_.setFillColor(sf::Color(0, 0, 0, 170));

    
    aiConfigText_.setFont(startFont_);
    aiConfigText_.setCharacterSize(static_cast<unsigned int>(10 * scaleFactor_));
    aiConfigText_.setFillColor(sf::Color::White);
    
    aiConfigText_.setString(useGnnAi_ ? "AI Mode: GNN (Neural)" : "AI Mode: Heuristic");

    difficultyText_.setFont(startFont_);
    difficultyText_.setCharacterSize(static_cast<unsigned int>(10 * scaleFactor_));
    difficultyText_.setFillColor(sf::Color::White);

    boardSizeText_.setFont(startFont_);
    boardSizeText_.setCharacterSize(static_cast<unsigned int>(10 * scaleFactor_));
    boardSizeText_.setFillColor(sf::Color::White);
    updateBoardSizeText();

    
    hardwareInfoText_.setFont(startFont_);
    hardwareInfoText_.setCharacterSize(static_cast<unsigned int>(10 * scaleFactor_));
    hardwareInfoText_.setFillColor(sf::Color(180, 180, 180));

    std::string gpuName = "No GPU detected";
    bool cudaAvailable = false;

    if (torch::cuda::is_available()) {
        cudaAvailable = true;
        cudaDeviceProp prop;
        
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpuName = prop.name;
        } else {
            gpuName = "Failed to get device name";
        }
    }

    
    if (cudaAvailable) {
        hardwareInfoText_.setString("Hardware: \n" + gpuName);
    } else {
        hardwareInfoText_.setString("Hardware: \nCPU (Heuristic mode)");
    }

    return true;
}

bool HexGameUI::loadPlayerSelectTextures() {
    if (playerSelectPagePath_.empty() || playerStartButtonPath_.empty()) {
        playerSelectEnabled_ = false;
        return true;
    }
    if (!playerSelectPageTexture_.loadFromFile(playerSelectPagePath_)) {
        error_ = "Failed to load player selection page texture: " + playerSelectPagePath_;
        return false;
    }
    if (!playerStartButtonTexture_.loadFromFile(playerStartButtonPath_)) {
        error_ = "Failed to load player start button texture: " + playerStartButtonPath_;
        return false;
    }
    if (!nextTypeButtonPath_.empty()) {
        if (!nextTypeButtonTexture_.loadFromFile(nextTypeButtonPath_)) {
            error_ = "Failed to load next type button texture: " + nextTypeButtonPath_;
            return false;
        }
        const sf::Vector2u nextSize = nextTypeButtonTexture_.getSize();
        if (nextSize.x == 0 || nextSize.y == 0) {
            error_ = "Invalid next type button texture size.";
            return false;
        }
        nextTypeButtonSprite_.setTexture(nextTypeButtonTexture_);
    }
    if (!plusNButtonPath_.empty()) {
        if (!plusNButtonTexture_.loadFromFile(plusNButtonPath_)) {
            error_ = "Failed to load plus N texture: " + plusNButtonPath_;
            return false;
        }
        const sf::Vector2u plusSize = plusNButtonTexture_.getSize();
        if (plusSize.x == 0 || plusSize.y == 0) {
            error_ = "Invalid plus N texture size.";
            return false;
        }
        plusNButtonSprite_.setTexture(plusNButtonTexture_);
    }
    if (!minusNButtonPath_.empty()) {
        if (!minusNButtonTexture_.loadFromFile(minusNButtonPath_)) {
            error_ = "Failed to load minus N texture: " + minusNButtonPath_;
            return false;
        }
        const sf::Vector2u minusSize = minusNButtonTexture_.getSize();
        if (minusSize.x == 0 || minusSize.y == 0) {
            error_ = "Invalid minus N texture size.";
            return false;
        }
        minusNButtonSprite_.setTexture(minusNButtonTexture_);
    }
    if (!humanLabelPath_.empty()) {
        if (!humanLabelTexture_.loadFromFile(humanLabelPath_)) {
            error_ = "Failed to load human label texture: " + humanLabelPath_;
            return false;
        }
        const sf::Vector2u labelSize = humanLabelTexture_.getSize();
        if (labelSize.x == 0 || labelSize.y == 0) {
            error_ = "Invalid human label texture size.";
            return false;
        }
        humanLabelSprite_.setTexture(humanLabelTexture_);
    }
    if (!player2HumanLabelPath_.empty()) {
        if (!player2HumanLabelTexture_.loadFromFile(player2HumanLabelPath_)) {
            error_ = "Failed to load player 2 human label texture: " + player2HumanLabelPath_;
            return false;
        }
        const sf::Vector2u labelSize = player2HumanLabelTexture_.getSize();
        if (labelSize.x == 0 || labelSize.y == 0) {
            error_ = "Invalid player 2 human label texture size.";
            return false;
        }
        player2HumanLabelSprite_.setTexture(player2HumanLabelTexture_);
    }
    if (!player2GnnLabelPath_.empty()) {
        if (!player2GnnLabelTexture_.loadFromFile(player2GnnLabelPath_)) {
            error_ = "Failed to load player 2 GNN label texture: " + player2GnnLabelPath_;
            return false;
        }
        const sf::Vector2u labelSize = player2GnnLabelTexture_.getSize();
        if (labelSize.x == 0 || labelSize.y == 0) {
            error_ = "Invalid player 2 GNN label texture size.";
            return false;
        }
        player2GnnLabelSprite_.setTexture(player2GnnLabelTexture_);
    }
    if (!player2HeuristicLabelPath_.empty()) {
        if (!player2HeuristicLabelTexture_.loadFromFile(player2HeuristicLabelPath_)) {
            error_ = "Failed to load player 2 heuristic label texture: " +
                     player2HeuristicLabelPath_;
            return false;
        }
        const sf::Vector2u labelSize = player2HeuristicLabelTexture_.getSize();
        if (labelSize.x == 0 || labelSize.y == 0) {
            error_ = "Invalid player 2 heuristic label texture size.";
            return false;
        }
        player2HeuristicLabelSprite_.setTexture(player2HeuristicLabelTexture_);
    }
    const sf::Vector2u pageSize = playerSelectPageTexture_.getSize();
    const sf::Vector2u buttonSize = playerStartButtonTexture_.getSize();
    if (pageSize.x == 0 || pageSize.y == 0 ||
        buttonSize.x == 0 || buttonSize.y == 0) {
        error_ = "Invalid player selection texture size.";
        return false;
    }
    playerSelectPageSprite_.setTexture(playerSelectPageTexture_);
    playerStartButtonSprite_.setTexture(playerStartButtonTexture_);
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

    // Load restart/quit buttons (optional)
    if (restartButtonTexture_.getSize().x == 0) {
        // attempt to load -- non-fatal
        restartButtonTexture_.loadFromFile("../assets/restart_button.png");
        restartButtonSprite_.setTexture(restartButtonTexture_);
    }
    if (quitButtonTexture_.getSize().x == 0) {
        quitButtonTexture_.loadFromFile("../assets/quit_button.png");
        quitButtonSprite_.setTexture(quitButtonTexture_);
    }
    return true;
}

bool HexGameUI::loadPauseTextures() {
    if (!pauseButtonTexture_.loadFromFile("../assets/pause_button.png")) {
        error_ = "Failed to load pause button texture.";
        return false;
    }
    if (!pauseMenuTexture_.loadFromFile("../assets/pause_menu.png")) {
        error_ = "Failed to load pause menu texture.";
        return false;
    }
    pauseButtonSprite_.setTexture(pauseButtonTexture_);
    pauseMenuSprite_.setTexture(pauseMenuTexture_);
    return true;
}

bool HexGameUI::initAudio() {
    //clean up any previously loaded audio
    menuMusic_.stop();
    gameMusic_.stop();
    gameOverSound_.stop();
    gameClickSound_.stop();

    //charge new audio files
    if (!menuMusic_.openFromFile("../assets/audio/hexMenu.ogg")) {
        error_ = "Failed to load menu music.";
        return false;
    }
    if (!gameMusic_.openFromFile("../assets/audio/hexGame.ogg")) {
        error_ = "Failed to load game music.";
        return false;
    }
    if (!gameOverBuffer_.loadFromFile("../assets/audio/gameOver.wav")) {
        error_ = "Failed to load game over sound.";
        return false;
    }
    gameOverSound_.setBuffer(gameOverBuffer_);

    if (!clickBuffer_.loadFromFile("../assets/audio/Click.wav")) {
        error_ = "Failed to load game click sound.";
        return false;
    }
    gameClickSound_.setBuffer(clickBuffer_);
    
    
    menuMusic_.setLoop(true); 
    gameMusic_.setLoop(true);
    menuMusic_.setVolume(kMusicVolume); 
    gameMusic_.setVolume(kMusicVolume);
    gameOverSound_.setVolume(kMusicVolume);
    gameClickSound_.setVolume(kSfxVolume);
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

    // Setup pause button in top-right corner
    float buttonSize = std::min(windowSize_.x, windowSize_.y) * 0.24f;
    pauseButtonSprite_.setScale(buttonSize / 1024.0f, buttonSize / 1024.0f);
    pauseButtonSprite_.setPosition(
        windowSize_.x - buttonSize - 15.0f,
        10.0f);

    // Setup pause menu overlay
    pauseMenuOverlay_.setSize(sf::Vector2f(windowSize_.x, windowSize_.y));
    pauseMenuOverlay_.setPosition(0.0f, 0.0f);
    pauseMenuOverlay_.setFillColor(sf::Color(0, 0, 0, 150));

    // Setup pause menu sprite centered
    float menuScale = std::min(windowSize_.x * 0.8f / 1024.0f, windowSize_.y * 0.8f / 1024.0f);
    pauseMenuSprite_.setScale(menuScale, menuScale);
    pauseMenuSprite_.setPosition(
        (windowSize_.x - pauseMenuSprite_.getLocalBounds().width * menuScale) * 0.5f,
        (windowSize_.y - pauseMenuSprite_.getLocalBounds().height * menuScale) * 0.5f);
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
        // Reproduce game over sound if player 2 wins and it's not a local game
        if (winnerId_ == 2 && !player2IsHuman_) {
            gameMusic_.stop();
            gameOverSound_.play();
        }
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
    gamePaused_ = false;
    victoryOverlay_.setFillColor(sf::Color(0, 0, 0, 0));
    aiMoveCount_ = 0;
    updateTileColors();
    printBoardStatus();
}

void HexGameUI::setPlayer2ModeIndex(int index) {
    if (index < 0) index = 0;
    if (index > 10) index = 10;
    player2ModeIndex_ = index;

    if (player2ModeIndex_ == 10) {
        player2IsHuman_ = true;
        useGnnAi_ = true;
        aiDifficulty_ = 1;
    } else if (player2ModeIndex_ < 5) {
        player2IsHuman_ = false;
        useGnnAi_ = true;
        aiDifficulty_ = player2ModeIndex_ + 1;
    } else {
        player2IsHuman_ = false;
        useGnnAi_ = false;
        aiDifficulty_ = player2ModeIndex_ - 4;
    }

    applyAiDifficulty();
}

void HexGameUI::advancePlayer2Mode() {
    int next = player2ModeIndex_ + 1;
    if (next > 10) {
        next = 0;
    }
    setPlayer2ModeIndex(next);
}

void HexGameUI::applyAiDifficulty() {
    switch (aiDifficulty_) {
        case 1:
            aiMaxDepth_ = 1;
            aiRandomEvery_ = 2;
            break;
        case 2:
            aiMaxDepth_ = 1;
            aiRandomEvery_ = 3;
            break;
        case 3:
            aiMaxDepth_ = 2;
            aiRandomEvery_ = 3;
            break;
        case 4:
            aiMaxDepth_ = 4;
            aiRandomEvery_ = 4;
            break;
        case 5:
        default:
            aiDifficulty_ = 5;
            aiMaxDepth_ = 5;
            aiRandomEvery_ = 0;
            break;
    }

    heuristicAI_ = AIPlayer(2, std::make_unique<NegamaxHeuristicStrategy>(aiMaxDepth_, 4000));
    gnnAI_ = AIPlayer(2, std::make_unique<NegamaxGnnStrategy>(aiMaxDepth_, 4000, modelPath_, preferCuda_));
    if (auto* strat = dynamic_cast<NegamaxStrategy*>(gnnAI_.Strategy())) {
        const unsigned int hc = std::thread::hardware_concurrency();
        const int threads = (hc > 1u ? static_cast<int>(hc) : 1);
        strat->setParallelThreads(threads);
    }
    aiMoveCount_ = 0;
    updateDifficultyText();
}

void HexGameUI::updateDifficultyText() {
    if (!startFontLoaded_) return;
    if (player2IsHuman_) {
        aiConfigText_.setString("Player 2: Human");
        difficultyText_.setString("Human");
        difficultyText_.setFillColor(sf::Color(140, 140, 140));
    } else {
        const std::string modeLabel = useGnnAi_ ? "GNN" : "Heuristic";
        aiConfigText_.setString("Player 2: " + modeLabel + " (Level " +
                                std::to_string(aiDifficulty_) + ")");
        difficultyText_.setString("Level: " + std::to_string(aiDifficulty_));
        difficultyText_.setFillColor(sf::Color::White);
    }
}

void HexGameUI::updateBoardSizeText() {
    if (!startFontLoaded_) return;
    boardSizeText_.setString("N: " + std::to_string(boardSize_));
}

void HexGameUI::applyBoardSize(int newSize) {
    if (newSize <= 0 || newSize == boardSize_) {
        return;
    }
    boardSize_ = newSize;

    const float baseScale = (baseTileScale_ > 0.0f) ? baseTileScale_ : tileScale_;
    float nextScale = baseScale;
    if (textureSize_.x > 0 && textureSize_.y > 0 &&
        baseWindowSize_.x > 0 && baseWindowSize_.y > 0) {
        const float tileW = static_cast<float>(textureSize_.x);
        const float tileH = static_cast<float>(textureSize_.y);
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();

        const float halfW = tileW * 0.5f;
        const float halfH = tileH * 0.5f;

        const float dxCol = 2.0f * halfW * 0.498269896f;
        const float dyCol = -2.0f * halfH * 0.532818532f;

        const float dxRowOdd = 2.0f * halfW * 1.0f;
        const float dyRowOdd = 2.0f * halfH * 0.0f;

        const float dxRowEven = 2.0f * halfW * 0.498269896f;
        const float dyRowEven = 2.0f * halfH * 0.525096525f;

        sf::Vector2f rowStart(0.0f, 0.0f);
        for (int row = 0; row < boardSize_; ++row) {
            sf::Vector2f c = rowStart;
            for (int col = 0; col < boardSize_; ++col) {
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

        float boardWidth = (maxX - minX) + tileW;
        float boardHeight = (maxY - minY) + tileH;
        if (boardWidth > 0.0f && boardHeight > 0.0f) {
            const float availW = std::max(
                1.0f, static_cast<float>(baseWindowSize_.x) - 2.0f * kWindowMargin);
            const float availH = std::max(
                1.0f, static_cast<float>(baseWindowSize_.y) - 2.0f * kWindowMargin);
            const float fitScale = std::min(availW / boardWidth, availH / boardHeight);
            nextScale = std::min(baseScale, fitScale);
        }
    }

    if (nextScale <= 0.0f) {
        nextScale = baseScale;
    }
    tileScale_ = std::max(0.01f, nextScale);
    tileWidth_ = static_cast<float>(textureSize_.x) * tileScale_;
    tileHeight_ = static_cast<float>(textureSize_.y) * tileScale_;

    board_ = Board(boardSize_);
    currentPlayerId_ = 1;
    gameOver_ = false;
    winnerId_ = 0;
    victoryAnimationActive_ = false;
    hoveredIndex_ = -1;
    aiMoveCount_ = 0;

    buildLayout();
    if (baseWindowSize_.x > 0 && baseWindowSize_.y > 0) {
        windowSize_ = baseWindowSize_;
    }
    updateTileColors();
    updateBoardSizeText();
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
    window.setVerticalSyncEnabled(false);
    if (!initAudio()) {
            std::cerr << "Failed to load music" << std::endl;
    }
    else switchMusic(false);
    
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
    if (screen_ == UIScreen::Start) {
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
        const float scaledTitleHeight = titleSize.y * titleScale;
        const float titleX =
            (static_cast<float>(windowSize_.x) - scaledTitleWidth) / 2.0f;
        const float titleY = buttonY - scaledTitleHeight/2.0f - kStartTitleGap*2;
        startTitleSprite_.setPosition(titleX, titleY);

        
        float margin = 10.0f * scaleFactor_;
        float settingsBtnW = 50.0f * scaleFactor_;
        float settingsBtnH = 20.0f * scaleFactor_;
        float settingsBtnX = windowSize_.x - settingsBtnW - margin;
        float settingsBtnY = margin;

        
        float menuW = windowSize_.x * 0.6f;
        float menuH = windowSize_.y * 0.6f;
        float menuX = (windowSize_.x - menuW) / 2.0f;
        float menuY = (windowSize_.y - menuH) / 2.0f;
        menuBackground_.setSize(sf::Vector2f(menuW, menuH));
        menuBackground_.setPosition(menuX, menuY);
        menuOverlay_.setSize(sf::Vector2f(windowSize_.x, windowSize_.y)); 

        
        aiConfigText_.setPosition(menuX + margin, menuY + margin);

        
        hardwareInfoText_.setPosition(menuX + margin, menuY + 2*margin );

        // Settings button is positioned at top-right corner
        float settingsBtnSize = std::min(windowSize_.x, windowSize_.y) * 0.10f;  // 20% smaller
        float settingsBtnScale = settingsBtnSize / 329.0f;  // Use height (329px) for scaling
        settingsButtonSprite_.setScale(settingsBtnScale, settingsBtnScale);
        settingsButtonSprite_.setPosition(
            windowSize_.x - (1024.0f * settingsBtnScale) - 15.0f,
            15.0f);  // Top-right corner
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
    } else if (screen_ == UIScreen::PlayerSelect) {
        window.setTitle("Hex UI - Player Select");
    } else {
        updateWindowTitle(window);
        printBoardStatus();
    }

    while (window.isOpen()) {
        sf::sleep(sf::milliseconds(1));
        bool humanMovedThisFrame = false;
        sf::Event event;
        if (screen_ == UIScreen::Start) {
            const float t = startScreenClock_.getElapsedTime().asSeconds();
            if (startFontLoaded_) {
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
            const sf::Vector2u buttonSize = startButtonTexture_.getSize();
            if (buttonSize.x != 0 && buttonSize.y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kStartButtonWidthRatio;
                const float baseScale = desiredWidth / buttonSize.x;
                const float pulse = 1.5f + 0.25f * std::sin(t * kStartButtonPulseSpeed);
                const float animatedScale = baseScale * pulse;
                startButtonSprite_.setScale(animatedScale, animatedScale);
                const float scaledButtonWidth = buttonSize.x * animatedScale;
                const float scaledButtonHeight = buttonSize.y * animatedScale;
                const float buttonX =
                    (static_cast<float>(windowSize_.x) - scaledButtonWidth) / 2.0f;
                const float buttonY =
                    (static_cast<float>(windowSize_.y) - scaledButtonHeight) / 2.0f;
                startButtonSprite_.setPosition(buttonX, buttonY);
            }
        } else if (screen_ == UIScreen::PlayerSelect) {
            float startButtonY = 0.0f;
            float startButtonHeight = 0.0f;
            const sf::Vector2u buttonSize = playerStartButtonTexture_.getSize();
            if (buttonSize.x != 0 && buttonSize.y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kPlayerStartButtonWidthRatio;
                const float buttonScale = desiredWidth / buttonSize.x;
                playerStartButtonSprite_.setScale(buttonScale, buttonScale);
                const float scaledButtonWidth = buttonSize.x * buttonScale;
                const float scaledButtonHeight = buttonSize.y * buttonScale;
                const float buttonX =
                    (static_cast<float>(windowSize_.x) - scaledButtonWidth) / 2.0f;
                const float buttonY =
                    (static_cast<float>(windowSize_.y*3.5f/2.0f) - scaledButtonHeight) / 2.0f;
                playerStartButtonSprite_.setPosition(buttonX, buttonY);
                startButtonY = buttonY;
                startButtonHeight = scaledButtonHeight;
            }
            const sf::Vector2u nextSize = nextTypeButtonTexture_.getSize();
            if (nextSize.x != 0 && nextSize.y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kPlayerNextButtonWidthRatio;
                const float nextScale = desiredWidth / nextSize.x;
                nextTypeButtonSprite_.setScale(nextScale, nextScale);
                const float scaledWidth = nextSize.x * nextScale;
                const float scaledHeight = nextSize.y * nextScale;
                const float buttonX =
                    (static_cast<float>(windowSize_.x) * 3.0f / 4.0f) - (scaledWidth / 2.0f);;
                const float buttonY =
                    (static_cast<float>(windowSize_.y) * 2.5f / 3.0f) - (scaledHeight / 2.0f);
                nextTypeButtonSprite_.setPosition(buttonX, buttonY);
            }
            const sf::Vector2u labelSize = humanLabelTexture_.getSize();
            if (labelSize.x != 0 && labelSize.y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kPlayerHumanLabelWidthRatio;
                const float labelScale = desiredWidth / labelSize.x;
                humanLabelSprite_.setScale(labelScale, labelScale);
                const float scaledWidth = labelSize.x * labelScale;
                const float scaledHeight = labelSize.y * labelScale;
                const float labelX =
                    (static_cast<float>(windowSize_.x) * 0.25f) - (scaledWidth / 2.0f);
                const float labelY =
                    (static_cast<float>(windowSize_.y) * 0.5f) - (scaledHeight / 2.0f);
                humanLabelSprite_.setPosition(labelX, labelY);
            }
            const sf::Texture* player2LabelTexture = nullptr;
            sf::Sprite* player2LabelSprite = nullptr;
            if (player2IsHuman_) {
                player2LabelTexture = &player2HumanLabelTexture_;
                player2LabelSprite = &player2HumanLabelSprite_;
            } else if (useGnnAi_) {
                player2LabelTexture = &player2GnnLabelTexture_;
                player2LabelSprite = &player2GnnLabelSprite_;
            } else {
                player2LabelTexture = &player2HeuristicLabelTexture_;
                player2LabelSprite = &player2HeuristicLabelSprite_;
            }
            if (player2LabelTexture && player2LabelSprite) {
                const sf::Vector2u player2Size = player2LabelTexture->getSize();
                if (player2Size.x != 0 && player2Size.y != 0) {
                    const float desiredWidth =
                        static_cast<float>(windowSize_.x) * kPlayerHumanLabelWidthRatio;
                    const float labelScale = desiredWidth / player2Size.x;
                    player2LabelSprite->setScale(labelScale, labelScale);
                    const float scaledWidth = player2Size.x * labelScale;
                    const float scaledHeight = player2Size.y * labelScale;
                    const float labelX =
                        (static_cast<float>(windowSize_.x) * 0.75f) - (scaledWidth / 2.0f);
                    const float labelY =
                        (static_cast<float>(windowSize_.y) * 0.5f) - (scaledHeight / 2.0f);
                    player2LabelSprite->setPosition(labelX, labelY);
                }
            }

            if (startFontLoaded_) {
                sf::FloatRect anchor(static_cast<float>(windowSize_.x) * 0.75f,
                                     static_cast<float>(windowSize_.y) * 0.5f,
                                     0.0f,
                                     0.0f);
                if (player2LabelSprite) {
                    anchor = player2LabelSprite->getGlobalBounds();
                }
                const sf::FloatRect textBounds = difficultyText_.getLocalBounds();
                difficultyText_.setPosition(
                    anchor.left + (anchor.width - textBounds.width) / 2.0f - textBounds.left,
                    anchor.top + anchor.height + 6.0f * scaleFactor_ - textBounds.top);
            }

            const float centerX = static_cast<float>(windowSize_.x) * 0.5f;
            const float centerY = static_cast<float>(windowSize_.y) * 0.5f;
            const float gap = kBoardSizeButtonGap * scaleFactor_;
            float textHalf = 0.0f;
            if (startFontLoaded_) {
                const sf::FloatRect textBounds = boardSizeText_.getLocalBounds();
                textHalf = textBounds.height * 0.5f;
                boardSizeText_.setPosition(
                    centerX - textBounds.width * 0.5f - textBounds.left,
                    centerY - textBounds.height * 0.5f - textBounds.top);
            }
            if (plusNButtonTexture_.getSize().x != 0 && plusNButtonTexture_.getSize().y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kBoardSizeButtonWidthRatio;
                const float scale =
                    desiredWidth / static_cast<float>(plusNButtonTexture_.getSize().x);
                const float buttonScale = scale * 0.6f;
                const float gapScaled = gap * 0.35f;
                plusNButtonSprite_.setScale(buttonScale, buttonScale);
                const float scaledWidth = plusNButtonTexture_.getSize().x * buttonScale;
                const float scaledHeight = plusNButtonTexture_.getSize().y * buttonScale;
                const float plusY = centerY - textHalf - gapScaled - scaledHeight;
                plusNButtonSprite_.setPosition(centerX - scaledWidth * 0.5f, plusY);
            }
            if (minusNButtonTexture_.getSize().x != 0 && minusNButtonTexture_.getSize().y != 0) {
                const float desiredWidth =
                    static_cast<float>(windowSize_.x) * kBoardSizeButtonWidthRatio;
                const float scale =
                    desiredWidth / static_cast<float>(minusNButtonTexture_.getSize().x);
                const float buttonScale = scale * 0.6f;
                const float gapScaled = gap * 0.35f;
                minusNButtonSprite_.setScale(buttonScale, buttonScale);
                const float scaledWidth = minusNButtonTexture_.getSize().x * buttonScale;
                const float minusY = centerY + textHalf + gapScaled;
                minusNButtonSprite_.setPosition(centerX - scaledWidth * 0.5f, minusY);
            }
        }
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }

            if (screen_ == UIScreen::Start) {
                if (event.type == sf::Event::MouseButtonPressed &&
                    event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2f mousePos = window.mapPixelToCoords(
                        sf::Vector2i(event.mouseButton.x, event.mouseButton.y));

                    if (!showSettingsMenu_) {
                        if (startButtonSprite_.getGlobalBounds().contains(mousePos)) {
                            gameClickSound_.play();
                            showSettingsMenu_ = false;
                            if (playerSelectEnabled_) {
                                screen_ = UIScreen::PlayerSelect;
                                window.setTitle("Hex UI - Player Select");
                            } else {
                                screen_ = UIScreen::Game;
                                updateWindowTitle(window);
                                printBoardStatus();
                                switchMusic(true);
                            }
                        }

                        if (settingsButtonSprite_.getGlobalBounds().contains(mousePos)) {
                            showSettingsMenu_ = true;
                        }
                    } else {
                        
                        
                        if (aiConfigBox_.getGlobalBounds().contains(mousePos)) {
                            advancePlayer2Mode();
                            std::cout << aiConfigText_.getString().toAnsiString() << "\n";
                        }

                        
                        if (!menuBackground_.getGlobalBounds().contains(mousePos)) {
                            showSettingsMenu_ = false;
                        }
                    }
                }
                continue; 
            }

            if (screen_ == UIScreen::PlayerSelect) {
                if (event.type == sf::Event::MouseButtonPressed &&
                    event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2f mousePos = window.mapPixelToCoords(
                        sf::Vector2i(event.mouseButton.x, event.mouseButton.y));

                    if (plusNButtonTexture_.getSize().x != 0 &&
                        plusNButtonTexture_.getSize().y != 0 &&
                        plusNButtonSprite_.getGlobalBounds().contains(mousePos)) {
                        applyBoardSize(boardSize_ + 1);
                        if (startFontLoaded_) {
                            std::cout << boardSizeText_.getString().toAnsiString() << "\n";
                        }
                    } else if (minusNButtonTexture_.getSize().x != 0 &&
                        minusNButtonTexture_.getSize().y != 0 &&
                        minusNButtonSprite_.getGlobalBounds().contains(mousePos)) {
                        applyBoardSize(boardSize_ - 1);
                        if (startFontLoaded_) {
                            std::cout << boardSizeText_.getString().toAnsiString() << "\n";
                        }
                    } else if (nextTypeButtonTexture_.getSize().x != 0 &&
                        nextTypeButtonTexture_.getSize().y != 0 &&
                        nextTypeButtonSprite_.getGlobalBounds().contains(mousePos)) {
                        advancePlayer2Mode();
                        std::cout << aiConfigText_.getString().toAnsiString() << "\n";
                        gameClickSound_.stop();
                        gameClickSound_.play();
                        if (player2IsHuman_) {
                            player2IsHuman_ = false;
                            useGnnAi_ = true;
                        } else if (useGnnAi_) {
                            useGnnAi_ = false;
                        } else {
                            player2IsHuman_ = true;
                        }
                        if (player2IsHuman_) {
                            aiConfigText_.setString("Jugador 2: Humano");
                            std::cout << "Jugador 2: Humano\n";
                        } else {
                            aiConfigText_.setString(useGnnAi_ ? "Modo IA: GNN" : "Modo IA: Heuristico");
                            std::cout << "Jugador 2: IA (" << (useGnnAi_ ? "GNN" : "Heuristico")
                                      << ")\n";
                        }
                    } else if (playerStartButtonSprite_.getGlobalBounds().contains(mousePos)) {
                        gameClickSound_.play();
                        screen_ = UIScreen::Game;
                        updateWindowTitle(window);
                        printBoardStatus();
                        aiMoveCount_ = 0;

                        
                        switchMusic(true);
                    }
                }
                continue;
            }

            
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::R) {
                resetGame();
                updateWindowTitle(window);
            }

            // Handle pause button click
            if (screen_ == UIScreen::Game &&
                event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f pos = window.mapPixelToCoords(
                    sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
                if (pauseButtonSprite_.getGlobalBounds().contains(pos)) {
                    gameClickSound_.play();
                    gamePaused_ = !gamePaused_;
                }
            }

            // Handle victory buttons clicks (restart / quit)
            if (screen_ == UIScreen::Game && gameOver_ &&
                event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f pos = window.mapPixelToCoords(
                    sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
                if (restartButtonTexture_.getSize().x != 0 &&
                    restartButtonSprite_.getGlobalBounds().contains(pos)) {
                    gameClickSound_.play();
                    // Restart keeping player/AI settings
                    resetGame();
                    updateWindowTitle(window);
                    switchMusic(true); // start game music
                } else if (quitButtonTexture_.getSize().x != 0 &&
                           quitButtonSprite_.getGlobalBounds().contains(pos)) {
                    gameClickSound_.play();
                    // Go back to start screen
                    resetGame();
                    screen_ = UIScreen::Start;
                    showSettingsMenu_ = false;
                    switchMusic(false); // play menu music
                }
            }

            if (!gameOver_ && !gamePaused_ &&
                event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f pos = window.mapPixelToCoords(
                    sf::Vector2i(event.mouseButton.x, event.mouseButton.y));
                int moveIdx = pickTileIndex(pos);
                if (applyMove(moveIdx)) {
                    gameClickSound_.play();
                    updateWindowTitle(window);
                    humanMovedThisFrame = true;
                }
            }
        }
        sf::sleep(sf::milliseconds(1));

        if (screen_ == UIScreen::Start) {
            window.clear(sf::Color(30, 30, 40));

            if (startPageTexture_.getSize().x != 0) {
                window.draw(startPageSprite_);
            }

            if (startTitleTexture_.getSize().x != 0) {
                window.draw(startTitleSprite_);
            }

            if (startButtonTexture_.getSize().x != 0) {
                window.draw(startButtonSprite_);
            }

            if (settingsButtonTexture_.getSize().x != 0) {
                window.draw(settingsButtonSprite_);
            }

            if (showSettingsMenu_) {
                window.draw(menuOverlay_);    
                window.draw(menuBackground_); 
                
                if (startFontLoaded_) {
                    window.draw(aiConfigBox_);
                    window.draw(aiConfigText_);
                    window.draw(hardwareInfoText_);
                }
            }

            window.display();
            continue;
        }

        if (screen_ == UIScreen::PlayerSelect) {
            window.clear(sf::Color(30, 30, 40));

            if (playerSelectPageTexture_.getSize().x != 0) {
                const sf::Vector2u pageSize = playerSelectPageTexture_.getSize();
                const float pageScaleX = static_cast<float>(windowSize_.x) / pageSize.x;
                const float pageScaleY = static_cast<float>(windowSize_.y) / pageSize.y;
                playerSelectPageSprite_.setScale(pageScaleX, pageScaleY);
                playerSelectPageSprite_.setPosition(0.0f, 0.0f);
                window.draw(playerSelectPageSprite_);
            }

            if (humanLabelTexture_.getSize().x != 0 && humanLabelTexture_.getSize().y != 0) {
                window.draw(humanLabelSprite_);
            }

            const sf::Sprite* player2LabelSprite = nullptr;
            if (player2IsHuman_) {
                if (player2HumanLabelTexture_.getSize().x != 0 &&
                    player2HumanLabelTexture_.getSize().y != 0) {
                    player2LabelSprite = &player2HumanLabelSprite_;
                }
            } else if (useGnnAi_) {
                if (player2GnnLabelTexture_.getSize().x != 0 &&
                    player2GnnLabelTexture_.getSize().y != 0) {
                    player2LabelSprite = &player2GnnLabelSprite_;
                }
            } else {
                if (player2HeuristicLabelTexture_.getSize().x != 0 &&
                    player2HeuristicLabelTexture_.getSize().y != 0) {
                    player2LabelSprite = &player2HeuristicLabelSprite_;
                }
            }
            if (player2LabelSprite) {
                window.draw(*player2LabelSprite);
            }

            if (plusNButtonTexture_.getSize().x != 0 && plusNButtonTexture_.getSize().y != 0) {
                window.draw(plusNButtonSprite_);
            }
            if (startFontLoaded_) {
                window.draw(boardSizeText_);
            }
            if (minusNButtonTexture_.getSize().x != 0 && minusNButtonTexture_.getSize().y != 0) {
                window.draw(minusNButtonSprite_);
            }

            if (nextTypeButtonTexture_.getSize().x != 0 && nextTypeButtonTexture_.getSize().y != 0) {
                window.draw(nextTypeButtonSprite_);
            }

            if (startFontLoaded_) {
                window.draw(difficultyText_);
            }

            if (playerStartButtonTexture_.getSize().x != 0) {
                window.draw(playerStartButtonSprite_);
            }

            window.display();

            continue;
        }


        updateHover(window);

        if (!gameOver_ && currentPlayerId_ == 2 && !humanMovedThisFrame && !player2IsHuman_) {
            GameState state(board_, currentPlayerId_);
            int moveIdx = -1;
            const bool useRandom =
                (aiRandomEvery_ > 0) && (((aiMoveCount_ + 1) % aiRandomEvery_) == 0);
            if (useRandom) {
                const auto moves = state.GetAvailableMoves();
                if (!moves.empty()) {
                    std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
                    moveIdx = moves[dist(aiRng_)];
                }
            }
            if (moveIdx < 0) {
                moveIdx = useGnnAi_ ? gnnAI_.ChooseMove(state)
                                    : heuristicAI_.ChooseMove(state);
            }
            aiMoveCount_++;
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
            // Draw restart and quit buttons at bottom-left and bottom-right of the win sprite
            if (restartButtonTexture_.getSize().x != 0 && quitButtonTexture_.getSize().x != 0) {
                const sf::Vector2u btnOrigSize = restartButtonTexture_.getSize();
                // desired width relative to win sprite (25% of win width)
                const float desiredBtnWidth = scaledWidth * 0.25f;
                const float btnScale = desiredBtnWidth / static_cast<float>(btnOrigSize.x);
                const float btnW = btnOrigSize.x * btnScale;
                const float btnH = btnOrigSize.y * btnScale;
                const float padding = 16.0f;
                const float winX = (static_cast<float>(windowSize_.x) - scaledWidth) / 2.0f;
                const float winY = (static_cast<float>(windowSize_.y) - scaledHeight) / 2.0f;

                // Left (restart)
                restartButtonSprite_.setScale(btnScale, btnScale);
                restartButtonSprite_.setPosition(winX + padding, winY + scaledHeight - btnH - padding);
                restartButtonSprite_.setColor(sf::Color(255, 255, 255, winAlpha));
                window.draw(restartButtonSprite_);

                // Right (quit)
                quitButtonSprite_.setScale(btnScale, btnScale);
                quitButtonSprite_.setPosition(winX + scaledWidth - btnW - padding, winY + scaledHeight - btnH - padding);
                quitButtonSprite_.setColor(sf::Color(255, 255, 255, winAlpha));
                window.draw(quitButtonSprite_);
            }
        }

        // Draw pause button
        if (pauseButtonTexture_.getSize().x != 0) {
            window.draw(pauseButtonSprite_);
        }

        // Draw pause menu if paused
        if (gamePaused_) {
            window.draw(pauseMenuOverlay_);
            if (pauseMenuTexture_.getSize().x != 0) {
                window.draw(pauseMenuSprite_);
            }
        }

        window.display();
    }
    return 0;
}
