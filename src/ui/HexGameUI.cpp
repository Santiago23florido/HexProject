#include "ui/HexGameUI.hpp"

#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sys/stat.h>
#include <thread>
#include <torch/torch.h>
#if defined(HEX_HAS_CUDA)
#include <cuda_runtime.h>
#endif

// Implements the SFML UI flow, assets, and interactions for Hex gameplay.

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
constexpr float kHelpFrameDelaySeconds = 1.0f;
constexpr float kHelpFrameDelayLastSeconds = 2.0f;
const std::string kConfigFilePath = "../config/volume_config.txt";

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
    loadVolumeConfig();  // Load saved volume settings
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
    if (menuMusic_.getStatus() == sf::Music::Playing) {
        menuMusic_.stop();
    }
    if (gameMusic_.getStatus() == sf::Music::Playing) {
        gameMusic_.stop();
    }
    if (gameOverSound_.getStatus() == sf::Sound::Playing) {
        gameOverSound_.stop();
    }
    gameOverSound_.setBuffer(sf::SoundBuffer());
    if (gameClickSound_.getStatus() == sf::Sound::Playing) {
        gameClickSound_.stop();
    }
    gameClickSound_.setBuffer(sf::SoundBuffer());
}

void HexGameUI::switchMusic(bool playGameMusic) {
    if (playGameMusic) {
        if (menuMusic_.getStatus() == sf::Music::Playing) {
            menuMusic_.stop();
        }
        sf::sleep(sf::milliseconds(10));  // Small delay before playing
        if (gameMusic_.getStatus() != sf::Music::Playing) {
            gameMusic_.play();
        }
    } else {
        if (gameMusic_.getStatus() == sf::Music::Playing) {
            gameMusic_.stop();
        }
        sf::sleep(sf::milliseconds(10));  // Small delay before playing
        if (menuMusic_.getStatus() != sf::Music::Playing) {
            menuMusic_.play();
        }
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
    
    // Load settings menu texture and buttons
    if (!settingsMenuTexture_.loadFromFile("../assets/settings_menu_short_hd.png")) {
        error_ = "Failed to load settings menu texture.";
        return false;
    }
    settingsMenuSprite_.setTexture(settingsMenuTexture_);
    
    if (!videoButtonTexture_.loadFromFile("../assets/video_button.png")) {
        error_ = "Failed to load video button texture.";
        return false;
    }
    videoButtonSprite_.setTexture(videoButtonTexture_);
    
    if (!audioButtonTexture_.loadFromFile("../assets/audio_button.png")) {
        error_ = "Failed to load audio button texture.";
        return false;
    }
    audioButtonSprite_.setTexture(audioButtonTexture_);
    
    if (!creditsButtonTexture_.loadFromFile("../assets/credits_button.png")) {
        error_ = "Failed to load credits button texture.";
        return false;
    }
    creditsButtonSprite_.setTexture(creditsButtonTexture_); 
    
    if (!settingsBackButtonTexture_.loadFromFile("../assets/back_button.png")) {
        error_ = "Failed to load settings back button texture.";
        return false;
    }
    settingsBackButtonSprite_.setTexture(settingsBackButtonTexture_);
    
    // Load video menu texture
    if (!videoMenuTexture_.loadFromFile("../assets/video_menu_hd.png")) {
        error_ = "Failed to load video menu texture.";
        return false;
    }
    videoMenuSprite_.setTexture(videoMenuTexture_);
    
    // Load audio menu texture
    if (!audioMenuTexture_.loadFromFile("../assets/audio_menu_hd.png")) {
        error_ = "Failed to load audio menu texture.";
        return false;
    }
    audioMenuSprite_.setTexture(audioMenuTexture_);
    
    // Load credits menu texture
    if (!creditsMenuTexture_.loadFromFile("../assets/credits_menu.png")) {
        error_ = "Failed to load credits menu texture.";
        return false;
    }
    creditsMenuSprite_.setTexture(creditsMenuTexture_);
    
    // Load credits image (530x642)
    if (!creditsImageTexture_.loadFromFile("../assets/creditos_530x642.png")) {
        error_ = "Failed to load credits image texture.";
        return false;
    }
    creditsImageSprite_.setTexture(creditsImageTexture_);
    
    // Load volume icon textures (vol0.png to vol3.png)
    volumeIconTextures_.resize(4);
    for (int i = 0; i < 4; ++i) {
        std::string volumePath = "../assets/vol" + std::to_string(i) + ".png";
        if (!volumeIconTextures_[i].loadFromFile(volumePath)) {
            error_ = "Failed to load volume icon " + std::to_string(i) + " texture.";
            return false;
        }
    }
    // Initialize volume icon sprites
    volumeIconSprites_.resize(3);
    for (int i = 0; i < 3; ++i) {
        volumeIconSprites_[i].setTexture(volumeIconTextures_[3]);  // Start with vol3
    }
    
    // Load slider handle texture
    if (!sliderHandleTexture_.loadFromFile("../assets/circle_white.png")) {
        error_ = "Failed to load slider handle texture.";
        return false;
    }
    
    // Load volume label textures
    if (!masterVolumeLabelTexture_.loadFromFile("../assets/master_volume.png")) {
        error_ = "Failed to load master volume label texture.";
        return false;
    }
    masterVolumeLabelSprite_.setTexture(masterVolumeLabelTexture_);
    
    if (!musicVolumeLabelTexture_.loadFromFile("../assets/music_volume.png")) {
        error_ = "Failed to load music volume label texture.";
        return false;
    }
    musicVolumeLabelSprite_.setTexture(musicVolumeLabelTexture_);
    
    if (!effectsVolumeLabelTexture_.loadFromFile("../assets/effects_volume.png")) {
        error_ = "Failed to load effects volume label texture.";
        return false;
    }
    effectsVolumeLabelSprite_.setTexture(effectsVolumeLabelTexture_);
    
    // Load submenu back button texture (same as main back button)
    if (!submenuBackButtonTexture_.loadFromFile("../assets/back_button.png")) {
        error_ = "Failed to load submenu back button texture.";
        return false;
    }
    submenuBackButtonSprite_.setTexture(submenuBackButtonTexture_);

    // Load video menu quality selector textures
    if (!qualityLabelTexture_.loadFromFile("../assets/quality_289x37.png")) {
        error_ = "Failed to load quality label texture.";
        return false;
    }
    qualityLabelSprite_.setTexture(qualityLabelTexture_);

    if (!leftButtonTexture_.loadFromFile("../assets/left.png")) {
        error_ = "Failed to load left button texture.";
        return false;
    }
    leftButtonSprite_.setTexture(leftButtonTexture_);
    
    if (!rightButtonTexture_.loadFromFile("../assets/right.png")) {
        error_ = "Failed to load right button texture.";
        return false;
    }
    rightButtonSprite_.setTexture(rightButtonTexture_);
    
    // Load quality display textures: low, medium, high
    const std::array<std::string, 3> qualityTextures = {
        "../assets/low_145x37.png",
        "../assets/medium_278x37.png",
        "../assets/high_171x37.png"
    };
    for (int i = 0; i < 3; ++i) {
        if (!qualityDisplayTextures_[i].loadFromFile(qualityTextures[i])) {
            error_ = "Failed to load quality display texture: " + qualityTextures[i];
            return false;
        }
    }
    qualityDisplaySprite_.setTexture(qualityDisplayTextures_[static_cast<int>(currentVideoQuality_)]);

    // Load fullscreen label
    if (!fullscreenLabelTexture_.loadFromFile("../assets/fullscreen_413x37.png")) {
        error_ = "Failed to load fullscreen label texture.";
        return false;
    }
    fullscreenLabelSprite_.setTexture(fullscreenLabelTexture_);

    // Load fullscreen display textures: disabled, enabled
    const std::array<std::string, 2> fullscreenTextures = {
        "../assets/disabled_332x37.png",
        "../assets/enabled_305x37.png"
    };
    for (int i = 0; i < 2; ++i) {
        if (!fullscreenDisplayTextures_[i].loadFromFile(fullscreenTextures[i])) {
            error_ = "Failed to load fullscreen display texture: " + fullscreenTextures[i];
            return false;
        }
    }
    fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[fullscreenEnabled_ ? 1 : 0]);

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

    /*
    // Hardware info (commented for future use)
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
    */

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
    if (!boardSizeLabelTexture_.loadFromFile("../assets/N.png")) {
        error_ = "Failed to load board size label texture.";
        return false;
    }
    const sf::Vector2u labelSize = boardSizeLabelTexture_.getSize();
    if (labelSize.x == 0 || labelSize.y == 0) {
        error_ = "Invalid board size label texture size.";
        return false;
    }
    boardSizeLabelSprite_.setTexture(boardSizeLabelTexture_);
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
    
    // Load pause menu buttons
    if (!resumeButtonTexture_.loadFromFile("../assets/resume_button.png")) {
        error_ = "Failed to load resume button texture.";
        return false;
    }
    resumeButtonSprite_.setTexture(resumeButtonTexture_);
    
    if (!restartButtonTexture_.loadFromFile("../assets/restart_button.png")) {
        error_ = "Failed to load restart button texture.";
        return false;
    }
    restartButtonSprite_.setTexture(restartButtonTexture_);
    
    if (!helpButtonTexture_.loadFromFile("../assets/help_button.png")) {
        error_ = "Failed to load help button texture.";
        return false;
    }
    helpButtonSprite_.setTexture(helpButtonTexture_);
    helpSelectButtonSprite_.setTexture(helpButtonTexture_);

    struct HelpFrameEntry {
        int index;
        std::filesystem::path path;
    };

    std::vector<HelpFrameEntry> helpFrames;
    const std::filesystem::path helpDir("../assets/howtoplay");
    if (std::filesystem::exists(helpDir) && std::filesystem::is_directory(helpDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(helpDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::filesystem::path path = entry.path();
            if (path.extension() != ".png") {
                continue;
            }
            const std::string filename = path.filename().string();
            if (filename.rfind("frame", 0) != 0) {
                continue;
            }
            const std::size_t dotPos = filename.find('.');
            if (dotPos == std::string::npos || dotPos <= 5) {
                continue;
            }
            const std::string indexText = filename.substr(5, dotPos - 5);
            const bool allDigits = std::all_of(
                indexText.begin(),
                indexText.end(),
                [](unsigned char c) { return std::isdigit(c) != 0; });
            if (!allDigits) {
                continue;
            }
            helpFrames.push_back({std::stoi(indexText), path});
        }
    }

    if (helpFrames.empty()) {
        error_ = "No help frames found in ../assets/howtoplay";
        return false;
    }

    std::sort(helpFrames.begin(), helpFrames.end(), [](const HelpFrameEntry& a, const HelpFrameEntry& b) {
        return a.index < b.index;
    });

    helpFrameTextures_.clear();
    helpFrameTextures_.reserve(helpFrames.size());
    for (const auto& frame : helpFrames) {
        auto texture = std::make_unique<sf::Texture>();
        if (!texture->loadFromFile(frame.path.string())) {
            error_ = "Failed to load help frame texture: " + frame.path.string();
            return false;
        }
        const sf::Vector2u size = texture->getSize();
        if (size.x == 0 || size.y == 0) {
            error_ = "Invalid help frame texture size.";
            return false;
        }
        helpFrameTextures_.push_back(std::move(texture));
    }
    helpFrameIndex_ = 0;
    helpFrameSprite_.setTexture(*helpFrameTextures_.front());
    
    if (!pauseSettingsButtonTexture_.loadFromFile("../assets/settings_button.png")) {
        error_ = "Failed to load pause settings button texture.";
        return false;
    }
    pauseSettingsButtonSprite_.setTexture(pauseSettingsButtonTexture_);

    if (!backToMenuTexture_.loadFromFile("../assets/back_button.png")) {
        error_ = "Failed to load back to menu texture.";
        return false;
    }
    backToMenuSprite_.setTexture(backToMenuTexture_);
    
    if (!quitButtonTexture_.loadFromFile("../assets/quit_button.png")) {
        error_ = "Failed to load quit button texture.";
        return false;
    }
    quitButtonSprite_.setTexture(quitButtonTexture_);
    
    return true;
}

bool HexGameUI::initAudio() {
    //clean up any previously loaded audio with state checks
    if (menuMusic_.getStatus() == sf::Music::Playing) {
        menuMusic_.stop();
    }
    if (gameMusic_.getStatus() == sf::Music::Playing) {
        gameMusic_.stop();
    }
    if (gameOverSound_.getStatus() == sf::Sound::Playing) {
        gameOverSound_.stop();
    }
    if (gameClickSound_.getStatus() == sf::Sound::Playing) {
        gameClickSound_.stop();
    }

    // Add a small delay to let OpenAL settle
    sf::sleep(sf::milliseconds(50));

    //charge new audio files
    if (!menuMusic_.openFromFile("../assets/audio/hexMenu.ogg")) {
        error_ = "Failed to load menu music.";
        std::cerr << "Audio error: " << error_ << std::endl;
        return false;
    }
    if (!gameMusic_.openFromFile("../assets/audio/hexGame.ogg")) {
        error_ = "Failed to load game music.";
        std::cerr << "Audio error: " << error_ << std::endl;
        return false;
    }
    if (!gameOverBuffer_.loadFromFile("../assets/audio/gameOver.wav")) {
        error_ = "Failed to load game over sound.";
        std::cerr << "Audio error: " << error_ << std::endl;
        return false;
    }
    gameOverSound_.setBuffer(gameOverBuffer_);

    if (!clickBuffer_.loadFromFile("../assets/audio/Click.wav")) {
        error_ = "Failed to load game click sound.";
        std::cerr << "Audio error: " << error_ << std::endl;
        return false;
    }
    gameClickSound_.setBuffer(clickBuffer_);
    
    // Verify buffers are valid
    if (gameOverBuffer_.getSampleCount() == 0 || clickBuffer_.getSampleCount() == 0) {
        error_ = "Audio buffers are empty or invalid.";
        std::cerr << "Audio error: " << error_ << std::endl;
        return false;
    }
    
    menuMusic_.setLoop(true); 
    gameMusic_.setLoop(true);
    menuMusic_.setVolume(musicVolume_ * (masterVolume_ / 100.0f)); 
    gameMusic_.setVolume(musicVolume_ * (masterVolume_ / 100.0f));
    gameOverSound_.setVolume(musicVolume_ * (masterVolume_ / 100.0f));
    gameClickSound_.setVolume(sfxVolume_ * (masterVolume_ / 100.0f));
    
    // Add another small delay
    sf::sleep(sf::milliseconds(50));
    
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
    float buttonSize = windowSize_.y * 0.24f;
    pauseButtonSprite_.setScale(buttonSize / 1024.0f, buttonSize / 1024.0f);
    pauseButtonSprite_.setPosition(
        windowSize_.x - buttonSize - 15.0f,
        10.0f);

    // Setup pause menu overlay
    pauseMenuOverlay_.setSize(sf::Vector2f(windowSize_.x, windowSize_.y));
    pauseMenuOverlay_.setPosition(0.0f, 0.0f);
    pauseMenuOverlay_.setFillColor(sf::Color(0, 0, 0, 150));

    // Setup pause menu sprite centered
    float menuScale = windowSize_.y * 0.8f / 1024.0f;
    pauseMenuSprite_.setScale(menuScale, menuScale);
    pauseMenuSprite_.setPosition(
        (windowSize_.x - pauseMenuSprite_.getLocalBounds().width * menuScale) * 0.5f,
        (windowSize_.y - pauseMenuSprite_.getLocalBounds().height * menuScale) * 0.5f);
    
    // Setup pause menu buttons (5 buttons arranged vertically)
    // Button dimensions: 1792x576 (except settings which is 1024x329)
    float pauseMenuCenterX = windowSize_.x / 2.0f;
    float pauseMenuCenterY = windowSize_.y / 2.0f;
    float buttonHeight = std::min(windowSize_.x, windowSize_.y) * 0.12f;  // Adjusted button height
    float gap = buttonHeight * 0.05f;  // Reduced gap between buttons for tighter spacing
    
    // Resume button (1792x576)
    float resumeScale = buttonHeight / 576.0f;
    resumeButtonSprite_.setScale(resumeScale, resumeScale);
    float resumeWidth = 1792.0f * resumeScale;
    resumeButtonSprite_.setPosition(
        pauseMenuCenterX - resumeWidth / 2.0f,
        pauseMenuCenterY - buttonHeight * 2.0f - gap * 1.5f - 12.0f);
    
    // Restart button (1792x576)
    float restartScale = buttonHeight / 576.0f;
    restartButtonSprite_.setScale(restartScale, restartScale);
    float restartWidth = 1792.0f * restartScale;
    restartButtonSprite_.setPosition(
        pauseMenuCenterX - restartWidth / 2.0f,
        pauseMenuCenterY - buttonHeight - gap * 0.5f - 12.0f);
    
    // Help button (1792x576)
    float helpScale = buttonHeight / 576.0f;
    helpButtonSprite_.setScale(helpScale, helpScale);
    float helpWidth = 1792.0f * helpScale;
    helpButtonSprite_.setPosition(
        pauseMenuCenterX - helpWidth / 2.0f,
        pauseMenuCenterY + gap * 0.5f - 12.0f);
    
    // Settings button (1024x329)
    float settingsScale = buttonHeight / 329.0f;
    pauseSettingsButtonSprite_.setScale(settingsScale, settingsScale);
    float settingsWidth = 1024.0f * settingsScale;
    pauseSettingsButtonSprite_.setPosition(
        pauseMenuCenterX - settingsWidth / 2.0f,
        pauseMenuCenterY + buttonHeight + gap * 1.5f - 12.0f);
    
    // Quit button (1792x576)
    float quitScale = buttonHeight / 576.0f;
    quitButtonSprite_.setScale(quitScale, quitScale);
    float quitWidth = 1792.0f * quitScale;
    quitButtonSprite_.setPosition(
        pauseMenuCenterX - quitWidth / 2.0f,
        pauseMenuCenterY + buttonHeight * 2.0f + gap * 2.5f - 12.0f);
    
    // Setup settings menu (925x1520)
    float settingsMenuScale = (550.0f / 925.0f) * (windowSize_.y * 0.8f / 1024.0f);
    settingsMenuSprite_.setScale(settingsMenuScale, settingsMenuScale);
    float settingsMenuWidth = 925.0f * settingsMenuScale;
    float settingsMenuHeight = 1520.0f * settingsMenuScale;
    settingsMenuSprite_.setPosition(
        (windowSize_.x - settingsMenuWidth) / 2.0f,
        (windowSize_.y - settingsMenuHeight) / 2.0f);
    
    // Settings menu buttons (all 1792x576)
    float settingsButtonHeight = windowSize_.y * 0.12f;
    float settingsButtonScale = settingsButtonHeight / 576.0f;
    float settingsButtonWidth = 1792.0f * settingsButtonScale;
    float settingsMenuCenterX = windowSize_.x / 2.0f;
    float settingsMenuCenterY = windowSize_.y / 2.0f;
    float settingsGap = 8.0f;  // Reducido de 12.0f a 8.0f (4 píxeles menos)
    
    // Calcular altura total de los 4 botones con gaps entre ellos
    float totalHeight = (4.0f * settingsButtonHeight) + (3.0f * settingsGap);
    float startY = settingsMenuCenterY - (totalHeight / 2.0f)+10.0f;
    
    // Video button
    videoButtonSprite_.setScale(settingsButtonScale, settingsButtonScale);
    videoButtonSprite_.setPosition(
        settingsMenuCenterX - settingsButtonWidth / 2.0f,
        startY);
    
    // Audio button
    audioButtonSprite_.setScale(settingsButtonScale, settingsButtonScale);
    audioButtonSprite_.setPosition(
        settingsMenuCenterX - settingsButtonWidth / 2.0f,
        startY + settingsButtonHeight + settingsGap);
    
    // Credits button
    creditsButtonSprite_.setScale(settingsButtonScale, settingsButtonScale);
    creditsButtonSprite_.setPosition(
        settingsMenuCenterX - settingsButtonWidth / 2.0f,
        startY + 2.0f * settingsButtonHeight + 2.0f * settingsGap);
    
    // Back button
    settingsBackButtonSprite_.setScale(settingsButtonScale, settingsButtonScale);
    settingsBackButtonSprite_.setPosition(
        settingsMenuCenterX - settingsButtonWidth / 2.0f,
        startY + 3.0f * settingsButtonHeight + 3.0f * settingsGap);
    
    // Setup video menu (925x1520)
    float videoMenuScale = (550.0f / 925.0f) * (windowSize_.y * 0.8f / 1024.0f);
    videoMenuSprite_.setScale(videoMenuScale, videoMenuScale);
    float videoMenuWidth = 925.0f * videoMenuScale;
    float videoMenuHeight = 1520.0f * videoMenuScale;
    videoMenuSprite_.setPosition(
        (windowSize_.x - videoMenuWidth) / 2.0f,
        (windowSize_.y - videoMenuHeight) / 2.0f);
    
// --- Setup video menu selectors ---
    float videoMenuCenterX = windowSize_.x / 2.0f;
    float videoMenuCenterY = windowSize_.y / 2.0f;
    float videoContentStartY = videoMenuCenterY - videoMenuHeight / 2.0f + 60.0f;
    float videoItemGap = windowSize_.y * 0.08f;
    float spacing = 20.0f; // Un poco más de espacio para que no se vea amontonado

    // 1. Quality label
    float qualityLabelHeight = windowSize_.y * 0.03f; 
    float qualityLabelScale = qualityLabelHeight / 45.0f;
    qualityLabelSprite_.setScale(qualityLabelScale, qualityLabelScale);
    float qualityLabelWidth = qualityLabelTexture_.getSize().x * qualityLabelScale;
    float qualityLabelY = videoContentStartY + 10.0f;
    qualityLabelSprite_.setPosition(videoMenuCenterX - qualityLabelWidth / 2.0f, qualityLabelY);

    // 2. Quality selector row (Dinamismo puro)
    float qualityDisplayHeight = qualityLabelHeight * 0.6f;
    int qualityIndex = static_cast<int>(currentVideoQuality_);
    
    // Asignamos textura ANTES de calcular el ancho
    qualityDisplaySprite_.setTexture(qualityDisplayTextures_[qualityIndex]);
    float qTextScale = qualityDisplayHeight / qualityDisplayTextures_[qualityIndex].getSize().y;
    qualityDisplaySprite_.setScale(qTextScale, qTextScale);
    
    // El ancho cambia según si es "LOW", "MEDIUM" o "HIGH"
    float qTextWidth = qualityDisplayTextures_[qualityIndex].getSize().x * qTextScale;
    float qBtnScale = qualityDisplayHeight / 100.0f; 
    float qBtnSize = 100.0f * qBtnScale;

    // Calculamos el inicio de la fila basándonos en el ancho REAL del texto actual
    float qRowTotalWidth = qBtnSize + spacing + qTextWidth + spacing + qBtnSize;
    float qRowStartX = videoMenuCenterX - (qRowTotalWidth / 2.0f);
    float qRowY = qualityLabelY + qualityLabelHeight + 15.0f;

    // Posicionamos flechas de la primera fila
    leftButtonSprite_.setScale(qBtnScale, qBtnScale);
    leftButtonSprite_.setPosition(qRowStartX, qRowY);

    qualityDisplaySprite_.setPosition(qRowStartX + qBtnSize + spacing, qRowY);

    rightButtonSprite_.setScale(qBtnScale, qBtnScale);
    rightButtonSprite_.setPosition(qRowStartX + qBtnSize + spacing + qTextWidth + spacing, qRowY);

    leftButtonBounds_ = leftButtonSprite_.getGlobalBounds();
    rightButtonBounds_ = rightButtonSprite_.getGlobalBounds();

    // ---------------------------------------------------------

    // 3. Fullscreen label
    float fullscreenLabelY = qRowY + qualityDisplayHeight + videoItemGap;
    // Simplifiqué tu escala para que sea consistente con la de arriba
    float fsLabelScale = qualityLabelHeight / 45.0f; 
    fullscreenLabelSprite_.setScale(fsLabelScale, fsLabelScale);
    float fsLabelWidth = fullscreenLabelTexture_.getSize().x * fsLabelScale;
    fullscreenLabelSprite_.setPosition(videoMenuCenterX - fsLabelWidth / 2.0f, fullscreenLabelY);

    // 4. Fullscreen selector row (ENABLED / DISABLED)
    float fsDisplayHeight = qualityDisplayHeight;
    int fsIndex = fullscreenEnabled_ ? 1 : 0;
    
    fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[fsIndex]);
    float fsTextScale = fsDisplayHeight / fullscreenDisplayTextures_[fsIndex].getSize().y;
    fullscreenDisplaySprite_.setScale(fsTextScale, fsTextScale);
    
    // Ancho dinámico para Enabled/Disabled
    float fsTextWidth = fullscreenDisplayTextures_[fsIndex].getSize().x * fsTextScale;
    float fsBtnScale = fsDisplayHeight / 100.0f;
    float fsBtnSize = 100.0f * fsBtnScale;

    float fsRowTotalWidth = fsBtnSize + spacing + fsTextWidth + spacing + fsBtnSize;
    float fsRowStartX = videoMenuCenterX - (fsRowTotalWidth / 2.0f);
    float fsRowY = fullscreenLabelY + qualityLabelHeight + 15.0f;

    // IMPORTANTE: Aquí posicionamos los SPRITES de las flechas de nuevo
    // Nota: Como SFML usa los mismos sprites para dibujar, si los dibujas en dos sitios distintos 
    // en el render, debes guardar sus posiciones o usar sprites diferentes. 
    // Si usas los mismos sprites, la lógica de abajo solo sirve para definir los BOUNDS de clic:
    
    fullscreenLeftButtonBounds_ = sf::FloatRect(fsRowStartX, fsRowY, fsBtnSize, fsBtnSize);
    fullscreenDisplaySprite_.setPosition(fsRowStartX + fsBtnSize + spacing, fsRowY);
    fullscreenRightButtonBounds_ = sf::FloatRect(fsRowStartX + fsBtnSize + spacing + fsTextWidth + spacing, fsRowY, fsBtnSize, fsBtnSize);

    // Guardar posiciones de las flechas para fullscreen para usarlas al renderizar
    fsLeftBtnPos_ = sf::Vector2f(fsRowStartX, fsRowY);
    fsRightBtnPos_ = sf::Vector2f(fsRowStartX + fsBtnSize + spacing + fsTextWidth + spacing, fsRowY);
    fsBtnScale_ = fsBtnScale;

    // Setup audio menu (925x1520)
    float audioMenuScale = (550.0f / 925.0f) * (windowSize_.y * 0.8f / 1024.0f);
    audioMenuSprite_.setScale(audioMenuScale, audioMenuScale);
    float audioMenuWidth = 925.0f * audioMenuScale;
    float audioMenuHeight = 1520.0f * audioMenuScale;
    audioMenuSprite_.setPosition(
        (windowSize_.x - audioMenuWidth) / 2.0f,
        (windowSize_.y - audioMenuHeight) / 2.0f);
    
    // Setup credits menu (925x1520)
    float creditsMenuScale = (550.0f / 925.0f) * (windowSize_.y * 0.8f / 1024.0f);
    creditsMenuSprite_.setScale(creditsMenuScale, creditsMenuScale);
    float creditsMenuWidth = 925.0f * creditsMenuScale;
    float creditsMenuHeight = 1520.0f * creditsMenuScale;
    creditsMenuSprite_.setPosition(
        (windowSize_.x - creditsMenuWidth) / 2.0f,
        (windowSize_.y - creditsMenuHeight) / 2.0f);
    
    // Setup submenu back button dimensions (needed for positioning credits image)
    float submenuBackButtonHeight = windowSize_.y * 0.12f;
    
    // Setup credits image (530x642) - scale to fit nicely in the menu
    float creditsImageMaxWidth = windowSize_.x * 0.22f;
    float creditsImageScale = creditsImageMaxWidth / 530.0f;
    creditsImageSprite_.setScale(creditsImageScale, creditsImageScale);
    float creditsImageWidth = 530.0f * creditsImageScale;
    float creditsImageHeight = 642.0f * creditsImageScale;
    float creditsImageX = (windowSize_.x - creditsImageWidth) / 2.0f;
    float creditsImageY = (windowSize_.y - creditsImageHeight) / 2.0f - submenuBackButtonHeight * 0.5f+8.0f;
    creditsImageSprite_.setPosition(creditsImageX, creditsImageY);
    creditsImageBounds_ = sf::FloatRect(creditsImageX, creditsImageY, creditsImageWidth, creditsImageHeight);
    
    // Setup submenu back button at the bottom (below credits image)
    float submenuBackButtonScale = submenuBackButtonHeight / 576.0f;
    submenuBackButtonSprite_.setScale(submenuBackButtonScale, submenuBackButtonScale);
    float submenuBackButtonWidth = 1792.0f * submenuBackButtonScale;
    submenuBackButtonSprite_.setPosition(
        settingsMenuCenterX - submenuBackButtonWidth / 2.0f,
        creditsImageY + creditsImageHeight + 10.0f);
    
    // Setup audio sliders (Master Volume, Music Volume, SFX Volume)
    float sliderWidth = windowSize_.x * 0.16f; 
    float sliderHeight = windowSize_.y * 0.016f;  
    float handleSize = windowSize_.y * 0.05f;   
    sliderHandleRadius_ = handleSize / 2.0f;
    float sliderCenterX = windowSize_.x / 2.0f;
    
    // Setup volume icon sprites
    float iconSize = windowSize_.y * 0.05f;
    for (int i = 0; i < 3; ++i) {
        volumeIconSprites_[i].setScale(iconSize / volumeIconTextures_[3].getSize().y, 
                                       iconSize / volumeIconTextures_[3].getSize().y);
    }
    
    // Setup volume label sprites (900x100 images)
    float labelWidth = windowSize_.x * 0.22f; 
    float labelHeight = labelWidth / 9.0f; 
    masterVolumeLabelSprite_.setScale(labelWidth / masterVolumeLabelTexture_.getSize().x, 
                                      labelHeight / masterVolumeLabelTexture_.getSize().y);
    musicVolumeLabelSprite_.setScale(labelWidth / musicVolumeLabelTexture_.getSize().x, 
                                     labelHeight / musicVolumeLabelTexture_.getSize().y);
    effectsVolumeLabelSprite_.setScale(labelWidth / effectsVolumeLabelTexture_.getSize().x, 
                                       labelHeight / effectsVolumeLabelTexture_.getSize().y);
    
    // Calculate total height needed for all volume controls
    float volumeItemGap = windowSize_.y * 0.04f;  // Gap between label+slider items (50% reduction)
    float totalVolumeHeight = 3.0f * (labelHeight + sliderHeight + volumeItemGap) + submenuBackButtonHeight;
    
    // Calculate audio menu center
    float audioMenuCenterY = (windowSize_.y - audioMenuHeight) / 2.0f + audioMenuHeight / 2.0f;
    
    // Position all volume controls centered in the audio menu with 10px offset
    float volumeStartY = audioMenuCenterY - totalVolumeHeight / 2.0f + 10.0f;
    
    // Center both icon and slider together as one unit
    float iconPadding = 6.0f;
    float totalWidth = iconSize + iconPadding + sliderWidth;
    float totalCenterX = sliderCenterX - totalWidth / 2.0f;
    
    // Master Volume Slider
    float labelToSliderGap = 4.0f;
    float masterLabelY = volumeStartY;
    float masterSliderY = masterLabelY + labelHeight + labelToSliderGap;
    masterVolumeLabelSprite_.setPosition(sliderCenterX - labelWidth / 2.0f, masterLabelY);
    masterVolumeSlider_.background.setSize(sf::Vector2f(sliderWidth, sliderHeight));
    masterVolumeSlider_.background.setFillColor(sf::Color(0, 255, 255));  // Cyan color
    masterVolumeSlider_.handle.setTexture(sliderHandleTexture_);
    masterVolumeSlider_.handle.setScale(handleSize / sliderHandleTexture_.getSize().x, 
                                         handleSize / sliderHandleTexture_.getSize().y);
    volumeIconSprites_[0].setPosition(totalCenterX, masterSliderY + (sliderHeight - iconSize) / 2.0f);
    masterVolumeSlider_.background.setPosition(totalCenterX + iconSize + iconPadding, masterSliderY);
    masterVolumeSlider_.minX = totalCenterX + iconSize + iconPadding;
    masterVolumeSlider_.maxX = totalCenterX + iconSize + iconPadding + sliderWidth;
    masterVolumeSlider_.value = 100.0f;
    float masterHandleX = masterVolumeSlider_.minX + (masterVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
    masterVolumeSlider_.handle.setPosition(masterHandleX, masterSliderY - (handleSize - sliderHeight) / 2.0f);
    
    // Music Volume Slider
    float musicLabelY = masterSliderY + sliderHeight + volumeItemGap;
    float musicSliderY = musicLabelY + labelHeight + labelToSliderGap;
    musicVolumeLabelSprite_.setPosition(sliderCenterX - labelWidth / 2.0f, musicLabelY);
    musicVolumeSlider_.background.setSize(sf::Vector2f(sliderWidth, sliderHeight));
    musicVolumeSlider_.background.setFillColor(sf::Color(0, 255, 255));  // Cyan color
    musicVolumeSlider_.handle.setTexture(sliderHandleTexture_);
    musicVolumeSlider_.handle.setScale(handleSize / sliderHandleTexture_.getSize().x, 
                                        handleSize / sliderHandleTexture_.getSize().y);
    volumeIconSprites_[1].setPosition(totalCenterX, musicSliderY + (sliderHeight - iconSize) / 2.0f);
    musicVolumeSlider_.background.setPosition(totalCenterX + iconSize + iconPadding, musicSliderY);
    musicVolumeSlider_.minX = totalCenterX + iconSize + iconPadding;
    musicVolumeSlider_.maxX = totalCenterX + iconSize + iconPadding + sliderWidth;
    musicVolumeSlider_.value = 50.0f;
    float musicHandleX = musicVolumeSlider_.minX + (musicVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
    musicVolumeSlider_.handle.setPosition(musicHandleX, musicSliderY - (handleSize - sliderHeight) / 2.0f);
    
    // SFX Volume Slider
    float effectsLabelY = musicSliderY + sliderHeight + volumeItemGap;
    float effectsSliderY = effectsLabelY + labelHeight + labelToSliderGap;
    effectsVolumeLabelSprite_.setPosition(sliderCenterX - labelWidth / 2.0f, effectsLabelY);
    sfxVolumeSlider_.background.setSize(sf::Vector2f(sliderWidth, sliderHeight));
    sfxVolumeSlider_.background.setFillColor(sf::Color(0, 255, 255));  // Cyan color
    sfxVolumeSlider_.handle.setTexture(sliderHandleTexture_);
    sfxVolumeSlider_.handle.setScale(handleSize / sliderHandleTexture_.getSize().x, 
                                      handleSize / sliderHandleTexture_.getSize().y);
    volumeIconSprites_[2].setPosition(totalCenterX, effectsSliderY + (sliderHeight - iconSize) / 2.0f);
    sfxVolumeSlider_.background.setPosition(totalCenterX + iconSize + iconPadding, effectsSliderY);
    sfxVolumeSlider_.minX = totalCenterX + iconSize + iconPadding;
    sfxVolumeSlider_.maxX = totalCenterX + iconSize + iconPadding + sliderWidth;
    sfxVolumeSlider_.value = 80.0f;
    float sfxHandleX = sfxVolumeSlider_.minX + (sfxVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
    sfxVolumeSlider_.handle.setPosition(sfxHandleX, effectsSliderY - (handleSize - sliderHeight) / 2.0f);
    
    // Position back button below all volume controls (or under credits image when Credits menu active)
    {
        float backX = settingsMenuCenterX - submenuBackButtonWidth / 2.0f;
        float backYAudio = effectsSliderY + sliderHeight + volumeItemGap;
        float backY = backYAudio;
        // If the credits image was laid out, prefer placing the back button below it
        if (creditsImageHeight > 0.0f) {
            float creditsBackY = creditsImageY + creditsImageHeight + 6.0f;
            // Use credits position when it is lower (so it doesn't overlap the image)
            if (creditsBackY > backY) {
                backY = creditsBackY;
            }
        }
        submenuBackButtonSprite_.setPosition(backX, backY);
    }
    
    // Setup help frame sprite
    if (!helpFrameTextures_.empty()) {
        float helpFrameMaxWidth = windowSize_.x * 0.8f;
        float helpFrameMaxHeight = windowSize_.y * 0.8f;
        sf::Vector2u frameSize = helpFrameTextures_.front()->getSize();
        float helpFrameScale = std::min(helpFrameMaxWidth / frameSize.x, helpFrameMaxHeight / frameSize.y);
        helpFrameSprite_.setScale(helpFrameScale, helpFrameScale);
        float helpFrameWidth = frameSize.x * helpFrameScale;
        float helpFrameHeight = frameSize.y * helpFrameScale;
        helpFrameSprite_.setPosition(
            (windowSize_.x - helpFrameWidth) / 2.0f,
            (windowSize_.y - helpFrameHeight) / 2.0f);
    }
    
    // Setup back to menu button
    if (backToMenuTexture_.getSize().x != 0 && backToMenuTexture_.getSize().y != 0) {
        float backButtonHeight = windowSize_.y * 0.08f;
        float backButtonScale = backButtonHeight / backToMenuTexture_.getSize().y;
        backToMenuSprite_.setScale(backButtonScale, backButtonScale);
        float backButtonWidth = backToMenuTexture_.getSize().x * backButtonScale;
        backToMenuSprite_.setPosition(
            windowSize_.x - backButtonWidth - 15.0f,
            15.0f);
    }
}

void HexGameUI::updateTileColors() {
    const sf::Color emptyColor(210, 210, 220);
    const sf::Color borderRed(235, 175, 175);
    const sf::Color borderBlue(175, 195, 235);
    const sf::Color borderMix(220, 185, 220);
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
        } else {
            const bool onRedSide = (col == 0) || (col == boardSize_ - 1);
            const bool onBlueSide = (row == 0) || (row == boardSize_ - 1);
            if (onRedSide && onBlueSide) {
                color = borderMix;
            } else if (onRedSide) {
                color = borderRed;
            } else if (onBlueSide) {
                color = borderBlue;
            }
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
            if (gameMusic_.getStatus() == sf::Music::Playing) {
                gameMusic_.stop();
            }
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
    showHelp_ = false;
    helpFrameIndex_ = 0;
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
    boardSizeText_.setString(std::to_string(boardSize_));
}

void HexGameUI::updateHelpFrameSprite() {
    if (!helpFrameTextures_.empty() && helpFrameIndex_ < helpFrameTextures_.size()) {
        helpFrameSprite_.setTexture(*helpFrameTextures_[helpFrameIndex_]);
    }
}

void HexGameUI::applyBoardSize(int newSize) {
    if (newSize < 7 || newSize > 20 || newSize == boardSize_) {
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

        // hardwareInfoText_.setPosition(menuX + margin, menuY + 2*margin );  // Commented for future use

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
            sf::FloatRect labelBounds(0.0f, 0.0f, 0.0f, 0.0f);
            if (boardSizeLabelTexture_.getSize().x != 0 &&
                boardSizeLabelTexture_.getSize().y != 0) {
                const float baseSize = std::min(windowSize_.x, windowSize_.y);
                const float desiredHeight = std::max(32.0f, baseSize * 0.08f);
                const float labelScale =
                    desiredHeight / static_cast<float>(boardSizeLabelTexture_.getSize().y);
                boardSizeLabelSprite_.setScale(labelScale, labelScale);
                const float labelWidth =
                    boardSizeLabelTexture_.getSize().x * labelScale;
                const float labelHeight =
                    boardSizeLabelTexture_.getSize().y * labelScale;
                boardSizeLabelSprite_.setPosition(
                    centerX - labelWidth * 0.5f,
                    centerY - labelHeight * 0.5f);
                labelBounds = boardSizeLabelSprite_.getGlobalBounds();
                textHalf = labelBounds.height * 0.5f;
            }
            if (startFontLoaded_) {
                if (labelBounds.height > 0.0f) {
                    const float targetHeight = labelBounds.height;
                    boardSizeText_.setCharacterSize(
                        static_cast<unsigned int>(std::max(14.0f, targetHeight * 0.6f)));
                }
                const sf::FloatRect textBounds = boardSizeText_.getLocalBounds();
                const float textX = (labelBounds.width > 0.0f)
                    ? labelBounds.left + (labelBounds.width - textBounds.width) / 2.0f - textBounds.left
                    : centerX - textBounds.width * 0.5f - textBounds.left;
                const float textY = (labelBounds.height > 0.0f)
                    ? labelBounds.top + (labelBounds.height - textBounds.height) / 2.0f - textBounds.top
                    : centerY - textBounds.height * 0.5f - textBounds.top;
                boardSizeText_.setPosition(textX, textY);
                if (textHalf == 0.0f) {
                    textHalf = textBounds.height * 0.5f;
                }
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

            if (helpButtonTexture_.getSize().x != 0 && helpButtonTexture_.getSize().y != 0) {
                const float desiredHeight = std::min(windowSize_.x, windowSize_.y) * 0.10f;
                const float helpScale =
                    desiredHeight / static_cast<float>(helpButtonTexture_.getSize().y);
                helpSelectButtonSprite_.setScale(helpScale, helpScale);
                const float helpWidth = helpButtonTexture_.getSize().x * helpScale;
                const float margin = 12.0f;
                helpSelectButtonSprite_.setPosition(
                    static_cast<float>(windowSize_.x) - helpWidth - margin,
                    margin);
            }
        }
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                    menuMusic_.stop();
                    gameMusic_.stop();
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
                        // Handle settings menu button clicks
                        if (showSettingsMenu_) {
                            // Handle submenu clicks
                            if (settingsMenuState_ == SettingsMenuState::Video) {
                                // Check quality selector buttons
                                if (leftButtonBounds_.contains(mousePos)) {
                                    // Decrease quality
                                    if (currentVideoQuality_ == VideoQuality::Low) {
                                        currentVideoQuality_ = VideoQuality::High;
                                    } else {
                                        currentVideoQuality_ = static_cast<VideoQuality>(
                                            static_cast<int>(currentVideoQuality_) - 1);
                                    }
                                    qualityDisplaySprite_.setTexture(
                                        qualityDisplayTextures_[static_cast<int>(currentVideoQuality_)]);
                                    gameClickSound_.play();
                                    buildLayout();  // Refresh layout to center the text dynamically
                                } else if (rightButtonBounds_.contains(mousePos)) {
                                    // Increase quality
                                    if (currentVideoQuality_ == VideoQuality::High) {
                                        currentVideoQuality_ = VideoQuality::Low;
                                    } else {
                                        currentVideoQuality_ = static_cast<VideoQuality>(
                                            static_cast<int>(currentVideoQuality_) + 1);
                                    }
                                    qualityDisplaySprite_.setTexture(
                                        qualityDisplayTextures_[static_cast<int>(currentVideoQuality_)]);
                                    gameClickSound_.play();
                                    buildLayout();  // Refresh layout to center the text dynamically
                                } else if (fullscreenLeftButtonBounds_.contains(mousePos)) {
                                    // Toggle fullscreen (left button decreases - windowed)
                                    fullscreenEnabled_ = false;
                                    fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[0]);
                                    gameClickSound_.play();
                                    buildLayout();  // Refresh layout to center the text dynamically
                                } else if (fullscreenRightButtonBounds_.contains(mousePos)) {
                                    // Toggle fullscreen (right button increases - fullscreen)
                                    fullscreenEnabled_ = true;
                                    fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[1]);
                                    gameClickSound_.play();
                                    buildLayout();  // Refresh layout to center the text dynamically
                                } else if (submenuBackButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Main;
                                }
                            } else if (settingsMenuState_ == SettingsMenuState::Audio) {
                                // Check if clicking on slider handles
                                float handleRadius = sliderHandleRadius_;
                                if (masterVolumeSlider_.handle.getGlobalBounds().contains(mousePos)) {
                                    draggingSlider_ = &masterVolumeSlider_;
                                } else if (musicVolumeSlider_.handle.getGlobalBounds().contains(mousePos)) {
                                    draggingSlider_ = &musicVolumeSlider_;
                                } else if (sfxVolumeSlider_.handle.getGlobalBounds().contains(mousePos)) {
                                    draggingSlider_ = &sfxVolumeSlider_;
                                } else if (submenuBackButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Main;
                                }
                            } else if (settingsMenuState_ == SettingsMenuState::Credits) {
                                if (creditsImageBounds_.contains(mousePos)) {
                                    // Click on credits image - trigger animation
                                    creditsImageClicked_ = true;
                                    creditsClickClock_.restart();
                                    gameClickSound_.play();
                                } else if (submenuBackButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Main;
                                }
                            } else {
                                // Main settings menu
                                if (videoButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Video;
                                } else if (audioButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Audio;
                                } else if (creditsButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    settingsMenuState_ = SettingsMenuState::Credits;
                                } else if (settingsBackButtonSprite_.getGlobalBounds().contains(mousePos)) {
                                    gameClickSound_.play();
                                    showSettingsMenu_ = false;
                                    settingsMenuState_ = SettingsMenuState::Main;
                                } else if (!settingsMenuSprite_.getGlobalBounds().contains(mousePos)) {
                                    // Close menu if clicking outside
                                    showSettingsMenu_ = false;
                                    settingsMenuState_ = SettingsMenuState::Main;
                                }
                            }
                        }
                    }
                } else if (event.type == sf::Event::MouseButtonReleased) {
                    // Stop dragging slider and save configuration
                    if (draggingSlider_ != nullptr) {
                        saveVolumeConfig();
                    }
                    draggingSlider_ = nullptr;
                } else if (event.type == sf::Event::MouseMoved) {
                    // Handle slider dragging
                    if (draggingSlider_ != nullptr && showSettingsMenu_ && settingsMenuState_ == SettingsMenuState::Audio) {
                        sf::Vector2f mousePos = window.mapPixelToCoords(
                            sf::Vector2i(event.mouseMove.x, event.mouseMove.y));
                        
                        // Calculate new slider value based on mouse X position
                        float sliderWidth = draggingSlider_->maxX - draggingSlider_->minX;
                        float relativeX = mousePos.x - draggingSlider_->minX;
                        relativeX = std::max(0.0f, std::min(relativeX, sliderWidth));
                        draggingSlider_->value = (relativeX / sliderWidth) * 100.0f;
                        
                        // Update handle position
                        float handleSize = sliderHandleRadius_ * 2.0f;
                        float newHandleX = draggingSlider_->minX + (draggingSlider_->value / 100.0f) * sliderWidth - handleSize / 2.0f;
                        draggingSlider_->handle.setPosition(newHandleX, draggingSlider_->handle.getPosition().y);
                        
                        // Update volume variables and audio based on which slider is being dragged
                        if (draggingSlider_ == &masterVolumeSlider_) {
                            masterVolume_ = draggingSlider_->value;
                            updateVolumeIcon(0, draggingSlider_->value);
                        } else if (draggingSlider_ == &musicVolumeSlider_) {
                            musicVolume_ = draggingSlider_->value;
                            updateVolumeIcon(1, draggingSlider_->value);
                        } else if (draggingSlider_ == &sfxVolumeSlider_) {
                            sfxVolume_ = draggingSlider_->value;
                            updateVolumeIcon(2, draggingSlider_->value);
                        }
                        
                        // Apply volume changes immediately
                        applyVolumeChanges();
                    }
                }
                continue; 
            }

            if (screen_ == UIScreen::PlayerSelect) {
                if (event.type == sf::Event::MouseButtonPressed &&
                    event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2f mousePos = window.mapPixelToCoords(
                        sf::Vector2i(event.mouseButton.x, event.mouseButton.y));

                    if (showHelp_) {
                        if (backToMenuTexture_.getSize().x != 0 &&
                            backToMenuSprite_.getGlobalBounds().contains(mousePos)) {
                            gameClickSound_.play();
                            showHelp_ = false;
                        }
                        continue;
                    }

                    if (helpButtonTexture_.getSize().x != 0 &&
                        helpSelectButtonSprite_.getGlobalBounds().contains(mousePos)) {
                        gameClickSound_.play();
                        showHelp_ = true;
                        helpFrameIndex_ = 0;
                        helpFrameClock_.restart();
                        updateHelpFrameSprite();
                        continue;
                    }

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
                if (gamePaused_ && showHelp_) {
                    if (backToMenuTexture_.getSize().x != 0 &&
                        backToMenuSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        showHelp_ = false;
                    }
                    continue; // Prevent further processing
                }
                if (pauseButtonSprite_.getGlobalBounds().contains(pos) && winnerId_ == 0) {
                    gameClickSound_.play();
                    gamePaused_ = !gamePaused_;
                    if (!gamePaused_) {
                        showHelp_ = false;
                    }
                    continue; // Prevent further processing
                }
                
                // Handle pause menu button clicks
                if (gamePaused_) {
                    // Handle settings menu if open
                    if (showSettingsMenu_) {
                        if (settingsMenuState_ == SettingsMenuState::Video) {
                            // Check quality selector buttons
                            if (leftButtonBounds_.contains(pos)) {
                                // Decrease quality
                                if (currentVideoQuality_ == VideoQuality::Low) {
                                    currentVideoQuality_ = VideoQuality::High;
                                } else {
                                    currentVideoQuality_ = static_cast<VideoQuality>(
                                        static_cast<int>(currentVideoQuality_) - 1);
                                }
                                qualityDisplaySprite_.setTexture(
                                    qualityDisplayTextures_[static_cast<int>(currentVideoQuality_)]);
                                gameClickSound_.play();
                                buildLayout();  // Refresh layout to center the text dynamically
                            } else if (rightButtonBounds_.contains(pos)) {
                                // Increase quality
                                if (currentVideoQuality_ == VideoQuality::High) {
                                    currentVideoQuality_ = VideoQuality::Low;
                                } else {
                                    currentVideoQuality_ = static_cast<VideoQuality>(
                                        static_cast<int>(currentVideoQuality_) + 1);
                                }
                                qualityDisplaySprite_.setTexture(
                                    qualityDisplayTextures_[static_cast<int>(currentVideoQuality_)]);
                                gameClickSound_.play();
                                buildLayout();  // Refresh layout to center the text dynamically
                            } else if (fullscreenLeftButtonBounds_.contains(pos)) {
                                // Toggle fullscreen (left button decreases - windowed)
                                fullscreenEnabled_ = false;
                                fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[0]);
                                gameClickSound_.play();
                                buildLayout();  // Refresh layout to center the text dynamically
                            } else if (fullscreenRightButtonBounds_.contains(pos)) {
                                // Toggle fullscreen (right button increases - fullscreen)
                                fullscreenEnabled_ = true;
                                fullscreenDisplaySprite_.setTexture(fullscreenDisplayTextures_[1]);
                                gameClickSound_.play();
                                buildLayout();  // Refresh layout to center the text dynamically
                            } else if (submenuBackButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Main;
                            }
                        } else if (settingsMenuState_ == SettingsMenuState::Audio) {
                            // Check if clicking on slider handles
                            float handleRadius = sliderHandleRadius_;
                            if (masterVolumeSlider_.handle.getGlobalBounds().contains(pos)) {
                                draggingSlider_ = &masterVolumeSlider_;
                            } else if (musicVolumeSlider_.handle.getGlobalBounds().contains(pos)) {
                                draggingSlider_ = &musicVolumeSlider_;
                            } else if (sfxVolumeSlider_.handle.getGlobalBounds().contains(pos)) {
                                draggingSlider_ = &sfxVolumeSlider_;
                            } else if (submenuBackButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Main;
                            }
                        } else if (settingsMenuState_ == SettingsMenuState::Credits) {
                            if (submenuBackButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Main;
                            }
                        } else {
                            // Main settings menu
                            if (videoButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Video;
                            } else if (audioButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Audio;
                            } else if (creditsButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                settingsMenuState_ = SettingsMenuState::Credits;
                            } else if (settingsBackButtonSprite_.getGlobalBounds().contains(pos)) {
                                gameClickSound_.play();
                                showSettingsMenu_ = false;
                                settingsMenuState_ = SettingsMenuState::Main;
                            } else if (!settingsMenuSprite_.getGlobalBounds().contains(pos)) {
                                // Close menu if clicking outside
                                showSettingsMenu_ = false;
                                settingsMenuState_ = SettingsMenuState::Main;
                            }
                        }
                        continue; // Prevent further processing
                    }
                    
                    if (resumeButtonSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        gamePaused_ = false;
                        showHelp_ = false;
                        continue; // Prevent further processing
                    } else if (restartButtonSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        gamePaused_ = false;
                        showHelp_ = false;
                        resetGame();
                        updateWindowTitle(window);
                        continue; // Prevent further processing
                    } else if (helpButtonSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        showHelp_ = true;
                        helpFrameIndex_ = 0;
                        helpFrameClock_.restart();
                        updateHelpFrameSprite();
                        continue; // Prevent further processing
                    } else if (pauseSettingsButtonSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        showSettingsMenu_ = true;
                        settingsMenuState_ = SettingsMenuState::Main;
                        continue; // Prevent further processing
                    } else if (quitButtonSprite_.getGlobalBounds().contains(pos)) {
                        gameClickSound_.play();
                        gamePaused_ = false;
                        showHelp_ = false;
                        screen_ = UIScreen::Start;
                        switchMusic(false);
                        resetGame();
                        continue; // Prevent further processing
                    }
                }
            }

            // Handle pause menu mouse release
            if (screen_ == UIScreen::Game && gamePaused_ && event.type == sf::Event::MouseButtonReleased) {
                // Stop dragging slider and save configuration
                if (draggingSlider_ != nullptr) {
                    saveVolumeConfig();
                }
                draggingSlider_ = nullptr;
            }
            
            // Handle pause menu mouse movement for slider dragging
            if (screen_ == UIScreen::Game && gamePaused_ && event.type == sf::Event::MouseMoved) {
                // Handle slider dragging
                if (draggingSlider_ != nullptr && showSettingsMenu_ && settingsMenuState_ == SettingsMenuState::Audio) {
                    sf::Vector2f mousePos = window.mapPixelToCoords(
                        sf::Vector2i(event.mouseMove.x, event.mouseMove.y));
                    
                    // Calculate new slider value based on mouse X position
                    float sliderWidth = draggingSlider_->maxX - draggingSlider_->minX;
                    float relativeX = mousePos.x - draggingSlider_->minX;
                    relativeX = std::max(0.0f, std::min(relativeX, sliderWidth));
                    draggingSlider_->value = (relativeX / sliderWidth) * 100.0f;
                    
                    // Update handle position
                    float handleSize = sliderHandleRadius_ * 2.0f;
                    float newHandleX = draggingSlider_->minX + (draggingSlider_->value / 100.0f) * sliderWidth - handleSize / 2.0f;
                    draggingSlider_->handle.setPosition(newHandleX, draggingSlider_->handle.getPosition().y);
                    
                    // Update volume variables and audio based on which slider is being dragged
                    if (draggingSlider_ == &masterVolumeSlider_) {
                        masterVolume_ = draggingSlider_->value;
                        updateVolumeIcon(0, draggingSlider_->value);
                    } else if (draggingSlider_ == &musicVolumeSlider_) {
                        musicVolume_ = draggingSlider_->value;
                        updateVolumeIcon(1, draggingSlider_->value);
                    } else if (draggingSlider_ == &sfxVolumeSlider_) {
                        sfxVolume_ = draggingSlider_->value;
                        updateVolumeIcon(2, draggingSlider_->value);
                    }
                    
                    // Apply volume changes immediately
                    applyVolumeChanges();
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
                    buildLayout();
                    updateWindowTitle(window);
                    switchMusic(true); // start game music
                    continue; // Prevent further processing
                } else if (quitButtonTexture_.getSize().x != 0 &&
                           quitButtonSprite_.getGlobalBounds().contains(pos)) {
                    gameClickSound_.play();
                    // Go back to start screen
                    resetGame();
                    buildLayout();
                    screen_ = UIScreen::Start;
                    showSettingsMenu_ = false;
                    switchMusic(false); // play menu music
                    continue; // Prevent further processing
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

        if (showHelp_ && !helpFrameTextures_.empty()) {
            float frameDelay = kHelpFrameDelaySeconds;
            if (helpFrameTextures_.size() >= 2 &&
                helpFrameIndex_ >= helpFrameTextures_.size() - 2) {
                frameDelay = kHelpFrameDelayLastSeconds;
            }
            if (helpFrameClock_.getElapsedTime().asSeconds() >= frameDelay) {
                helpFrameClock_.restart();
                helpFrameIndex_ = (helpFrameIndex_ + 1) % helpFrameTextures_.size();
                updateHelpFrameSprite();
            }
        }

        if (screen_ == UIScreen::Start) {
            window.clear(sf::Color(30, 30, 40));

            // Update credits image animation
            if (settingsMenuState_ == SettingsMenuState::Credits && creditsImageClicked_) {
                const float animationDuration = 0.6f;  // Animation duration in seconds
                const float t = creditsClickClock_.getElapsedTime().asSeconds();
                if (t < animationDuration) {
                    // Rotation animation (360 degrees)
                    float rotation = (t / animationDuration) * 360.0f;
                    creditsImageSprite_.setRotation(rotation);
                    
                    // Scale animation (1.0 to 1.2 and back to 1.0)
                    float scale = 1.0f + 0.2f * std::sin(t / animationDuration * 3.14159f);
                    const sf::Vector2u textureSize = creditsImageTexture_.getSize();
                    float baseScale = (windowSize_.x * 0.22f) / 530.0f;
                    creditsImageSprite_.setScale(baseScale * scale, baseScale * scale);
                } else {
                    // Reset animation
                    creditsImageClicked_ = false;
                    float baseScale = (windowSize_.x * 0.22f) / 530.0f;
                    creditsImageSprite_.setScale(baseScale, baseScale);
                    creditsImageSprite_.setRotation(0.0f);
                }
            }

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
                
                if (settingsMenuState_ == SettingsMenuState::Main) {
                    // Draw main settings menu
                    if (settingsMenuTexture_.getSize().x != 0) {
                        window.draw(settingsMenuSprite_);
                    }
                    
                    // Draw settings menu buttons
                    if (videoButtonTexture_.getSize().x != 0) {
                        window.draw(videoButtonSprite_);
                    }
                    if (audioButtonTexture_.getSize().x != 0) {
                        window.draw(audioButtonSprite_);
                    }
                    if (creditsButtonTexture_.getSize().x != 0) {
                        window.draw(creditsButtonSprite_);
                    }
                    if (settingsBackButtonTexture_.getSize().x != 0) {
                        window.draw(settingsBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Video) {
                    // Draw video menu
                    if (videoMenuTexture_.getSize().x != 0) {
                        window.draw(videoMenuSprite_);
                    }
                    // Draw quality label and selector
                    window.draw(qualityLabelSprite_);
                    window.draw(leftButtonSprite_);
                    window.draw(qualityDisplaySprite_);
                    window.draw(rightButtonSprite_);
                    // Draw fullscreen label and selector
                    window.draw(fullscreenLabelSprite_);
                    // Move left button to fullscreen row position
                    leftButtonSprite_.setPosition(fsLeftBtnPos_);
                    leftButtonSprite_.setScale(fsBtnScale_, fsBtnScale_);
                    window.draw(leftButtonSprite_);
                    window.draw(fullscreenDisplaySprite_);
                    // Move right button to fullscreen row position
                    rightButtonSprite_.setPosition(fsRightBtnPos_);
                    rightButtonSprite_.setScale(fsBtnScale_, fsBtnScale_);
                    window.draw(rightButtonSprite_);
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Audio) {
                    // Draw audio menu
                    if (audioMenuTexture_.getSize().x != 0) {
                        window.draw(audioMenuSprite_);
                    }
                    // Draw volume labels
                    window.draw(masterVolumeLabelSprite_);
                    window.draw(musicVolumeLabelSprite_);
                    window.draw(effectsVolumeLabelSprite_);
                    // Draw volume sliders
                    window.draw(masterVolumeSlider_.background);
                    window.draw(masterVolumeSlider_.handle);
                    window.draw(musicVolumeSlider_.background);
                    window.draw(musicVolumeSlider_.handle);
                    window.draw(sfxVolumeSlider_.background);
                    window.draw(sfxVolumeSlider_.handle);
                    // Draw volume icons
                    for (int i = 0; i < 3; ++i) {
                        window.draw(volumeIconSprites_[i]);
                    }
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Credits) {
                    // Draw credits menu
                    if (creditsMenuTexture_.getSize().x != 0) {
                        window.draw(creditsMenuSprite_);
                    }
                    // Draw credits image
                    if (creditsImageTexture_.getSize().x != 0) {
                        window.draw(creditsImageSprite_);
                    }
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
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
            if (boardSizeLabelTexture_.getSize().x != 0 &&
                boardSizeLabelTexture_.getSize().y != 0) {
                window.draw(boardSizeLabelSprite_);
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

            if (showHelp_) {
                if (!helpFrameTextures_.empty()) {
                    window.draw(helpFrameSprite_);
                }
                if (backToMenuTexture_.getSize().x != 0) {
                    window.draw(backToMenuSprite_);
                }
            } else if (helpButtonTexture_.getSize().x != 0) {
                window.draw(helpSelectButtonSprite_);
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

        // Update credits image animation (same as in Start screen)
        if (showSettingsMenu_ && settingsMenuState_ == SettingsMenuState::Credits && creditsImageClicked_) {
            const float animationDuration = 0.6f;  // Animation duration in seconds
            const float t = creditsClickClock_.getElapsedTime().asSeconds();
            if (t < animationDuration) {
                // Rotation animation (360 degrees)
                float rotation = (t / animationDuration) * 360.0f;
                creditsImageSprite_.setRotation(rotation);
                
                // Scale animation (1.0 to 1.2 and back to 1.0)
                float scale = 1.0f + 0.2f * std::sin(t / animationDuration * 3.14159f);
                const sf::Vector2u textureSize = creditsImageTexture_.getSize();
                float baseScale = (windowSize_.x * 0.22f) / 530.0f;
                creditsImageSprite_.setScale(baseScale * scale, baseScale * scale);
            } else {
                // Reset animation
                creditsImageClicked_ = false;
                float baseScale = (windowSize_.x * 0.22f) / 530.0f;
                creditsImageSprite_.setScale(baseScale, baseScale);
                creditsImageSprite_.setRotation(0.0f);
            }
        }

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
        
        // Draw victory overlay on top of pause button if game is over
        if (gameOver_ && (winnerId_ == 1 || winnerId_ == 2)) {
            float progress = 1.0f;
            if (victoryAnimationActive_) {
                const float t = victoryClock_.getElapsedTime().asSeconds();
                progress = std::min(t / kVictoryFadeDuration, 1.0f);
            }
            const sf::Uint8 overlayAlpha = static_cast<sf::Uint8>(
                std::min(kVictoryOverlayMaxAlpha, kVictoryOverlayMaxAlpha * progress));
            sf::RectangleShape pauseButtonOverlay;
            pauseButtonOverlay.setSize(sf::Vector2f(pauseButtonSprite_.getLocalBounds().width * pauseButtonSprite_.getScale().x,
                                                     pauseButtonSprite_.getLocalBounds().height * pauseButtonSprite_.getScale().y));
            pauseButtonOverlay.setPosition(pauseButtonSprite_.getPosition());
            pauseButtonOverlay.setFillColor(sf::Color(0, 0, 0, overlayAlpha));
            window.draw(pauseButtonOverlay);
        }

        // Draw pause menu if paused
        if (gamePaused_) {
            window.draw(pauseMenuOverlay_);
            if (showSettingsMenu_) {
                // Draw settings menu when open from pause
                window.draw(menuOverlay_);
                
                if (settingsMenuState_ == SettingsMenuState::Main) {
                    // Draw main settings menu
                    if (settingsMenuTexture_.getSize().x != 0) {
                        window.draw(settingsMenuSprite_);
                    }
                    
                    // Draw settings menu buttons
                    if (videoButtonTexture_.getSize().x != 0) {
                        window.draw(videoButtonSprite_);
                    }
                    if (audioButtonTexture_.getSize().x != 0) {
                        window.draw(audioButtonSprite_);
                    }
                    if (creditsButtonTexture_.getSize().x != 0) {
                        window.draw(creditsButtonSprite_);
                    }
                    if (settingsBackButtonTexture_.getSize().x != 0) {
                        window.draw(settingsBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Video) {
                    // Draw video menu
                    if (videoMenuTexture_.getSize().x != 0) {
                        window.draw(videoMenuSprite_);
                    }
                    // Draw quality label and selector
                    window.draw(qualityLabelSprite_);
                    window.draw(leftButtonSprite_);
                    window.draw(qualityDisplaySprite_);
                    window.draw(rightButtonSprite_);
                    // Draw fullscreen label and selector
                    window.draw(fullscreenLabelSprite_);
                    // Move left button to fullscreen row position
                    leftButtonSprite_.setPosition(fsLeftBtnPos_);
                    leftButtonSprite_.setScale(fsBtnScale_, fsBtnScale_);
                    window.draw(leftButtonSprite_);
                    window.draw(fullscreenDisplaySprite_);
                    // Move right button to fullscreen row position
                    rightButtonSprite_.setPosition(fsRightBtnPos_);
                    rightButtonSprite_.setScale(fsBtnScale_, fsBtnScale_);
                    window.draw(rightButtonSprite_);
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Audio) {
                    // Draw audio menu
                    if (audioMenuTexture_.getSize().x != 0) {
                        window.draw(audioMenuSprite_);
                    }
                    // Draw volume labels
                    window.draw(masterVolumeLabelSprite_);
                    window.draw(musicVolumeLabelSprite_);
                    window.draw(effectsVolumeLabelSprite_);
                    // Draw volume sliders
                    window.draw(masterVolumeSlider_.background);
                    window.draw(masterVolumeSlider_.handle);
                    window.draw(musicVolumeSlider_.background);
                    window.draw(musicVolumeSlider_.handle);
                    window.draw(sfxVolumeSlider_.background);
                    window.draw(sfxVolumeSlider_.handle);
                    // Draw volume icons
                    for (int i = 0; i < 3; ++i) {
                        window.draw(volumeIconSprites_[i]);
                    }
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
                } else if (settingsMenuState_ == SettingsMenuState::Credits) {
                    // Draw credits menu
                    if (creditsMenuTexture_.getSize().x != 0) {
                        window.draw(creditsMenuSprite_);
                    }
                    // Draw credits image
                    if (creditsImageTexture_.getSize().x != 0) {
                        window.draw(creditsImageSprite_);
                    }
                    // Draw back button at the bottom
                    if (submenuBackButtonTexture_.getSize().x != 0) {
                        window.draw(submenuBackButtonSprite_);
                    }
                }
            } else if (showHelp_) {
                if (!helpFrameTextures_.empty()) {
                    window.draw(helpFrameSprite_);
                }
                if (backToMenuTexture_.getSize().x != 0) {
                    window.draw(backToMenuSprite_);
                }
            } else {
                if (pauseMenuTexture_.getSize().x != 0) {
                    window.draw(pauseMenuSprite_);
                }
                // Draw pause menu buttons
                if (resumeButtonTexture_.getSize().x != 0) {
                    window.draw(resumeButtonSprite_);
                }
                if (restartButtonTexture_.getSize().x != 0) {
                    window.draw(restartButtonSprite_);
                }
                if (helpButtonTexture_.getSize().x != 0) {
                    window.draw(helpButtonSprite_);
                }
                if (pauseSettingsButtonTexture_.getSize().x != 0) {
                    window.draw(pauseSettingsButtonSprite_);
                }
                if (quitButtonTexture_.getSize().x != 0) {
                    window.draw(quitButtonSprite_);
                }
            }
        }

        window.display();
    }
    return 0;
}

void HexGameUI::updateVolumeIcon(int sliderIndex, float value) {
    // Determine which icon to show based on volume value
    // vol0.png: 0-25%, vol1.png: 26-50%, vol2.png: 51-75%, vol3.png: 76-100%
    int iconIndex = 3;  // Default to vol3
    
    if (value <= 25.0f) {
        iconIndex = 0;
    } else if (value <= 50.0f) {
        iconIndex = 1;
    } else if (value <= 75.0f) {
        iconIndex = 2;
    } else {
        iconIndex = 3;
    }
    
    if (sliderIndex >= 0 && sliderIndex < 3) {
        volumeIconSprites_[sliderIndex].setTexture(volumeIconTextures_[iconIndex]);
        
        // Update stored icon indices for reference
        if (sliderIndex == 0) {
            masterVolumeIcon_ = iconIndex;
        } else if (sliderIndex == 1) {
            musicVolumeIcon_ = iconIndex;
        } else if (sliderIndex == 2) {
            sfxVolumeIcon_ = iconIndex;
        }
    }
}

void HexGameUI::applyVolumeChanges() {
    // Apply volume changes with master volume as multiplier
    float masterMultiplier = masterVolume_ / 100.0f;
    
    menuMusic_.setVolume(musicVolume_ * masterMultiplier);
    gameMusic_.setVolume(musicVolume_ * masterMultiplier);
    gameOverSound_.setVolume(musicVolume_ * masterMultiplier);
    gameClickSound_.setVolume(sfxVolume_ * masterMultiplier);
}

void HexGameUI::saveVolumeConfig() {
    try {
        // Create config directory if it doesn't exist
        const char* configDir = "../config";
        struct stat st = {};
        if (stat(configDir, &st) == -1) {
            mkdir(configDir, 0755);
        }
        
        std::ofstream configFile(kConfigFilePath);
        if (configFile.is_open()) {
            configFile << masterVolume_ << "\n";
            configFile << musicVolume_ << "\n";
            configFile << sfxVolume_ << "\n";
            configFile.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving volume config: " << e.what() << std::endl;
    }
}

void HexGameUI::loadVolumeConfig() {
    try {
        std::ifstream configFile(kConfigFilePath);
        if (configFile.is_open()) {
            configFile >> masterVolume_;
            configFile >> musicVolume_;
            configFile >> sfxVolume_;
            configFile.close();
            
            // Clamp values to valid range
            masterVolume_ = std::max(0.0f, std::min(100.0f, masterVolume_));
            musicVolume_ = std::max(0.0f, std::min(100.0f, musicVolume_));
            sfxVolume_ = std::max(0.0f, std::min(100.0f, sfxVolume_));
            
            // Update slider values
            masterVolumeSlider_.value = masterVolume_;
            musicVolumeSlider_.value = musicVolume_;
            sfxVolumeSlider_.value = sfxVolume_;
            
            // Update slider handle positions
            float sliderWidth = windowSize_.x * 0.16f;
            float handleSize = windowSize_.y * 0.06f;
            
            float masterHandleX = masterVolumeSlider_.minX + (masterVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
            masterVolumeSlider_.handle.setPosition(masterHandleX, masterVolumeSlider_.handle.getPosition().y);
            
            float musicHandleX = musicVolumeSlider_.minX + (musicVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
            musicVolumeSlider_.handle.setPosition(musicHandleX, musicVolumeSlider_.handle.getPosition().y);
            
            float sfxHandleX = sfxVolumeSlider_.minX + (sfxVolumeSlider_.value / 100.0f) * sliderWidth - handleSize / 2.0f;
            sfxVolumeSlider_.handle.setPosition(sfxHandleX, sfxVolumeSlider_.handle.getPosition().y);
            
            // Update volume icons
            updateVolumeIcon(0, masterVolumeSlider_.value);
            updateVolumeIcon(1, musicVolumeSlider_.value);
            updateVolumeIcon(2, sfxVolumeSlider_.value);
            
            // Apply audio changes
            applyVolumeChanges();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading volume config: " << e.what() << std::endl;
    }
}
