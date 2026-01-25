#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <memory>
#include <random>
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
        bool preferCuda);

    ~HexGameUI();

    int run();

private:
    enum class UIScreen { Start, PlayerSelect, Game };

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
    bool loadPlayerSelectTextures();
    bool loadVictoryTextures();
    bool loadPauseTextures();
    void buildLayout();
    void updateTileColors();
    void updateHelpFrameSprite();
    bool applyMove(int moveIdx);
    int pickTileIndex(const sf::Vector2f& pos) const;
    bool pointInHex(const sf::Vector2f& pos, const sf::Vector2f& center) const;
    void switchMusic(bool playGameMusic);
    void updateWindowTitle(sf::RenderWindow& window) const;
    void updateHover(const sf::RenderWindow& window);
    void printBoardStatus() const;
    void resetGame();
    void setPlayer2ModeIndex(int index);
    void advancePlayer2Mode();
    void applyAiDifficulty();
    void updateDifficultyText();
    void applyBoardSize(int newSize);
    void updateBoardSizeText();

    std::string texturePath_;
    std::string backgroundPath_;
    std::string player1Path_;
    std::string player2Path_;
    std::string startPagePath_;
    std::string startButtonPath_;
    std::string startTitlePath_;
    std::string playerSelectPagePath_;
    std::string playerStartButtonPath_;
    std::string nextTypeButtonPath_;
    std::string plusNButtonPath_;
    std::string minusNButtonPath_;
    std::string humanLabelPath_;
    std::string player2HumanLabelPath_;
    std::string player2GnnLabelPath_;
    std::string player2HeuristicLabelPath_;
    std::string player1WinPath_;
    std::string player2WinPath_;
    std::string modelPath_;
    int boardSize_ = 0;
    float tileScale_ = 1.0f;
    float scaleFactor_ = 2.0f;
    bool useGnnAi_ = false;
    bool player2IsHuman_ = false;
    bool preferCuda_ = false;
    int player2ModeIndex_ = 2;
    int aiDifficulty_ = 3;
    int aiMaxDepth_ = 3;
    int aiRandomEvery_ = 0;
    int aiMoveCount_ = 0;
    std::mt19937 aiRng_;

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
    sf::Texture playerSelectPageTexture_;
    sf::Sprite playerSelectPageSprite_;
    sf::Texture playerStartButtonTexture_;
    sf::Sprite playerStartButtonSprite_;
    sf::Texture nextTypeButtonTexture_;
    sf::Sprite nextTypeButtonSprite_;
    sf::Texture plusNButtonTexture_;
    sf::Texture minusNButtonTexture_;
    sf::Sprite plusNButtonSprite_;
    sf::Sprite minusNButtonSprite_;
    sf::Texture humanLabelTexture_;
    sf::Sprite humanLabelSprite_;
    sf::Texture player2HumanLabelTexture_;
    sf::Sprite player2HumanLabelSprite_;
    sf::Texture player2GnnLabelTexture_;
    sf::Sprite player2GnnLabelSprite_;
    sf::Texture player2HeuristicLabelTexture_;
    sf::Sprite player2HeuristicLabelSprite_;
    sf::Font startFont_;
    sf::Text startHintText_;
    sf::RectangleShape startHintBox_;

    sf::Texture settingsButtonTexture_;
    sf::Sprite settingsButtonSprite_; 

    enum class SettingsMenuState { Main, Video, Audio, Credits };
    SettingsMenuState settingsMenuState_ = SettingsMenuState::Main;
    bool showSettingsMenu_ = false;   
    sf::RectangleShape menuBackground_; 
    sf::RectangleShape menuOverlay_;  
    
    // Settings menu texture and sprite
    sf::Texture settingsMenuTexture_;
    sf::Sprite settingsMenuSprite_;
    // Settings menu buttons
    sf::Texture videoButtonTexture_;
    sf::Sprite videoButtonSprite_;
    sf::Texture audioButtonTexture_;
    sf::Sprite audioButtonSprite_;
    sf::Texture creditsButtonTexture_;
    sf::Sprite creditsButtonSprite_;
    sf::Texture settingsBackButtonTexture_;
    sf::Sprite settingsBackButtonSprite_;
    
    // Video and Audio submenu textures and sprites
    sf::Texture videoMenuTexture_;
    sf::Sprite videoMenuSprite_;
    sf::Texture audioMenuTexture_;
    sf::Sprite audioMenuSprite_;
    sf::Texture creditsMenuTexture_;
    sf::Sprite creditsMenuSprite_;
    // Back button for submenus
    sf::Texture submenuBackButtonTexture_;
    sf::Sprite submenuBackButtonSprite_;

    sf::RectangleShape aiConfigBox_;    
    sf::Text aiConfigText_;             
    sf::Text difficultyText_;
    sf::Text boardSizeText_;
    sf::Texture boardSizeLabelTexture_;
    sf::Sprite boardSizeLabelSprite_;
    
    // Options in the settings menu
    sf::Text menuTitleText_;            
    sf::Text aiOptionText_;             
    sf::Text hardwareInfoText_;         
    sf::Text resolutionText_;

    // Pause button and menu
    bool gamePaused_ = false;
    bool showHelp_ = false;
    sf::Texture pauseButtonTexture_;
    sf::Sprite pauseButtonSprite_;
    sf::Texture pauseMenuTexture_;
    sf::Sprite pauseMenuSprite_;
    sf::RectangleShape pauseMenuOverlay_;
    
    // Pause menu buttons
    sf::Texture resumeButtonTexture_;
    sf::Sprite resumeButtonSprite_;
    sf::Texture helpButtonTexture_;
    sf::Sprite helpButtonSprite_;
    sf::Sprite helpSelectButtonSprite_;
    sf::Texture pauseSettingsButtonTexture_;
    sf::Sprite pauseSettingsButtonSprite_;
    std::vector<std::unique_ptr<sf::Texture>> helpFrameTextures_;
    std::size_t helpFrameIndex_ = 0;
    sf::Clock helpFrameClock_;
    sf::Sprite helpFrameSprite_;
    sf::Texture backToMenuTexture_;
    sf::Sprite backToMenuSprite_;


    sf::Clock startScreenClock_;
    sf::Vector2f startHintBoxBasePos_{0.0f, 0.0f};
    sf::Vector2f startHintTextBasePos_{0.0f, 0.0f};
    sf::Texture player1WinTexture_;
    sf::Texture player2WinTexture_;
    sf::Sprite player1WinSprite_;
    sf::Sprite player2WinSprite_;
    // Victory screen buttons
    sf::Texture restartButtonTexture_;
    sf::Texture quitButtonTexture_;
    sf::Sprite restartButtonSprite_;
    sf::Sprite quitButtonSprite_;
    sf::RectangleShape victoryOverlay_;
    sf::Clock victoryClock_;
    sf::Vector2u textureSize_{0, 0};
    float tileWidth_ = 0.0f;
    float tileHeight_ = 0.0f;
    sf::Vector2u windowSize_{0, 0};
    sf::Vector2u baseWindowSize_{0, 0};
    float baseTileScale_ = 1.0f;

    std::vector<Tile> tiles_;
    std::string error_;
    UIScreen screen_ = UIScreen::Start;
    bool playerSelectEnabled_ = true;
    bool startFontLoaded_ = false;
    bool victoryAnimationActive_ = false;

    //Audio
    bool initAudio();
    //Music
    sf::Music menuMusic_;
    sf::Music gameMusic_;
    //Sound Effects
    sf::SoundBuffer gameOverBuffer_;
    sf::Sound gameOverSound_;
    sf::SoundBuffer clickBuffer_;
    sf::Sound gameClickSound_;
    static constexpr float MUSIC_VOLUME = 50.0f;
    static constexpr float SFX_VOLUME = 80.0f;
};
