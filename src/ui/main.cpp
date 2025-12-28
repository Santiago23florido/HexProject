#include "ui/HexGameUI.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    const std::string modelPath = "../scripts/models/hex_value_ts.pt";

    char modeChoice = 'h';
    std::cout << "Jugar contra IA heuristica (h) o IA GNN (g)? [h]: ";
    if (!(std::cin >> modeChoice)) {
        modeChoice = 'h';
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    bool useGnnAi = (modeChoice == 'g' || modeChoice == 'G');

    const int boardSize = 7;
    const float tileScale = 0.1f;

    HexGameUI game(
        "../assets/hex1.png",
        "../assets/background.png",
        "../assets/Player 1.png",
        "../assets/Player 2.png",
        boardSize,
        tileScale,
        useGnnAi,
        modelPath);
    return game.run();
}
