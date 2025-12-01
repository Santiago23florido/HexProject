#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "DataCollector.hpp"
#include "GameRunner.hpp"
#include "MoveStrategy.hpp"
#include "Serializer.hpp"

int main(int argc, char** argv) {
    int games = 10;
    int boardSize = 7;
    std::string outputPath = "selfplay_data.jsonl";

    if (argc > 1) games = std::atoi(argv[1]);
    if (argc > 2) boardSize = std::atoi(argv[2]);
    if (argc > 3) outputPath = argv[3];

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    RandomStrategy p1;
    RandomStrategy p2;
    GameRunner runner(boardSize, p1, p2);
    DataCollector collector;

    for (int i = 0; i < games; ++i) {
        int winner = runner.playOne(collector);
        std::cout << "Game " << (i + 1) << "/" << games << " finished. Winner: " << winner << "\n";
    }

    bool ok = Serializer::writeJsonl(collector.samples(), outputPath);
    if (ok) {
        std::cout << "Wrote " << collector.samples().size() << " samples to " << outputPath << "\n";
    } else {
        std::cerr << "Failed to write output to " << outputPath << "\n";
        return 1;
    }

    return 0;
}
