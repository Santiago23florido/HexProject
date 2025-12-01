#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "DataCollector.hpp"
#include "GameRunner.hpp"
#include "MoveStrategy.hpp"
#include "Serializer.hpp"

int main(int argc, char** argv) {
    int games = 100;
    int sims = 10;
    const int minSims = 4;
    const int maxSims = 20;
    std::string outputPath = "selfplay_data.jsonl";

    if (argc > 1) games = std::atoi(argv[1]);
    if (argc > 2) sims = std::atoi(argv[2]);
    if (argc > 3) outputPath = argv[3];

    if (sims < minSims) sims = minSims;
    if (sims > maxSims) sims = maxSims;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Run self-play for board sizes 4..11 (inclusive)
    std::vector<int> sizes;
    for (int n = 4; n <= 8; ++n) sizes.push_back(n);

    for (int boardSize : sizes) {
        DataCollector collector;
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < games; ++i) {
            int simsThisGame = minSims + (std::rand() % (maxSims - minSims + 1));
            MonteCarloStrategy p1(simsThisGame);
            MonteCarloStrategy p2(simsThisGame);
            GameRunner runner(boardSize, p1, p2);

            int winner = runner.playOne(collector);
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            double avgPerGame = (i + 1) > 0 ? static_cast<double>(elapsed) / static_cast<double>(i + 1) : 0.0;
            double remaining = avgPerGame * static_cast<double>(games - i - 1);

            std::cout << "[N=" << boardSize << "] Game " << (i + 1) << "/" << games
                      << " winner: " << winner
                      << " | sims per move: " << simsThisGame
                      << " | elapsed: " << elapsed << "s"
                      << " | est. remaining: " << static_cast<long long>(remaining) << "s"
                      << "\n";
        }

        // Write one file per board size
        std::string outPath = outputPath;
        if (outputPath.find(".jsonl") != std::string::npos) {
            auto pos = outputPath.find(".jsonl");
            outPath = outputPath.substr(0, pos) + "_N" + std::to_string(boardSize) + ".jsonl";
        } else {
            outPath = outputPath + "_N" + std::to_string(boardSize);
        }

        bool ok = Serializer::writeJsonl(collector.samples(), outPath);
        if (ok) {
            std::cout << "[N=" << boardSize << "] Wrote " << collector.samples().size()
                      << " samples to " << outPath
                      << " with sims per move randomized in [" << minSims << "," << maxSims << "]\n";
        } else {
            std::cerr << "[N=" << boardSize << "] Failed to write output to " << outPath << "\n";
            return 1;
        }
    }

    return 0;
}
