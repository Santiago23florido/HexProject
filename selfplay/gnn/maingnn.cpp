#include <chrono>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>

#include "gnn/DataCollector.hpp"
#include "gnn/GameRunner.hpp"
#include "core/MoveStrategy.hpp"
#include "gnn/Serializer.hpp"

namespace {

// Build a starting board with an equal number of random stones for both players.
Board randomStartingBoard(int boardSize, int minPairs, int maxPairs, std::mt19937& rng, int& outPairs) {
    const int totalCells = boardSize * boardSize;
    int safeMax = std::max(1, std::min(maxPairs, totalCells / 2));
    int safeMin = std::max(1, std::min(minPairs, safeMax));
    std::uniform_int_distribution<int> pairsDist(safeMin, safeMax);
    std::vector<int> cells(totalCells);

    Board last(boardSize);
    for (int attempt = 0; attempt < 10; ++attempt) {
        Board b(boardSize);
        std::iota(cells.begin(), cells.end(), 0);
        std::shuffle(cells.begin(), cells.end(), rng);

        int pairs = pairsDist(rng);
        for (int i = 0; i < pairs * 2; ++i) {
            int player = (i % 2) + 1;
            b.place(cells[i], player);
        }

        auto hasConnectedChain = [&](int player, int minLen) {
            static const int evenDirs[6][2] = {{-1, -1}, {-1, 0}, {0, -1}, {0, 1}, {1, -1}, {1, 0}};
            static const int oddDirs[6][2] = {{-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, 0}, {1, 1}};
            std::vector<int> visited(totalCells, 0);
            for (int idx = 0; idx < totalCells; ++idx) {
                int r = idx / boardSize;
                int c = idx % boardSize;
                if (b.board[r][c] != player || visited[idx]) continue;
                int comp = 0;
                std::queue<int> q;
                q.push(idx);
                visited[idx] = 1;
                while (!q.empty()) {
                    int cur = q.front();
                    q.pop();
                    comp++;
                    int cr = cur / boardSize;
                    int cc = cur % boardSize;
                    const int (*dirs)[2] = (cr % 2 == 0) ? evenDirs : oddDirs;
                    for (int d = 0; d < 6; ++d) {
                        int nr = cr + dirs[d][0];
                        int nc = cc + dirs[d][1];
                        if (nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize) continue;
                        if (b.board[nr][nc] != player) continue;
                        int nidx = nr * boardSize + nc;
                        if (!visited[nidx]) {
                            visited[nidx] = 1;
                            q.push(nidx);
                        }
                    }
                }
                if (comp >= minLen) return true;
            }
            return false;
        };

        GameState st(b, 1);
        bool p1Chain = hasConnectedChain(1, 2);
        bool p2Chain = hasConnectedChain(2, 2);
        if (st.Winner() == 0 && p1Chain && p2Chain) {
            outPairs = pairs;
            return b;
        }
        last = std::move(b);
    }
    outPairs = safeMin;
    return last; // fallback even if an early win was drawn
}

} // namespace

int main(int argc, char** argv) {
    int games = 1000;            // default number of games (can be overridden by argv)
    int minDepth = 10;
    int maxDepth = 20;
    int minPairs = 2;             // starting random stone pairs per player (curriculum control)
    int maxPairs = 6;
    int timeLimitMs = 10000;       // per-move time cap to keep self-play fast
    const int flushEveryGames = 20; // write to disk in small batches to avoid high RAM use
    std::string outputPath = "selfplay_data.jsonl";

    if (argc > 1) games = std::atoi(argv[1]);
    if (argc > 2) minDepth = std::atoi(argv[2]);
    if (argc > 3) maxDepth = std::atoi(argv[3]);
    if (argc > 4) outputPath = argv[4];
    if (argc > 5) minPairs = std::atoi(argv[5]);
    if (argc > 6) maxPairs = std::atoi(argv[6]);
    if (argc > 7) timeLimitMs = std::atoi(argv[7]);

    if (minDepth < 1) minDepth = 1;
    if (maxDepth < minDepth) maxDepth = minDepth;
    if (minPairs < 1) minPairs = 1;
    if (maxPairs < minPairs) maxPairs = minPairs;

    // Seed both rand() (used inside strategies) and a dedicated RNG for depth selection.
    unsigned seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count() ^
        static_cast<unsigned>(std::random_device{}()));
    std::srand(seed);
    std::mt19937 rng(seed);

    // Run self-play only for board size 7 (training focus)
    std::vector<int> sizes = {7};

    for (int boardSize : sizes) {
        DataCollector collector;
        auto start = std::chrono::steady_clock::now();
        std::uniform_int_distribution<int> depthDist(minDepth, maxDepth);

        // Build per-board-size output path once, then append batches.
        std::string outPath = outputPath;
        if (outputPath.find(".jsonl") != std::string::npos) {
            auto pos = outputPath.find(".jsonl");
            outPath = outputPath.substr(0, pos) + "_N" + std::to_string(boardSize) + ".jsonl";
        } else {
            outPath = outputPath + "_N" + std::to_string(boardSize);
        }

        bool firstChunk = true;
        size_t totalSamplesWritten = 0;
        auto flushSamples = [&](const std::string& reason) -> bool {
            auto chunk = collector.consumeSamples();
            if (chunk.empty()) return true;

            bool ok = Serializer::writeJsonl(chunk, outPath, /*append=*/!firstChunk);
            if (!ok) {
                std::cerr << "[N=" << boardSize << "] Failed to write output to " << outPath << "\n";
                return false;
            }

            firstChunk = false;
            totalSamplesWritten += chunk.size();
            std::cout << "[N=" << boardSize << "] Flushed " << chunk.size()
                      << " samples (" << totalSamplesWritten << " total) to " << outPath
                      << " after " << reason << "\n";
            return true;
        };

        for (int i = 0; i < games; ++i) {
            int depthP1 = depthDist(rng);
            int depthP2 = depthDist(rng);

            NegamaxHeuristicStrategy p1(depthP1, timeLimitMs);
            NegamaxHeuristicStrategy p2(depthP2, timeLimitMs);
            int initPairs = 0;
            Board startBoard = randomStartingBoard(boardSize, minPairs, maxPairs, rng, initPairs);

            GameRunner runner(boardSize, p1, p2);

            int winner = runner.playOne(collector, &startBoard);
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            double avgPerGame = (i + 1) > 0 ? static_cast<double>(elapsed) / static_cast<double>(i + 1) : 0.0;
            double remaining = avgPerGame * static_cast<double>(games - i - 1);

            std::cout << "[N=" << boardSize << "] Game " << (i + 1) << "/" << games
                      << " winner: " << winner
                      << " | depths: P1=" << depthP1 << " P2=" << depthP2
                      << " | init stones/player: " << initPairs
                      << " | time cap: " << timeLimitMs << "ms"
                      << " | elapsed: " << elapsed << "s"
                      << " | est. remaining: " << static_cast<long long>(remaining) << "s"
                      << "\n";

            if ((i + 1) % flushEveryGames == 0) {
                if (!flushSamples(std::to_string(i + 1) + " games")) return 1;
            }
        }

        if (!flushSamples("final batch")) return 1;
        std::cout << "[N=" << boardSize << "] Completed " << games << " games, wrote "
                  << totalSamplesWritten << " samples to " << outPath
                  << " with depths randomized in [" << minDepth << "," << maxDepth << "]\n";
    }

    return 0;
}
