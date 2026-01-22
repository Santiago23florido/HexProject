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
#include <stdexcept>

#include "RLTrainer.hpp"
#include "DataCollector.hpp"
#include "GameRunner.hpp"
#include "core/MoveStrategy.hpp"
#include "Serializer.hpp"

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

struct CliOptions {
    bool selfplayTrain{false};
    int games{1000};
    int minDepth{10};
    int maxDepth{20};
    int minPairs{2};
    int maxPairs{6};
    int timeLimitMs{10000};
    std::string outputPath{"selfplay_data.jsonl"};

    int trainGames{200};
    std::size_t bufferSize{50000};
    std::size_t batchSize{256};
    int updatesPerGame{1};
    float alpha{0.2f};
    std::string device{"cuda"};
    bool exportTs{false};
    int reportEvery{10};
    int checkpointEvery{10};
    int snapshotEvery{20};
    float probFrozen{0.3f};
    float probHeuristic{0.0f};
    std::size_t maxFrozen{5};
    int selfplayThreads{1};
    bool randomFirstMove{true};
};

bool hasFlagStyleArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) return true;
    }
    return false;
}

bool parseFlagOptions(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--selfplay-train") {
            opts.selfplayTrain = true;
        } else if (arg == "--train-games" && i + 1 < argc) {
            opts.trainGames = std::atoi(argv[++i]);
        } else if (arg == "--buffer-size" && i + 1 < argc) {
            opts.bufferSize = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            opts.batchSize = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--updates-per-game" && i + 1 < argc) {
            opts.updatesPerGame = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            opts.alpha = std::stof(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts.device = argv[++i];
        } else if (arg == "--report-every" && i + 1 < argc) {
            opts.reportEvery = std::atoi(argv[++i]);
        } else if (arg == "--checkpoint-every" && i + 1 < argc) {
            opts.checkpointEvery = std::atoi(argv[++i]);
        } else if (arg == "--snapshot-every" && i + 1 < argc) {
            opts.snapshotEvery = std::atoi(argv[++i]);
        } else if (arg == "--prob-frozen" && i + 1 < argc) {
            opts.probFrozen = std::stof(argv[++i]);
        } else if (arg == "--prob-heuristic" && i + 1 < argc) {
            opts.probHeuristic = std::stof(argv[++i]);
        } else if (arg == "--max-frozen" && i + 1 < argc) {
            opts.maxFrozen = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--selfplay-threads" && i + 1 < argc) {
            opts.selfplayThreads = std::atoi(argv[++i]);
        } else if (arg == "--export-ts") {
            opts.exportTs = true;
        } else if (arg == "--random-first-move") {
            opts.randomFirstMove = true;
        } else if (arg == "--no-random-first-move") {
            opts.randomFirstMove = false;
        } else if (arg == "--games" && i + 1 < argc) {
            opts.games = std::atoi(argv[++i]);
        } else if (arg == "--min-depth" && i + 1 < argc) {
            opts.minDepth = std::atoi(argv[++i]);
        } else if (arg == "--max-depth" && i + 1 < argc) {
            opts.maxDepth = std::atoi(argv[++i]);
        } else if (arg == "--min-pairs" && i + 1 < argc) {
            opts.minPairs = std::atoi(argv[++i]);
        } else if (arg == "--max-pairs" && i + 1 < argc) {
            opts.maxPairs = std::atoi(argv[++i]);
        } else if (arg == "--time-limit-ms" && i + 1 < argc) {
            opts.timeLimitMs = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opts.outputPath = argv[++i];
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliOptions opts;
        const bool useFlags = hasFlagStyleArgs(argc, argv);
        if (useFlags) {
            parseFlagOptions(argc, argv, opts);
        } else {
            if (argc > 1) opts.games = std::atoi(argv[1]);
            if (argc > 2) opts.minDepth = std::atoi(argv[2]);
            if (argc > 3) opts.maxDepth = std::atoi(argv[3]);
            if (argc > 4) opts.outputPath = argv[4];
            if (argc > 5) opts.minPairs = std::atoi(argv[5]);
            if (argc > 6) opts.maxPairs = std::atoi(argv[6]);
            if (argc > 7) opts.timeLimitMs = std::atoi(argv[7]);
        }

        if (opts.minDepth < 1) opts.minDepth = 1;
        if (opts.maxDepth < opts.minDepth) opts.maxDepth = opts.minDepth;
        if (opts.minPairs < 1) opts.minPairs = 1;
        if (opts.maxPairs < opts.minPairs) opts.maxPairs = opts.minPairs;

        if (opts.selfplayTrain) {
            RLConfig cfg;
            cfg.boardSize = 7;
            cfg.minDepth = opts.minDepth;
            cfg.maxDepth = opts.maxDepth;
            cfg.timeLimitMs = opts.timeLimitMs;
            cfg.trainGames = opts.trainGames;
            cfg.bufferSize = opts.bufferSize;
            cfg.batchSize = opts.batchSize;
            cfg.updatesPerGame = opts.updatesPerGame;
            cfg.alpha = opts.alpha;
            cfg.device = opts.device;
            cfg.exportTs = opts.exportTs;
            cfg.reportEvery = opts.reportEvery;
            cfg.checkpointEvery = opts.checkpointEvery;
            cfg.snapshotEvery = opts.snapshotEvery;
            cfg.probFrozen = opts.probFrozen;
            cfg.probHeuristic = opts.probHeuristic;
            cfg.maxFrozen = opts.maxFrozen;
            cfg.selfplayThreads = opts.selfplayThreads;
            cfg.randomFirstMove = opts.randomFirstMove;
            RLTrainer trainer(cfg);
            return trainer.run();
        }

        const int flushEveryGames = 20; // write to disk in small batches to avoid high RAM use

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
            std::uniform_int_distribution<int> depthDist(opts.minDepth, opts.maxDepth);

            // Build per-board-size output path once, then append batches.
            std::string outPath = opts.outputPath;
            if (opts.outputPath.find(".jsonl") != std::string::npos) {
                auto pos = opts.outputPath.find(".jsonl");
                outPath = opts.outputPath.substr(0, pos) + "_N" + std::to_string(boardSize) + ".jsonl";
            } else {
                outPath = opts.outputPath + "_N" + std::to_string(boardSize);
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

            for (int i = 0; i < opts.games; ++i) {
                int depthP1 = depthDist(rng);
                int depthP2 = depthDist(rng);

                NegamaxHeuristicStrategy p1(depthP1, opts.timeLimitMs);
                NegamaxHeuristicStrategy p2(depthP2, opts.timeLimitMs);
                int initPairs = 0;
                Board startBoard = randomStartingBoard(boardSize, opts.minPairs, opts.maxPairs, rng, initPairs);

                GameRunner runner(boardSize, p1, p2);

                int winner = runner.playOne(collector, &startBoard);
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
                double avgPerGame = (i + 1) > 0 ? static_cast<double>(elapsed) / static_cast<double>(i + 1) : 0.0;
                double remaining = avgPerGame * static_cast<double>(opts.games - i - 1);

                std::cout << "[N=" << boardSize << "] Game " << (i + 1) << "/" << opts.games
                          << " winner: " << winner
                          << " | depths: P1=" << depthP1 << " P2=" << depthP2
                          << " | init stones/player: " << initPairs
                          << " | time cap: " << opts.timeLimitMs << "ms"
                          << " | elapsed: " << elapsed << "s"
                          << " | est. remaining: " << static_cast<long long>(remaining) << "s"
                          << "\n";

                if ((i + 1) % flushEveryGames == 0) {
                    if (!flushSamples(std::to_string(i + 1) + " games")) return 1;
                }
            }

            if (!flushSamples("final batch")) return 1;
            std::cout << "[N=" << boardSize << "] Completed " << opts.games << " games, wrote "
                      << totalSamplesWritten << " samples to " << outPath
                      << " with depths randomized in [" << opts.minDepth << "," << opts.maxDepth << "]\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
