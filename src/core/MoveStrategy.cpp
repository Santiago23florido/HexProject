#include "core/GameState.hpp"
#include "core/Board.hpp"
#include "core/Cube.hpp"
#include "core/MoveStrategy.hpp"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <queue>
#include <deque>
#include <filesystem>
#include "gnn/Graph.hpp"

namespace {

std::string defaultModelPath() {
    return "scripts/models/hex_value_ts.pt";
}


std::string resolveModelPath(const std::string& hint) {
    namespace fs = std::filesystem;
    const fs::path hinted = hint.empty() ? fs::path(defaultModelPath()) : fs::path(hint);
    std::vector<fs::path> candidates;

    
    candidates.push_back(hinted);

    
    fs::path cwd = fs::current_path();
    for (int up = 0; up < 4; ++up) {
        fs::path base = cwd;
        for (int i = 0; i < up && base.has_parent_path(); ++i) {
            base = base.parent_path();
        }
        candidates.push_back(base / hinted);
        candidates.push_back(base / defaultModelPath());
    }

    for (const auto& cand : candidates) {
        if (!cand.empty() && fs::exists(cand)) {
            return cand.lexically_normal().string();
        }
    }

    
    return hinted.lexically_normal().string();
}

} 


static const Graph& getPlainGraph(int N) {
    static std::unordered_map<int, Graph> cache;
    auto it = cache.find(N);
    if (it != cache.end()) return it->second;
    Graph g = buildHexGraph(N, true);
    auto res = cache.emplace(N, std::move(g));
    return res.first->second;
}

static Board buildBoard(const std::vector<int>& linear) {
    const int n = static_cast<int>(std::sqrt(linear.size()));
    Board b(n);
    for (int idx = 0; idx < static_cast<int>(linear.size()); ++idx) {
        if (linear[idx] != 0) {
            b.place(idx, linear[idx]);
        }
    }
    return b;
}

static int countImmediateWins(const Board& base, const std::vector<int>& moves, int player) {
    int wins = 0;
    for (int m : moves) {
        Board copy(base);
        copy.place(m, player);
        int nextPlayer = (player == 1 ? 2 : 1);
        GameState child(copy, nextPlayer);
        if (child.Winner() == player) {
            wins++;
            if (wins > 1) break; 
        }
    }
    return wins;
}

static int findImmediateWinningMove(const GameState& state, int player) {
    const auto moves = state.GetAvailableMoves();
    if (moves.empty()) return -1;
    Board base = buildBoard(state.LinearBoard());
    for (int m : moves) {
        Board copy(base);
        copy.place(m, player);
        int nextPlayer = (player == 1 ? 2 : 1);
        GameState child(copy, nextPlayer);
        if (child.Winner() == player) {
            return m;
        }
    }
    return -1;
}

static int boundaryDistance(const std::vector<int>& linear, int player) {
    const int N = static_cast<int>(std::sqrt(linear.size()));
    if (N <= 0) return 0;
    const Graph& g = getPlainGraph(N);
    const int opp = (player == 1 ? 2 : 1);
    const int total = N * N;
    const int INF = total * 2;

    std::vector<int> dist(total, INF);
    std::deque<int> dq;

    auto push_start = [&](int idx) {
        if (idx < 0 || idx >= total) return;
        if (linear[idx] == opp) return;
        int cost = (linear[idx] == player) ? 0 : 1;
        if (cost < dist[idx]) {
            dist[idx] = cost;
            if (cost == 0) dq.push_front(idx);
            else dq.push_back(idx);
        }
    };

    if (player == 1) {
        for (int r = 0; r < N; ++r) {
            push_start(r * N); 
        }
    } else {
        for (int c = 0; c < N; ++c) {
            push_start(c); 
        }
    }

    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();
        int du = dist[u];
        if (player == 1) {
            if ((u % N) == N - 1) return du;
        } else {
            if ((u / N) == N - 1) return du;
        }
        for (int v : g.adj[u]) {
            if (v >= total) continue;
            if (linear[v] == opp) continue;
            int w = (linear[v] == player) ? 0 : 1;
            int nd = du + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                if (w == 0) dq.push_front(v);
                else dq.push_back(v);
            }
        }
    }
    return INF;
}

static int libertiesScore(const std::vector<int>& linear, int player) {
    const int N = static_cast<int>(std::sqrt(linear.size()));
    const Graph& g = getPlainGraph(N);
    int score = 0;
    for (int idx = 0; idx < N * N; ++idx) {
        if (linear[idx] != player) continue;
        for (int nb : g.adj[idx]) {
            if (nb < N * N && linear[nb] == 0) {
                score++;
            }
        }
    }
    return score;
}

static int bridgeScore(const std::vector<int>& linear, int player) {
    const int N = static_cast<int>(std::sqrt(linear.size()));
    const Graph& g = getPlainGraph(N);
    int score = 0;
    for (int idx = 0; idx < N * N; ++idx) {
        if (linear[idx] != 0) continue;
        int neighbors = 0;
        for (int nb : g.adj[idx]) {
            if (nb < N * N && linear[nb] == player) {
                neighbors++;
            }
        }
        if (neighbors >= 2) {
            score += neighbors - 1;
        }
    }
    return score;
}

static int centerScore(const std::vector<int>& linear, int player) {
    const int N = static_cast<int>(std::sqrt(linear.size()));
    const float center = static_cast<float>(N - 1) / 2.0f;
    int self = 0, opp = 0;
    for (int idx = 0; idx < N * N; ++idx) {
        int val = linear[idx];
        if (val == 0) continue;
        int r = idx / N;
        int c = idx % N;
        float dist = std::abs(r - center) + std::abs(c - center);
        int weight = static_cast<int>(std::round((N - dist)));
        if (val == player) self += weight;
        else self -= weight;
    }
    return self;
}

struct HeuristicFeatures {
    int N{0};
    int distSelf{0};
    int distOpp{0};
    int libsSelf{0};
    int libsOpp{0};
    int bridgesSelf{0};
    int bridgesOpp{0};
    int center{0};
};

static HeuristicFeatures computeHeuristicFeatures(const GameState& state, int playerId) {
    const auto linear = state.LinearBoard();
    const int N = static_cast<int>(std::sqrt(linear.size()));
    const int opp = (playerId == 1 ? 2 : 1);

    int distSelf = boundaryDistance(linear, playerId);
    int distOpp = boundaryDistance(linear, opp);
    const int maxDist = std::max(1, N * N);
    distSelf = std::min(distSelf, maxDist);
    distOpp = std::min(distOpp, maxDist);

    int libsSelf = libertiesScore(linear, playerId);
    int libsOpp = libertiesScore(linear, opp);

    int bridgesSelf = bridgeScore(linear, playerId);
    int bridgesOpp = bridgeScore(linear, opp);

    int center = centerScore(linear, playerId);

    return {
        N,
        distSelf,
        distOpp,
        libsSelf,
        libsOpp,
        bridgesSelf,
        bridgesOpp,
        center,
    };
}

static int heuristicEval(const GameState& state, int playerId) {
    const HeuristicFeatures feat = computeHeuristicFeatures(state, playerId);
    if (feat.distSelf == 0) return 90000;
    if (feat.distOpp == 0) return -90000;

    const int pathWeight = 600 + feat.N * 10;
    const int threatWeight = 12000 + feat.N * 300;
    const int bridgeWeight = 6 + feat.N;
    const int libertyWeight = 2;
    const int centerWeight = 1;

    int score = 0;
    score += (feat.distOpp - feat.distSelf) * pathWeight;
    if (feat.distSelf == 1) score += threatWeight;
    if (feat.distOpp == 1) score -= threatWeight;
    score += (feat.bridgesSelf - feat.bridgesOpp) * bridgeWeight;
    score += (feat.libsSelf - feat.libsOpp) * libertyWeight;
    score += feat.center * centerWeight;
    return score;
}


int RandomStrategy::select(const GameState& state, int playerId) {
    std::vector<int> moves = state.GetAvailableMoves();
    if (moves.empty()) return -1;
    return moves[std::rand() % moves.size()];
}

MonteCarloStrategy::MonteCarloStrategy(int sims) : SimulationsPerMove(sims) {}


int MonteCarloStrategy::simulate(GameState state, int currentPlayer) {
    const auto linear = state.LinearBoard();
    if (linear.empty()) return 0;
    const int n = static_cast<int>(std::sqrt(linear.size()));
    Board board(n);
    for (int idx = 0; idx < static_cast<int>(linear.size()); ++idx) {
        if (linear[idx] != 0) {
            board.place(idx, linear[idx]);
        }
    }
    GameState rollout(board, currentPlayer);
    while (true) {
        int w = rollout.Winner();
        if (w != 0) return w;

        auto moves = rollout.GetAvailableMoves();
        if (moves.empty()) return 0;

        int move = moves[std::rand() % moves.size()];
        int r = move / n;
        int c = move % n;
        board.place(r, c, currentPlayer);

        currentPlayer = (currentPlayer == 1 ? 2 : 1);
        rollout.Update(board, currentPlayer);
    }
}


int MonteCarloStrategy::select(const GameState& state, int playerId){
    std::vector<int> moves = state.GetAvailableMoves();
    if (moves.empty()) return -1;

    const auto linear = state.LinearBoard();
    const int n = static_cast<int>(std::sqrt(linear.size()));

    int bestMove = moves.front();
    int bestWins = -1;

    for (int m : moves) {
        
        Board board(n);
        for (int idx = 0; idx < static_cast<int>(linear.size()); ++idx) {
            if (linear[idx] != 0) {
                board.place(idx, linear[idx]);
            }
        }
        board.place(m, playerId);

        int wins = 0;
        int playerToMove = (playerId == 1 ? 2 : 1);
        GameState simState(board, playerToMove);
        if (simState.Winner() == playerId) {
            return m; 
        }
        for (int s = 0; s < SimulationsPerMove; ++s) {
            int w = simulate(simState, playerToMove);
            if (w == playerId) wins++;
        }
        if (wins > bestWins) {
            bestWins = wins;
            bestMove = m;
        }
    }
    int br = bestMove / n;
    int bc = bestMove % n;
    std::cout << "[MonteCarlo] move (" << br << "," << bc << ") wins " << bestWins
              << "/" << SimulationsPerMove << "\n";
    return bestMove;

}

Zobrist::Zobrist(int boardSize) : boardSize(boardSize), keys(boardSize) {
    std::mt19937_64 rng(std::random_device{}());
    for (int cell = 0; cell < boardSize; ++cell) {
        keys[cell][0] = rng();
        keys[cell][1] = rng();
    }
    side = rng();
}

uint64_t Zobrist::computeHash(const Board& board) const{
    uint64_t h = 0;
    const int n = board.N;
    const int total = std::min(boardSize, n * n);
    for (int idx = 0; idx < total; ++idx) {
        int r = idx / n;
        int c = idx % n;
        int val = board.board[r][c];
        if (val == 1 || val == 2) {
            int colorIndex = val - 1; 
            h ^= keys[idx][colorIndex];
        }
    }
    return h;
}

uint64_t Zobrist::applyMoveHash(uint64_t hash, int moveIndex, int color) const{
    if (moveIndex < 0 || moveIndex >= boardSize) return hash;
    if (color != 1 && color != 2) return hash;
    int colorIndex = color - 1;
    hash ^= keys[moveIndex][colorIndex]; 
    hash ^= side;                        
    return hash;
}

uint64_t Zobrist::undoMoveHash(uint64_t hash, int moveIndex, int color) const{
    if (moveIndex < 0 || moveIndex >= boardSize) return hash;
    if (color != 1 && color != 2) return hash;
    int colorIndex = color - 1;
    hash ^= side;                        
    hash ^= keys[moveIndex][colorIndex]; 
    return hash;
}


NegamaxStrategy::NegamaxStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath, bool heuristicOnly, bool preferCuda)
    : maxDepth(maxDepth),
      timeLimitMs(timeLimitMs),
      useHeuristic(heuristicOnly),
      transposition(TT_SIZE),                  
      killers(MAX_DEPTH, { -1, -1 }),
      history(128, 0),
      model(heuristicOnly ? std::string()
                          : resolveModelPath(modelPath.empty() ? defaultModelPath() : modelPath),preferCuda) {}

NegamaxHeuristicStrategy::NegamaxHeuristicStrategy(int maxDepth, int timeLimitMs)
    : NegamaxStrategy(maxDepth, timeLimitMs, defaultModelPath(), true) {}

NegamaxGnnStrategy::NegamaxGnnStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath, bool preferCuda)
    : NegamaxStrategy(maxDepth, timeLimitMs, modelPath.empty() ? defaultModelPath() : modelPath, false,preferCuda) {}


int NegamaxStrategy::select(const GameState& state, int playerId) {
    if (!usageLogged) {
        usageLogged = true;
        if (useHeuristic) {
            std::cout << "[Negamax] Using heuristic evaluation\n";
        } else if (model.isLoaded()) {
            const bool cuda = model.usesCuda();
            const char* label = model.expectsEdgeIndex() ? "GNN" : "MLP";
            std::cout << "[Negamax] Using " << label << " evaluation (" << (cuda ? "CUDA" : "CPU") << ")\n";
        } else {
            std::cout << "[Negamax] GNN not loaded, falling back to heuristic evaluation\n";
        }
    }
    const int immediateWin = findImmediateWinningMove(state, playerId);
    if (immediateWin >= 0) {
        const auto linear = state.LinearBoard();
        const int n = static_cast<int>(std::sqrt(linear.size()));
        const int row = (n > 0 ? immediateWin / n : -1);
        const int col = (n > 0 ? immediateWin % n : -1);
        std::cout << "[Negamax] Immediate winning move in one step | move=" << immediateWin
                  << " (" << row << "," << col << ")\n";
        return immediateWin;
    }
    rootPlayer = playerId;
    SearchResult res = iterativeDeepening(state, playerId);
    return res.bestMove;
}

SearchResult NegamaxStrategy::iterativeDeepening(const GameState& state, int playerId) const {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();

    SearchResult best{-1, std::numeric_limits<int>::min(), true, false, false};
    int depthReached = 0;
    int guess = 0;
    const int window = valueScale * 4; 
    lastEvalPlayer = 0;
    lastEvalVal = 0.0f;
    lastEvalScaled = 0;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        auto now = clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsedMs >= timeLimitMs) break;

        int alpha = guess - window;
        int beta  = guess + window;

        std::cout << "[Negamax] Depth " << depth << " start | alpha=" << alpha << " beta=" << beta << "\n";
        SearchResult res = negamax(state, depth, alpha, beta, playerId, start);

        if (res.failLow || res.failHigh) {
            alpha = std::numeric_limits<int>::min();
            beta  = std::numeric_limits<int>::max();
            res = negamax(state, depth, alpha, beta, playerId, start);
        }

        if (!res.completed) break;
        best = res;
        guess = res.score;
        depthReached = depth;
        std::cout << "[Negamax] Depth " << depth << " done | bestMove=" << best.bestMove
                  << " score=" << best.score << " failLow=" << best.failLow << " failHigh=" << best.failHigh << "\n";
        if (lastEvalPlayer != 0) {
            const char* label = model.expectsEdgeIndex() ? "[GNN]" : "[MLP]";
            std::cout << label << " Depth " << depth << " last eval | nodePlayer=" << lastEvalPlayer
                      << " val=" << lastEvalVal << " scaled=" << lastEvalScaled << "\n";
        }
    }

    const bool usingModel = (!useHeuristic && model.isLoaded());
    if (best.bestMove >= 0) {
        const auto linear = state.LinearBoard();
        const int n = static_cast<int>(std::sqrt(linear.size()));
        const int row = (n > 0 ? best.bestMove / n : -1);
        const int col = (n > 0 ? best.bestMove % n : -1);
        const char* prefix = "[Heuristic]";
        if (usingModel) {
            prefix = model.expectsEdgeIndex() ? "[GNN]" : "[MLP]";
        }
        std::cout << prefix << " Final choice | depth=" << depthReached
                  << " move=" << best.bestMove << " (" << row << "," << col << ")"
                  << " score=" << best.score
                  << " failLow=" << best.failLow << " failHigh=" << best.failHigh << "\n";
    }

    return best;
}

SearchResult NegamaxStrategy::negamax(const GameState& state, int depth, int alpha, int beta, int playerId, std::chrono::steady_clock::time_point startTime) const{
    using clock = std::chrono::steady_clock;
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - startTime).count();
    if (elapsedMs >= timeLimitMs) {
        return {-1, 0, false, false, false}; 
    }

    int winner = state.Winner();
    int opponent = (playerId == 1 ? 2 : 1);
    if (winner == playerId) return {-1, 100000, true, false, false};
    if (winner == opponent) return {-1, -100000, true, false, false};
    if (depth == 0) {
        int evalScore = 0;
        if (!useHeuristic && model.isLoaded()) {
            if (model.expectsEdgeIndex()) {
                extractor.toBatch(state, gnnBatch);
                float val = model.evaluate(gnnBatch, playerId);
                evalScore = static_cast<int>(val * valueScale);
                lastEvalPlayer = playerId;
                lastEvalVal = val;
                lastEvalScaled = evalScore;
            } else {
                HeuristicFeatures feat = computeHeuristicFeatures(state, playerId);
                if (feat.distSelf == 0) {
                    evalScore = 90000;
                    lastEvalVal = static_cast<float>(evalScore);
                } else if (feat.distOpp == 0) {
                    evalScore = -90000;
                    lastEvalVal = static_cast<float>(evalScore);
                } else {
                    std::array<float, 7> inputs = {
                        static_cast<float>(feat.distSelf),
                        static_cast<float>(feat.distOpp),
                        static_cast<float>(feat.libsSelf),
                        static_cast<float>(feat.libsOpp),
                        static_cast<float>(feat.bridgesSelf),
                        static_cast<float>(feat.bridgesOpp),
                        static_cast<float>(feat.center),
                    };
                    float val = model.evaluateFeatures(inputs);
                    evalScore = static_cast<int>(std::lround(val));
                    lastEvalVal = val;
                }
                lastEvalPlayer = playerId;
                lastEvalScaled = evalScore;
            }
        } else {
            evalScore = heuristicEval(state, playerId);
            lastEvalPlayer = 0;
            lastEvalVal = 0.0f;
            lastEvalScaled = evalScore;
        }
        return {-1, evalScore, true, false, false};
    }

    std::vector<int> moves = state.GetAvailableMoves();
    if (moves.empty()) return {-1, 0, true, false, false};

    int bestMove = moves.front();
    int bestScore = std::numeric_limits<int>::min();
    const int alphaOrig = alpha;
    const int betaOrig = beta;

    
    auto linear = state.LinearBoard();
    const int n = static_cast<int>(std::sqrt(linear.size()));
    if (n * n != zobristCells) {
        zobrist = Zobrist(n * n);
        zobristCells = n * n;
    }
    Board base(n);
    for (int idx = 0; idx < static_cast<int>(linear.size()); ++idx) {
        if (linear[idx] != 0) {
            base.place(idx, linear[idx]);
        }
    }
    uint64_t key = zobrist.computeHash(base) ^ (static_cast<uint64_t>(playerId) * 0x9e3779b97f4a7c15ULL);
    TTEntry& tt = transposition[key % transposition.size()];
    if (tt.key == key && tt.depth >= depth) {
        if (tt.flag == TTFlag::EXACT) {
            bool failLow = tt.value <= alphaOrig;
            bool failHigh = tt.value >= betaOrig;
            return {tt.bestMove, tt.value, true, failLow, failHigh};
        } else if (tt.flag == TTFlag::LOWER) {
            alpha = std::max(alpha, tt.value);
        } else if (tt.flag == TTFlag::UPPER) {
            beta = std::min(beta, tt.value);
        }
        if (alpha >= beta) {
            bool failLow = tt.value <= alphaOrig;
            bool failHigh = tt.value >= betaOrig;
            return {tt.bestMove, tt.value, true, failLow, failHigh};
        }
    }

    if (static_cast<int>(history.size()) != n * n) {
        history.assign(n * n, 0);
    }
    const int depthIdx = std::min(depth, MAX_DEPTH - 1);
    const int ttMove = (tt.key == key ? tt.bestMove : -1);
    const int killer1 = killers[depthIdx][0];
    const int killer2 = killers[depthIdx][1];
    if (moves.size() > 1) {
        auto scoreMove = [&](int m) {
            if (m == ttMove) return 100000000;
            if (m == killer1) return 90000000;
            if (m == killer2) return 80000000;
            if (m >= 0 && m < static_cast<int>(history.size())) return history[m];
            return 0;
        };
        std::stable_sort(moves.begin(), moves.end(), [&](int a, int b) {
            return scoreMove(a) > scoreMove(b);
        });
    }

    for (int m : moves) {
        Board childBoard(base); 
        childBoard.place(m, playerId);
        int nextPlayer = (playerId == 1 ? 2 : 1);
        GameState child(childBoard, nextPlayer);

        SearchResult childRes = negamax(child, depth - 1, -beta, -alpha, nextPlayer, startTime);
        if (!childRes.completed) return {bestMove, bestScore, false, false, false};

        int score = -childRes.score;
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (depthIdx >= 0 && depthIdx < static_cast<int>(killers.size())) {
                if (m != killers[depthIdx][0]) {
                    killers[depthIdx][1] = killers[depthIdx][0];
                    killers[depthIdx][0] = m;
                }
            }
            if (m >= 0 && m < static_cast<int>(history.size())) {
                history[m] += depth * depth;
            }
            break; 
        }
    }

    TTFlag flag = TTFlag::EXACT;
    if (bestScore <= alphaOrig) flag = TTFlag::UPPER;
    else if (bestScore >= betaOrig) flag = TTFlag::LOWER;
    tt = {key, depth, bestScore, flag, bestMove};

    bool failLow = bestScore <= alphaOrig;
    bool failHigh = bestScore >= betaOrig;
    return {bestMove, bestScore, true, failLow, failHigh};
}
