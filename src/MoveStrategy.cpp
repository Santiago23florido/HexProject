#include "GameState.hpp"
#include "Board.hpp"
#include "Cube.hpp"
#include "MoveStrategy.hpp"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>

//Random selection from the available moves list
int RandomStrategy::select(const GameState& state, int playerId) {
    std::vector<int> moves = state.GetAvailableMoves();
    if (moves.empty()) return -1;
    return moves[std::rand() % moves.size()];
}

MonteCarloStrategy::MonteCarloStrategy(int sims) : SimulationsPerMove(sims) {}

// Random playout from given state; returns winner (0 if none)
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
        // Build board for this move
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
            return m; // immediate win after this move
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
            int colorIndex = val - 1; // player 1 -> 0, player 2 -> 1
            h ^= keys[idx][colorIndex];
        }
    }
    return h;
}

uint64_t Zobrist::applyMoveHash(uint64_t hash, int moveIndex, int color) const{
    if (moveIndex < 0 || moveIndex >= boardSize) return hash;
    if (color != 1 && color != 2) return hash;
    int colorIndex = color - 1;
    hash ^= keys[moveIndex][colorIndex]; // add piece
    hash ^= side;                        // toggle side to move
    return hash;
}

uint64_t Zobrist::undoMoveHash(uint64_t hash, int moveIndex, int color) const{
    if (moveIndex < 0 || moveIndex >= boardSize) return hash;
    if (color != 1 && color != 2) return hash;
    int colorIndex = color - 1;
    hash ^= side;                        // revert side to move
    hash ^= keys[moveIndex][colorIndex]; // remove piece
    return hash;
}


NegamaxStrategy::NegamaxStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath)
    : maxDepth(maxDepth),
      timeLimitMs(timeLimitMs),
      transposition(1 << 16),                  // placeholder size
      killers(MAX_DEPTH, { -1, -1 }),
      history(128, 0),
      model(modelPath) {}                      // load TorchScript model


int NegamaxStrategy::select(const GameState& state, int playerId) {
    if (!usageLogged) {
        usageLogged = true;
        std::cout << "[Negamax] Using GNN evaluation\n";
    }
    SearchResult res = iterativeDeepening(state, playerId);
    return res.bestMove;
}

SearchResult NegamaxStrategy::iterativeDeepening(const GameState& state, int playerId) const {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();

    SearchResult best{-1, std::numeric_limits<int>::min(), true, false, false};
    int guess = 0;
    const int window = 50;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        auto now = clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsedMs >= timeLimitMs) break;

        int alpha = guess - window;
        int beta  = guess + window;

        std::cout << "[Negamax] Depth " << depth << " start | alpha=" << alpha << " beta=" << beta << "\n";
        SearchResult res = negamax(state, depth, alpha, beta, playerId, static_cast<uint64_t>(start.time_since_epoch().count()));

        if (res.failLow || res.failHigh) {
            alpha = std::numeric_limits<int>::min();
            beta  = std::numeric_limits<int>::max();
            res = negamax(state, depth, alpha, beta, playerId, static_cast<uint64_t>(start.time_since_epoch().count()));
        }

        if (!res.completed) break;
        best = res;
        guess = res.score;
        std::cout << "[Negamax] Depth " << depth << " done | bestMove=" << best.bestMove
                  << " score=" << best.score << " failLow=" << best.failLow << " failHigh=" << best.failHigh << "\n";
    }
    return best;
}

SearchResult NegamaxStrategy::negamax(const GameState& state, int depth, int alpha, int beta, int playerId, uint64_t startTime) const{
    using clock = std::chrono::steady_clock;
    auto now = clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch() - std::chrono::steady_clock::time_point::duration(startTime))
                         .count();
    if (elapsedMs >= timeLimitMs) {
        return {-1, 0, false, false, false}; // time limit hit
    }

    int winner = state.Winner();
    int opponent = (playerId == 1 ? 2 : 1);
    if (winner == playerId) return {-1, 100000, true, false, false};
    if (winner == opponent) return {-1, -100000, true, false, false};
    if (depth == 0) {
        int evalScore = 0;
        if (model.isLoaded()) {
            FeatureBatch batch = extractor.toBatch(state);
            float val = model.evaluate(batch, playerId);
            evalScore = static_cast<int>(val * valueScale);
            std::cout << "[GNN] Eval depth0 | player=" << playerId << " val=" << val
                      << " scaled=" << evalScore << "\n";
        }
        return {-1, evalScore, true, false, false};
    }

    std::vector<int> moves = state.GetAvailableMoves();
    if (moves.empty()) return {-1, 0, true, false, false};

    int bestMove = moves.front();
    int bestScore = std::numeric_limits<int>::min();
    const int alphaOrig = alpha;
    const int betaOrig = beta;

    // Rebuild board from state once for reuse
    auto linear = state.LinearBoard();
    const int n = static_cast<int>(std::sqrt(linear.size()));

    for (int m : moves) {
        Board b(n);
        for (int idx = 0; idx < static_cast<int>(linear.size()); ++idx) {
            if (linear[idx] != 0) {
                b.place(idx, linear[idx]);
            }
        }
        b.place(m, playerId);
        int nextPlayer = (playerId == 1 ? 2 : 1);
        GameState child(b, nextPlayer);

        SearchResult childRes = negamax(child, depth - 1, -beta, -alpha, nextPlayer, startTime);
        if (!childRes.completed) return {bestMove, bestScore, false, false, false};

        int score = -childRes.score;
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; // beta cut
    }

    bool failLow = bestScore <= alphaOrig;
    bool failHigh = bestScore >= betaOrig;
    return {bestMove, bestScore, true, failLow, failHigh};
}
