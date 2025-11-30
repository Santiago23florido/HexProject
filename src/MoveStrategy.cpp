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
    hash ^= keys[moveIndex][colorIndex]; // agrega la pieza
    hash ^= side;                        // alterna el turno
    return hash;
}

uint64_t Zobrist::undoMoveHash(uint64_t hash, int moveIndex, int color) const{
    if (moveIndex < 0 || moveIndex >= boardSize) return hash;
    if (color != 1 && color != 2) return hash;
    int colorIndex = color - 1;
    hash ^= side;                        // revierte el turno
    hash ^= keys[moveIndex][colorIndex]; // quita la pieza
    return hash;
}
