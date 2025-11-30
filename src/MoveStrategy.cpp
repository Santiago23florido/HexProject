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
