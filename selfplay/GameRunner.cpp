#include "GameRunner.hpp"

#include <cmath>
#include <vector>

GameRunner::GameRunner(int boardSize, IMoveStrategy& p1Strategy, IMoveStrategy& p2Strategy)
    : boardSize(boardSize), p1(p1Strategy), p2(p2Strategy) {}

int GameRunner::playOne(DataCollector& collector) {
    Board board(boardSize);
    int currentPlayer = 1;
    GameState state(board, currentPlayer);

    auto linearize = [&](const Board& b) {
        std::vector<int> linear;
        linear.reserve(boardSize * boardSize);
        for (int r = 0; r < boardSize; ++r) {
            for (int c = 0; c < boardSize; ++c) {
                linear.push_back(b.board[r][c]);
            }
        }
        return linear;
    };

    while (true) {
        collector.recordState(boardSize, linearize(board), currentPlayer);

        int winner = state.Winner();
        if (winner != 0) {
            collector.finalizeGame(winner);
            return winner;
        }
        auto moves = state.GetAvailableMoves();
        if (moves.empty()) {
            collector.finalizeGame(0);
            return 0;
        }

        IMoveStrategy& strat = (currentPlayer == 1) ? p1 : p2;
        int move = strat.select(state, currentPlayer);
        if (move < 0 || move >= boardSize * boardSize) {
            collector.finalizeGame(0);
            return 0;
        }

        board.place(move, currentPlayer);
        currentPlayer = (currentPlayer == 1 ? 2 : 1);
        state.Update(board, currentPlayer);
    }
}
