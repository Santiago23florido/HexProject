#include "GameRunner.hpp"

#include <vector>

// Runs a single self-play game with two provided strategies
GameRunner::GameRunner(int boardSize, IMoveStrategy& p1Strategy, IMoveStrategy& p2Strategy)
    : boardSize(boardSize), p1(p1Strategy), p2(p2Strategy) {}

int GameRunner::playOne(DataCollector& collector, const Board* startingBoard) {
    Board board = startingBoard ? *startingBoard : Board(boardSize);
    int currentPlayer = 1;
    GameState state(board, currentPlayer);

    while (true) {
        collector.recordState(boardSize, state.LinearBoard(), currentPlayer);

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
