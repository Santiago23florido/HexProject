#pragma once

#include "core/MoveStrategy.hpp"
#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "gnn/DataCollector.hpp"

/**
 * Runs a self-play game between two strategies and records states.
 *
 * Holds non-owning strategy references; boardSize is the side length in cells.
 */
class GameRunner {
public:
    /// Creates a runner for a board of size boardSize (cells per side).
    GameRunner(int boardSize, IMoveStrategy& p1Strategy, IMoveStrategy& p2Strategy);

    /// Plays one full game, records states into collector, and returns winner (0 = draw).
    int playOne(DataCollector& collector, const Board* startingBoard = nullptr);

private:
    int boardSize;
    IMoveStrategy& p1;
    IMoveStrategy& p2;
};
