#pragma once

#include "core/MoveStrategy.hpp"
#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "DataCollector.hpp"

class GameRunner {
public:
    GameRunner(int boardSize, IMoveStrategy& p1Strategy, IMoveStrategy& p2Strategy);

    // Plays one full game, records all states, returns winner (0 if draw)
    int playOne(DataCollector& collector, const Board* startingBoard = nullptr);

private:
    int boardSize;
    IMoveStrategy& p1;
    IMoveStrategy& p2;
};
