#pragma once

#include "MoveStrategy.hpp"
#include "Board.hpp"
#include "GameState.hpp"
#include "DataCollector.hpp"

class GameRunner {
public:
    GameRunner(int boardSize, IMoveStrategy& p1Strategy, IMoveStrategy& p2Strategy);

    // Plays one full game, records all states, returns winner (0 if draw)
    int playOne(DataCollector& collector);

private:
    int boardSize;
    IMoveStrategy& p1;
    IMoveStrategy& p2;
};
