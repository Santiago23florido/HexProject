#pragma once

#include <vector>

struct Sample {
    int N{0};
    std::vector<int> board; // linearized board
    int toMove{0};          // player to move in this state
    int result{0};          // +1 win, -1 loss, 0 draw from toMove perspective
    int movesToEnd{0};      // how many plies remain from this state until game end
};

class DataCollector {
public:
    void recordState(int N, const std::vector<int>& board, int toMove);
    void finalizeGame(int winner);
    void reset();
    const std::vector<Sample>& samples() const { return allSamples; }
    std::vector<Sample> consumeSamples();

private:
    std::vector<Sample> gameBuffer;
    std::vector<Sample> allSamples;
};
