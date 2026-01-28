#include "DataCollector.hpp"

// Collects self-play training samples, buffering per-game states and annotating them with outcomes and plies remaining.

void DataCollector::recordState(int N, const std::vector<int>& board, int toMove) {
    Sample s;
    s.N = N;
    s.board = board;
    s.toMove = toMove;
    gameBuffer.push_back(std::move(s));
}

void DataCollector::finalizeGame(int winner) {
    const int totalStates = static_cast<int>(gameBuffer.size());

    // Annotate each buffered state with its outcome and remaining plies from that position.
    for (int idx = 0; idx < totalStates; ++idx) {
        auto& s = gameBuffer[idx];
        s.movesToEnd = totalStates - idx - 1; // 0 for the terminal position

        if (winner == 0) {
            s.result = 0;
        } else if (winner == s.toMove) {
            s.result = 1;
        } else {
            s.result = -1;
        }
        allSamples.push_back(std::move(s));
    }
    gameBuffer.clear();
}

void DataCollector::reset() {
    gameBuffer.clear();
    allSamples.clear();
}

std::vector<Sample> DataCollector::consumeSamples() {
    std::vector<Sample> out;
    // Transfer ownership and clear accumulated samples.
    out.swap(allSamples);
    return out;
}
