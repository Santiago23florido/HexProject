#include "DataCollector.hpp"

void DataCollector::recordState(int N, const std::vector<int>& board, int toMove) {
    Sample s;
    s.N = N;
    s.board = board;
    s.toMove = toMove;
    gameBuffer.push_back(std::move(s));
}

void DataCollector::finalizeGame(int winner) {
    for (auto& s : gameBuffer) {
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
