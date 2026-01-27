#pragma once

#include <vector>

/**
 *  Compact training sample for a game state.
 *
 * Owns the linearized board and stores outcome from the player-to-move view; movesToEnd is in plies.
 */
struct Sample {
    int N{0};
    std::vector<int> board; // linearized board
    int toMove{0};          // player to move in this state
    int result{0};          // +1 win, -1 loss, 0 draw from toMove perspective
    int movesToEnd{0};      // how many plies remain from this state until game end
};

/**
 *  Collects self-play samples and aggregates them across games.
 *
 * Owns per-game buffers and accumulated samples; results are from each state's player-to-move view.
 */
class DataCollector {
public:
    /// Records a state snapshot for the current game.
    void recordState(int N, const std::vector<int>& board, int toMove);
    /// Finalizes the current game with the winner (0 = draw) and annotates outcomes.
    void finalizeGame(int winner);
    /// Clears all buffered and accumulated samples.
    void reset();
    /// Returns a read-only view of accumulated samples.
    const std::vector<Sample>& samples() const { return allSamples; }
    /// Returns and clears all accumulated samples.
    std::vector<Sample> consumeSamples();

private:
    std::vector<Sample> gameBuffer;
    std::vector<Sample> allSamples;
};
