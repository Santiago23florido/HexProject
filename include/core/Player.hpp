#pragma once
#include <memory>
#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"

/**
 * Player base interface.
 */
class Player {
    public:
        /// Chooses a move for the current state.
        virtual int ChooseMove(const GameState& state)=0;
        /// Returns the player id (1 or 2).
        virtual int Id() const = 0;
        /// Virtual destructor for safe polymorphic cleanup.
        virtual ~Player() = default;
};

/**
 * Human player reading moves from stdin.
 */
class HumanPlayer: public Player {
    int PlayerId; // immutable identity
    public:
    /// Creates a human player with the given id.
    HumanPlayer(int id);
    /// Prompts for and returns a valid move.
    int ChooseMove(const GameState& state) override;
    /// Returns the player id.
    int Id() const override;
};

/**
 * AI player driven by a move strategy.
 */
class AIPlayer : public Player {
    int playerId;
    std::unique_ptr<IMoveStrategy> strategy;
public:
    /// Creates an AI player with a default RandomStrategy.
    AIPlayer(int id); // defaults to RandomStrategy
    /// Creates an AI player with a provided strategy.
    AIPlayer(int id, std::unique_ptr<IMoveStrategy> s);
    /// Delegates move selection to the strategy.
    int ChooseMove(const GameState& state) override;
    /// Returns the player id.
    int Id() const override;
    /// Returns a mutable pointer to the strategy.
    IMoveStrategy* Strategy();
    /// Returns a const pointer to the strategy.
    const IMoveStrategy* Strategy() const;
};


/**
 * Hybrid player that can behave as human or AI.
 */
// Can act as a human (no strategy) or AI (with strategy)
class HybridPlayer : public Player {
    int playerId;
    std::unique_ptr<IMoveStrategy> strategy;
public:
    /// Creates a human-mode hybrid player.
    HybridPlayer(int id); // human by default
    /// Creates an AI-mode hybrid player when strategy is provided.
    HybridPlayer(int id, std::unique_ptr<IMoveStrategy> s); // AI when strategy is provided
    /// Copies identity from a human player.
    HybridPlayer(const HumanPlayer& other); // copy identity, stay human
    /// Copies identity from an AI player and becomes AI.
    HybridPlayer(const AIPlayer& other);    // copy identity, become AI (RandomStrategy)
    /// Copies identity and current mode.
    HybridPlayer(const HybridPlayer& other); // copy identity, keep mode (human/AI)
    /// Assigns identity and mode from another hybrid player.
    HybridPlayer& operator=(const HybridPlayer& other);
    HybridPlayer(HybridPlayer&&) noexcept = default;
    HybridPlayer& operator=(HybridPlayer&&) noexcept = default;
    /// Chooses a move using strategy or stdin.
    int ChooseMove(const GameState& state) override;
    /// Returns the player id.
    int Id() const override;
};
