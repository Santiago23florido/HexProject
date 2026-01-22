#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"
#include "core/Player.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

int main() {
    try {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        const std::string modelPath = "../scripts/models/hex_value_ts_mp.pt"; 
        Board board;

        char modeChoice = 'h';
        std::cout << "Play against heuristic AI (h) or GNN AI (g)? [h]: ";
        if (!(std::cin >> modeChoice)) {
            throw std::runtime_error("Failed to read AI mode selection");
        }
        bool useGnnAi = (modeChoice == 'g' || modeChoice == 'G');
        
        bool useCPU = false;
        if (useGnnAi){
            char gpuChoice;
            std::cout << "Prefer GPU if available? [y/n]: ";
            if (!(std::cin >> gpuChoice)) {
                throw std::runtime_error("Failed to read GPU preference");
            }
            useCPU = (gpuChoice == 'y' || gpuChoice == 'Y');
        }

        HumanPlayer humanPlayer(1);
        AIPlayer heuristicAI(2, std::make_unique<NegamaxHeuristicStrategy>(4, 4000));
        
        AIPlayer gnnAI(2, std::make_unique<NegamaxGnnStrategy>(4, 4000, modelPath, useCPU));
        if (auto* strat = dynamic_cast<NegamaxStrategy*>(gnnAI.Strategy())) {
            const unsigned int hc = std::thread::hardware_concurrency();
            const int threads = (hc > 1u ? static_cast<int>(hc) : 1);
            strat->setParallelThreads(threads);
        }

        Player* playerX = &humanPlayer;
        Player* playerO = useGnnAi ? static_cast<Player*>(&gnnAI)
                                   : static_cast<Player*>(&heuristicAI);
        Player* current = playerX;

        while (true) {
            board.print();
            std::cout << "\nPlayer " << (current->Id() == 1 ? "X" : "O") << " turn\n";
            GameState state(board, current->Id());
            int moveIdx = current->ChooseMove(state);
            if (!board.place(moveIdx, current->Id())) {
                std::cout << "Invalid move, try again.\n";
                continue;
            }
            state.Update(board, current->Id());
            int w = state.Winner();
            if (w == 1) {
                board.print();
                std::cout << "\nPlayer X wins!\n";
                break;
            }
            if (w == 2) {
                board.print();
                std::cout << "\nPlayer O wins!\n";
                break;
            }

            current = (current == playerX) ? playerO : playerX;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
