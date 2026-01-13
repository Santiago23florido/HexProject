#include "ui/HexGameUI.hpp"

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>  //info hardware cuda

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    const std::string modelPath = "../scripts/models/hex_value_ts.pt";

    bool useGnnAi = true;
    bool preferCuda = true;

    const int boardSize = 7;
    const float tileScale = 0.1f;

    std::cout << "--- HARDWARE CHECK ---" << std::endl;
    if (torch::cuda::is_available()) {
        auto properties = at::cuda::getDeviceProperties(0); 
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "Device Name: " << properties->name << std::endl;
        // Cambiamos total_memory por totalGlobalMem
        std::cout << "Memory: " << properties->totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    } else {
        std::cout << "CUDA NOT FOUND. Using CPU." << std::endl;
    }

    try
    {
        HexGameUI game(
            "../assets/hex1.png",
            "../assets/background.png",
            "../assets/Player 1.png",
            "../assets/Player 2.png",
            "../assets/start_page.png",
            "../assets/start_button.png",
            "../assets/HEX.png",
            "../assets/player_selection.png",
            "../assets/player_start.png",
            "../assets/next_type.png",
            "../assets/Player1Select/Human.png",
            "../assets/Player2Select/Human.png",
            "../assets/Player2Select/GNN.png",
            "../assets/Player2Select/Heuristic.png",
            "../assets/Player1win.png",
            "../assets/Player2win.png",
            boardSize,
            tileScale,
            useGnnAi,
            modelPath,
            preferCuda);

        return game.run();
    }
    catch(const std::exception& e)
    {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
