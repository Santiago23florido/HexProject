#pragma once
#include <vector>
#include <iostream>

class Board {
public:
    int N;
    std::vector<std::vector<int>> board; //0 = free cell, 1 = Player 1 , 2 Player 2
    Board(int n = 7);    
    Board(const Board& other);
    void print() const;  
    bool place(int r, int c, int player); 
    bool place(int idx, int player); // linear index helper
};
