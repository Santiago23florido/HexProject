#pragma once
#include "Board.hpp"
#include <vector>
#include <iostream>

class Cube {
    public:
    int x;
    int y;
    int z;
    Cube(int x_,int y_, int z_);
    long long key() const;
    Cube();
    Cube operator+(const Cube& other) const; // Adition cube coordinates operator
};//Cube coordinates structure