#pragma once
#include <vector>
#include <iostream>
#include "GameState.hpp"

//Parent class player
class Player {
    public: 
        virtual int ChooseMove(const GameState& state)=0;
        virtual int id() const = 0;
        virtual ~Player() = default;

};