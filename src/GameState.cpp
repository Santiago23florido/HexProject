#include "GameState.hpp"
#include "Board.hpp"
#include "Cube.hpp"
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>


static const Cube Directions[6] = {
    Cube(+1, -1, 0),
    Cube(+1, 0, -1),
    Cube(0, +1, -1),
    Cube(-1, +1, 0),
    Cube(-1, 0, +1),
    Cube(0, -1, +1)
}; //cube directions for neighbors calculation

GameState :: GameState(int n) : N(n), Hex(n, std::vector<int>(n, 0)) , Player(0){}
GameState :: GameState(const Board& b,int player) : N(b.N), Hex(b.board), Player(player){}
GameState :: GameState(const GameState& other) : N(other.N), Hex(other.Hex), Player(other.Player) {}
void GameState :: Update(const Board& b, int player) {
    N = b.N;
    Hex = b.board;
    Player = player;
}
//Linear list representation of indexes in current board
std::vector<int> GameState::LinearBoard() const {
    std :: vector<int> LinearHex;  //Single number  Board Representations
    std::vector<std::vector<int>>::const_iterator rows;
    std::vector<int>::const_iterator cols;
    for (rows = Hex.cbegin();rows != Hex.cend();++rows){
        for(cols = rows->cbegin();cols != rows->cend();++cols){
                int i = rows - Hex.cbegin();
                int j = cols - rows->cbegin();
                LinearHex.push_back(*cols);
        }
    }
    return LinearHex;
}
//Available moves for current GameState
std::vector<int> GameState::GetAvailableMoves() const {
    std :: vector<int> moves;  //Single number  Available Moves Representations
    std :: vector<int> LinearHex = GameState :: LinearBoard();
    std::vector<int>::const_iterator it;
    for (it = LinearHex.cbegin();it != LinearHex.cend();++it){
            if(*it == 0){ //free hexagons for the current GameState
                int i = it - LinearHex.cbegin();
                moves.push_back(i);
            }
        }
    return moves;
}
//Cube coordinates Board indexes conversion
std::vector<Cube> GameState::ToCubeCoordinates() const{
    std::vector<Cube> CubeCoords(N*N);
    std::vector<std::vector<int>>::const_iterator rows;
    std::vector<int>::const_iterator cols;
    for (rows = Hex.cbegin();rows != Hex.cend();++rows){
        for(cols = rows->cbegin();cols != rows->cend();++cols){
            int r = rows - Hex.cbegin();
            int c = cols - rows->cbegin();
            int index = r*N + c; //Linear index calculation
            int q = c - (r - (r % 2)) / 2; // offset odd-r → axial 
            int r_axial = r;
            // axial → cube
            int x = q;
            int z = r_axial;
            int y = -x - z;
            CubeCoords[index] = Cube(x,y,z);
        }
    }
    return CubeCoords;
}
//Verify if the game is over
int GameState :: IsTerminal() const{
    int winner = GameState::Winner();
    std::vector<int> availablemoves = GameState::GetAvailableMoves();
    if(winner!= 0){
        return 1;
    }else if(availablemoves.empty()){
        return 1;
    }else{
        return 0;
    }
}
//Determine if there is a winner in the current GameState
int GameState :: Winner() const {
    std::vector<Cube> CubeCoords = GameState::ToCubeCoordinates();
    std::vector<int> LinearHex = GameState::LinearBoard();
    std::unordered_map<long long, int> CoordToIndex;
    //Mapping of cube coordinates to linear board indexes
    for (int index = 0; index < N*N; index++) {
        auto &c = CubeCoords[index];
        CoordToIndex[c.key()] = index;
    }

    //Check win Player 1
    std::vector<int> Queue1;
    std::vector<int> Visited1;

    // First column check - starting points identification
    std::vector<int>::const_iterator it;
    for (int r = 0; r < N; r++) {
        it = LinearHex.cbegin() + (r * N); // (r,0)
        if (*it == 1) {
            int index = it - LinearHex.cbegin();
            Queue1.push_back(index);
            Visited1.push_back(index);
        }
    }

    while (!(Queue1.empty())) {
        int current = Queue1.back();
        Queue1.pop_back();
        int col = current % N;
        if(col == N-1){
            return 1; // If BFS reaches last column , player 1 wins
        }
        for (int d = 0;d < 6; d++){ //Exploration of neighbors
            Cube NeighborCube = CubeCoords[current] + Directions[d];
            long long NeighborKey = NeighborCube.key();
            auto found = CoordToIndex.find(NeighborKey);
            if(found != CoordToIndex.end()){
                int NeighborIndex = found-> second;
                if(LinearHex[NeighborIndex] == 1 && std::find(Visited1.begin(), Visited1.end(), NeighborIndex) == Visited1.end()){//Check if neighbor related to player 1 and non visited
                    Queue1.push_back(NeighborIndex);
                    Visited1.push_back(NeighborIndex);
                }
            }
        }
    }

    //Check win Player 2
    std::vector<int> Queue2;
    std::vector<int> Visited2;

    // First row check - starting points identification
    std::vector<int>::const_iterator it2;
    for (int r = 0; r < N; r++) {
        it2 = LinearHex.cbegin() + r; // (0,r)
        if (*it2 == 2) {
            int index = it2 - LinearHex.cbegin();
            Queue2.push_back(index);
            Visited2.push_back(index);
        }
    }
    while (!(Queue2.empty())) {
        int current = Queue2.back();
        Queue2.pop_back();
        int row = (current - current % N) / N;
        if(row == N-1){
            return 2; // If BFS reaches last row , player 2 wins
        }
        for (int d = 0;d < 6; d++){ //Exploration of neighbors
            Cube NeighborCube = CubeCoords[current] + Directions[d];
            long long NeighborKey = NeighborCube.key();
            auto found = CoordToIndex.find(NeighborKey);
            if(found != CoordToIndex.end()){
                int NeighborIndex = found-> second;
                if(LinearHex[NeighborIndex] == 2 && std::find(Visited2.begin(), Visited2.end(), NeighborIndex) == Visited2.end()){//Check if neighbor related to player 2 and non visited
                    Queue2.push_back(NeighborIndex);
                    Visited2.push_back(NeighborIndex);
                }
            }
        }
    }
    return 0; // No winner for the GameState
}
