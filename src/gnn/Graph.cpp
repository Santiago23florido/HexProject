#include "gnn/Graph.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include "Cube.hpp"

Graph buildHexGraph(int N, bool addBorderSuperNodes) {
    Graph g;
    g.N = N;
    const int baseNodes = N * N;
    g.superOffset = addBorderSuperNodes ? baseNodes : -1;
    g.superCount = addBorderSuperNodes ? 2 : 0;
    g.numNodes = baseNodes + g.superCount;

    g.adj.assign(g.numNodes, {});
    g.features.assign(g.numNodes, {});

    static const Cube Directions[6] = {
        Cube(+1, -1, 0),
        Cube(+1, 0, -1),
        Cube(0, +1, -1),
        Cube(-1, +1, 0),
        Cube(-1, 0, +1),
        Cube(0, -1, +1)
    };

    std::vector<Cube> cubes(baseNodes);
    std::unordered_map<long long, int> keyToIdx;
    keyToIdx.reserve(baseNodes * 2);

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * N + c;
            int q = c - (r - (r % 2)) / 2; 
            int r_axial = r;
            int x = q;
            int z = r_axial;
            int y = -x - z;
            cubes[idx] = Cube(x, y, z);
            keyToIdx[cubes[idx].key()] = idx;
        }
    }

    for (int idx = 0; idx < baseNodes; ++idx) {
        const Cube& cur = cubes[idx];
        for (int d = 0; d < 6; ++d) {
            Cube nb = cur + Directions[d];
            auto it = keyToIdx.find(nb.key()); //Unordered map iterator
            if (it != keyToIdx.end()) {
                int nidx = it->second;
                if (nidx > idx) {
                    g.adj[idx].push_back(nidx); // add edge idx -> nidx
                    g.adj[nidx].push_back(idx); // add edge nidx -> idx
                }
            }
        }

        int r = idx / N;
        int c = idx % N;
        NodeFeatures nf;
        nf.sideA = (c == 0 || c == N - 1) ? 1.f : 0.f;
        nf.sideB = (r == 0 || r == N - 1) ? 1.f : 0.f;
        nf.degree = static_cast<float>(g.adj[idx].size()) / 6.0f;
        g.features[idx] = nf;
    }

    if (addBorderSuperNodes) {
        const int superA = g.superOffset;     // for player 1 sides (columns)
        const int superB = g.superOffset + 1; // for player 2 sides (rows)
        for (int r = 0; r < N; ++r) {
            g.adj[superA].push_back(r * N);           // add edge superA -> left column cell
            g.adj[r * N].push_back(superA);           // add edge cell -> superA
            g.adj[superA].push_back(r * N + (N - 1)); // add edge superA -> right column cell
            g.adj[r * N + (N - 1)].push_back(superA); // add edge cell -> superA
        }
        for (int c = 0; c < N; ++c) {
            g.adj[superB].push_back(c);               // add edge superB -> top row cell
            g.adj[c].push_back(superB);               // add edge cell -> superB
            g.adj[superB].push_back((N - 1) * N + c); // add edge superB -> bottom row cell
            g.adj[(N - 1) * N + c].push_back(superB); // add edge cell -> superB
        }
        g.features[superA].degree = static_cast<float>(g.adj[superA].size()) / std::max(1, 2 * N);
        g.features[superB].degree = static_cast<float>(g.adj[superB].size()) / std::max(1, 2 * N);

        // Unweighted BFS from each supernode to compute shortest hop counts
        auto bfs = [&](int source, std::vector<float>& out) {
            out.assign(g.numNodes, std::numeric_limits<float>::infinity());
            std::queue<int> q;
            out[source] = 0.f;
            q.push(source);
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                float du = out[u];
                for (int v : g.adj[u]) {
                    if (out[v] > du + 1.f) {
                        out[v] = du + 1.f;
                        q.push(v);
                    }
                }
            }
        };

        std::vector<float> distA;
        std::vector<float> distB;
        bfs(superA, distA);
        bfs(superB, distB);

        float norm = static_cast<float>(baseNodes > 0 ? baseNodes : 1); // normalize by board cells
        for (int i = 0; i < baseNodes; ++i) {
            g.features[i].distToA = distA[i] / norm; // normalized shortest hops to side A
            g.features[i].distToB = distB[i] / norm; // normalized shortest hops to side B
        }
    }

    return g;
}

void fillFeatures(Graph& g, const Board& board) {
    const int N = board.N;
    const int total = std::min(static_cast<int>(g.features.size()), N * N);
    for (int idx = 0; idx < total; ++idx) {
        int r = idx / N;
        int c = idx % N;
        int val = board.board[r][c];
        NodeFeatures nf = g.features[idx]; 
        nf.p1 = (val == 1) ? 1.f : 0.f;
        nf.p2 = (val == 2) ? 1.f : 0.f;
        nf.empty = (val == 0) ? 1.f : 0.f;
        g.features[idx] = nf;
    }

}

void fillFeatures(Graph& g, const GameState& state) {
    const auto linear = state.LinearBoard();
    const int N = static_cast<int>(std::sqrt(linear.size()));
    const int total = std::min(static_cast<int>(g.features.size()), N * N);
    for (int idx = 0; idx < total; ++idx) {
        int val = linear[idx];
        NodeFeatures nf = g.features[idx];
        nf.p1 = (val == 1) ? 1.f : 0.f;
        nf.p2 = (val == 2) ? 1.f : 0.f;
        nf.empty = (val == 0) ? 1.f : 0.f;
        g.features[idx] = nf;
    }
    
}
