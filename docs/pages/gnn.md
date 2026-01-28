# MLP / GNN Module

This module represents the board as a graph, extracts node features, and provides a TorchScript wrapper for evaluating either a GNN or a flat‑feature MLP. It builds graph connectivity once per board size, fills dynamic features from a `Board` or `GameState`, and converts them to flattened batches for model inference.

## Classes and Structures
| Class/Structure (type) | Direct derivatives | Description |
| --- | --- | --- |
| Graph (structure) |  | Graph topology and per‑node features. |
| NodeFeatures (structure) |  | Feature vector for each node (occupancy, borders, distances, turn). |
| FeatureBatch (structure) |  | Flattened tensors for GNN/MLP evaluation. |
| FeatureExtractor (class) |  | Builds/reuses graphs and produces batches. |
| GNNModel (class) |  | TorchScript wrapper for GNN/MLP evaluation. |
