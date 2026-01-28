# Self-Play Module

The self‑play module covers data collection and reinforcement learning around the value model. It records game states, converts them to labeled training samples, stores them in a replay buffer, and trains a value MLP. It also manages checkpoints and TorchScript export for inference in the C++ engine.

## Classes and Structures
| Class/Structure (type) | Direct derivatives | Description |
| --- | --- | --- |
| RLTrainer (class) |  | Orchestrates self‑play, training, and persistence. |
| RLConfig (structure) |  | Training and self‑play configuration parameters. |
| ReplayBuffer (class) |  | Fixed‑capacity buffer with uniform sampling. |
| ReplaySample (structure) |  | Training example (features + target). |
| ValueMLP (class) |  | LibTorch wrapper around ValueMLPImpl. |
| ValueMLPImpl (class) |  | Value MLP network; inherits `torch::nn::Module`. |
| DataCollector (class) |  | Collects and aggregates self‑play states. |
| Sample (structure) |  | Game snapshot with outcome and plies remaining. |
| GameRunner (class) |  | Runs a game between strategies and records states. |
| Serializer (class) |  | Writes samples to JSON Lines files. |
