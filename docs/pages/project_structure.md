# Project Structure

Below is a concise architecture overview (build folders omitted), followed by the folder‑role table from the report.

```
HexProject/
|-- README.md
|-- docs/
|-- scripts/
|   `-- models/
|-- include/
|   |-- core/
|   |-- gnn/
|   `-- ui/
|-- src/
|   |-- core/
|   |-- gnn/
|   |-- ui/
|   `-- cli/
`-- selfplay/
    |-- gnn/
    `-- mlp/
```

## Main Folders
| Folder | Content and role |
| --- | --- |
| include/ | C++ headers (public API per module). |
| src/ | C++ implementation (core, GNN, UI, CLI). |
| selfplay/ | Self‑play data generation and training (gnn/mlp). |
| scripts/ | Python scripts and trained model artifacts. |
| docs/ | Project report and documentation pages. |

