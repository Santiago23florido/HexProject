# UI Module

The UI module provides the SFML-based front-end for interactive Hex play. It renders the board and UI screens (start, player selection, and in-game), handles mouse/keyboard input, and connects to the core game logic and AI players.

It manages textures, audio playback, and basic settings (volume and video quality), and can switch between heuristic and GNN evaluation paths at runtime. A lightweight ImageViewer utility is included for previewing tile layouts and textures.

## Notes
- Entry point: `src/ui/main.cpp` launches `HexGameUI` with asset paths and runtime options.
- UI assets are loaded from the `assets/` folder (textures, icons, and screens).

## Classes and Structures
| Class/Structure (type) | Direct derivatives | Description |
| --- | --- | --- |
| HexGameUI (class) |  | SFML interface with screens, input handling, and the game loop. |
| HexTile (class) |  | Hex tile sprite wrapper with placement and styling helpers. |
| ImageViewer (class) |  | Utility window for viewing tiled textures. |
