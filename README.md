# Hex Project

Motor de Hex en C++ con dos frentes de IA:
- **Heurística Negamax**: evaluación basada en distancias, libertades, puentes y detección de victoria inmediata.
- **GNN de valor**: modelo TorchScript que evalúa estados; la búsqueda se apoya en la misma estructura de Negamax.

Incluye generación de self-play para entrenamiento y un script de entrenamiento en Python.

## Cómo compilar y jugar

```bash
cmake -S . -B build
cmake --build build
./build/hex
```

El binario pregunta si quieres jugar contra IA heurística (`h`, por defecto) o IA GNN (`g`).  
Configuración actual: heurística profundidad 3 / 2000 ms; GNN profundidad 5 / 3000 ms.

## Self-play para datos

Target separado en `selfplay/`:

```bash
cmake -S selfplay -B selfplay/build
cmake --build selfplay/build
./selfplay/build/selfplay <games> <minDepth> <maxDepth> <outputPath> <minPairs> <maxPairs> <timeLimitMs>
```

- Estrategias: dos Negamax heurísticos con profundidad aleatoria en `[minDepth, maxDepth]`.
- Tiempo por movimiento: `timeLimitMs` ms.
- Estado inicial: se reparten aleatoriamente `pairs` fichas por jugador (`pairs` ∈ `[minPairs, maxPairs]`). Se exige que ambos tengan al menos una cadena conectada de 2 piedras para asegurar posiciones con grupos relevantes.
- Salida: JSONL por tamaño `selfplay_data_N7.jsonl` (para N=7). Cada línea: `N, board[], to_move, result, moves_to_end`.
- Escritura incremental cada 20 partidas para limitar memoria.

## GNN: features y entrenamiento

- Features por nodo (8): `p1, p2, empty, sideA, sideB, degree, distToA, distToB`. Se corresponden con `FeatureExtractor` en C++.
- Modelo (`scripts/train_gnn.py`):
  - Backbone simple con 3 capas lineales (hidden 128) y pooling global medio.
  - Cabeza principal: valor en `[-1,1]` (perspectiva de `to_move`).
  - Cabeza auxiliar: predice `moves_to_end` (normalizado) solo para entrenamiento; no se usa en inferencia.
  - Args útiles: `--epochs`, `--lr`, `--aux-weight`, `--endgame-weight` (permite ponderar más los estados finales).

Ejemplo de entrenamiento:

```bash
python3 scripts/train_gnn.py \
  --data selfplay/build/selfplay_data_N7.jsonl \
  --epochs 20 --lr 1e-3 \
  --aux-weight 0.1 --endgame-weight 1.0
```

El modelo se guarda en `scripts/models/hex_value_ts.pt` y es el que carga el binario (`./hex` y `selfplay`).

## Arquitectura (breve)

- `Board`, `GameState`, `Cube`: estado y detección de ganador vía BFS en coordenadas cúbicas.
- `MoveStrategy`: `Random`, `MonteCarlo` (básico) y `Negamax` con hashing Zobrist y tabla de transposición.
- `gnn/Graph`, `FeatureExtractor`: construyen el grafo Hex y las features para LibTorch.
- `GNNModel`: wrapper TorchScript para evaluación.
- `selfplay`: generador de datos configurable (profundidad, tiempo, densidad inicial).
