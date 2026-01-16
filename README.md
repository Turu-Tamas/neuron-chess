# Neuron Chess

Processes chess games from PGN files and generates neural network training data using Leela Chess Zero.

## Data File Format

**Location**: `data/lc0-hidden/lichess_elite_2025-11.h5` (HDF5 format)

### Root Dataset
- **`lc0_hidden`**: Board states (shape: N×64×8×8, dtype: float16)
  - Leela Chess Zero hidden layer outputs for each game position

### Metadata Group (`/metadata/`)
All metadata arrays are length N and aligned with `lc0_hidden`:

| Dataset | Type | Description |
|---------|------|-------------|
| `White` | uint32 | Player ID for white |
| `Black` | uint32 | Player ID for black |
| `WhiteElo` | uint16 | White player rating |
| `BlackElo` | uint16 | Black player rating |
| `WhiteRatingDiff` | int16 | White rating change |
| `BlackRatingDiff` | int16 | Black rating change |
| `Opening` | uint32 | Opening ID |
| `Termination` | string | Game end reason |
| `Result` | uint8 | winner, -1 for black, 1 for white, 0 for draw |

**Index alignment**: `lc0_hidden[i]` and all `metadata/*[i]` correspond to the same game.
