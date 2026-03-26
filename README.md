# Replication Package

**Paper:** The expert advantage concentrates in easy decisions — evidence from elite chess
**Journal:** Nature Human Behaviour (submitted)

## Overview

This package contains the analysis code for reproducing all figures, tables, and statistical results reported in the manuscript. The analyses use two datasets of elite chess games evaluated with Stockfish 18.

## Data

**Chess.com sample:** 40,546 blitz games (3+1) played by 256 titled players in Freestyle Friday (Chess960) and Titled Tuesday (standard) events. Collected via the Chess.com public API.

**Over-the-board sample:** 1,399 classical games (90+30) from FIDE Freestyle Chess events involving 448 elite players (mean FIDE Elo 2,625). Game records from publicly available PGN files.

**Engine evaluation:** Every game evaluated position-by-position using Stockfish 18 at two depths:
- Single-PV (0.5s per position, depth 15-22) for centipawn loss
- Multi-PV (depth 13, k=5 lines) for good-move density

The SQLite databases (`chesscom.db`, `chess960.db`) containing the evaluated games are available on request from the corresponding author. The raw game records are publicly available from Chess.com and FIDE.

## Scripts

| Script | Produces | Description |
|--------|----------|-------------|
| `fig1_hero_chesscom.py` | Figure 1 | Dose-response (panel a), density gradient (panel b), triage collapse (panel c) |
| `fig2_thinking_gap_chesscom.py` | Figure 2 | Format gap by thinking time and difficulty |
| `chesscom_formal_models.py` | Table 1, EDT 1, 2, 4 | Format effect, interaction models, density distribution, within-decile gaps |
| `chesscom_oster_experience.py` | EDT 2 (experience) | Experience non-attenuation, Oster bounds |
| `extended_data_nonlinearity.py` | EDT 3 | Dose-response nonlinearity, per-piece displacement, offset share bootstrap |
| `reviewer_analyses.py` | EDT 4 (robustness) | Move-level density interaction, time-allocation model |

## Requirements

```
python >= 3.9
numpy
pandas
scipy
statsmodels
matplotlib
scikit-learn
python-chess
```

Install: `pip install numpy pandas scipy statsmodels matplotlib scikit-learn python-chess`

## Engine

Stockfish 18 (open-source): https://stockfishchess.org/

## Running

Each script is self-contained and can be run independently:

```bash
python scripts/fig1_hero_chesscom.py
python scripts/fig2_thinking_gap_chesscom.py
python scripts/chesscom_formal_models.py
python scripts/chesscom_oster_experience.py
python scripts/extended_data_nonlinearity.py
python scripts/reviewer_analyses.py
```

Scripts expect the database files at `../../chess960/db/chesscom.db` and `../../chess960/db/chess960.db` (relative to the script location). Adjust the `DB_PATH` variable at the top of each script if your database is elsewhere.

## License

MIT
