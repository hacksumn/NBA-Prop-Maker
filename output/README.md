# Project Name

One-sentence description of what this NBA prediction project does.

## Goal
Describe the specific prediction task:
- game winner
- spread
- totals
- player props
- same game parlays
- grading / backtesting

## Current Status
See STATUS.md for current state.

## Repository Map
- /data_raw          raw source files
- /data_processed    cleaned / feature-ready data
- /src               main code
- /src/features      feature generation
- /src/models        training/inference
- /src/eval          evaluation / backtesting
- /reports           saved metrics and analysis
- /artifacts         trained models and manifests
- /tests             tests

## Canonical Workflow
1. Ingest raw data
2. Clean / normalize
3. Generate features
4. Split chronologically
5. Train model
6. Evaluate
7. Generate predictions
8. Grade predictions

## Quick Start
Example:
```bash
python -m src.ingest.run
python -m src.features.build
python -m src.models.train --config configs/baseline.yaml
python -m src.eval.walk_forward --config configs/baseline.yaml