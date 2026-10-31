# Local experiment outputs

Generated logs, checkpoints, optimizer states, predictions, and progress files are excluded from Git. The active full run is `trm/sudoku_mlp_seed0/`; future ODE runs belong in `trm_ode/`.

The training service is `trm_reproduction`. Inspect `trm/sudoku_mlp_seed0/status.json` on the GPU instance for live progress. A run is not a successful reproduction merely because training has started or finished; the full official test and `replication_target_met` flag determine the recorded result.

Static preflight evidence and the resolved configuration are versioned separately under `reproduction/`. They are snapshots, not live training status or completed official test results.
