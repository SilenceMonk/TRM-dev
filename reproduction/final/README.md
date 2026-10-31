# Completed TRM reproduction

All 65,100 optimizer updates and the full 422,786-puzzle official test completed.

| Metric | Validation (1,000) | Official test (422,786) |
| --- | ---: | ---: |
| Exact puzzle accuracy | 87.4000% | 86.1556% |
| Blank-cell accuracy | 93.1964% | 92.4553% |
| All-cell accuracy | 95.3222% | 94.8023% |

The official test solved 364,254/422,786 puzzles exactly. Its Wilson 95% interval is [86.0512%, 86.2594%]. The 87.4% paper reference is a test score; this run's validation accuracy of 87.4% should not be substituted for its official test accuracy of 86.1556%.

The actual official test result is 1.2444 percentage points below the paper reference, within the planned approximate ±2 percentage-point tolerance. This is one seed with a newly fixed 1K subset, not an exact match to the unpublished author subset. Training followed the official README 50K-epoch recipe.

- `test_metrics.json`: complete official test counts and metrics.
- `validation_metrics.json`, `validation_history.csv`, `best.json`: final and periodic held-out validation.
- `status.json`, `RESULTS.md`: completed-run summary.
- `test_signature.json`, `provenance.json`: evaluated checkpoint hash and evaluation settings.
- `config.json`, `environment.json`, `code_manifest.json`, `data_manifest.json`: reproducibility records including split indices and file hashes.

The full test evaluated the final EMA checkpoint at step 65,100. It is also the checkpoint with the best validation score. No Sudoku solver or constraint postprocessing was applied.

Model weights, optimizer state, generated arrays, predictions, and full logs remain on the GPU instance and are excluded from Git. TRM-ODE remains an unimplemented scaffold.
