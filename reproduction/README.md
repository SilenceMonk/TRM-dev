# Reproduction evidence

The completed training and full official test results are in [final/](final/README.md). Official test exact accuracy is 86.1556% over all 422,786 puzzles; held-out validation exact accuracy is 87.4% over 1,000 puzzles.

Files directly under this directory record the original resolved configuration, preflight checks, source hashes and hardware benchmarks. `snapshot.json` preserves the historical in-progress observation from the initial GitHub synchronization; it is not the final status.

`final/` contains the completed status, metrics, validation history, data split manifest and evaluated checkpoint hash. Checkpoints, generated arrays, predictions and full logs remain local.
