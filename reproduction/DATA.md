# Shared Sudoku data

TRM and the future TRM-ODE must use the same generated files and `manifest.json` in this directory. Large data files are excluded from Git.

From the repository root:

```bash
/venv/main/bin/python trm/data.py
```

The builder pins `sapientinc/sudoku-extreme` to revision `58942f96baeb572ca3127e2a9e9c70f330783d6b`, fixes seed 0, records original indices and SHA-256 hashes, and creates 1,000 training puzzles, 1,000 independent validation puzzles, 422,786 official test puzzles, and a fixed pool of 1,001 variants per training puzzle (original plus 1,000 augmentations).

The original paper's exact 1K subset was not published. This is a deterministic reproduction of its data protocol, not a claim of identical original indices. Existing data with a manifest is verified; a nonempty incomplete directory is not overwritten.


The generated data directory is created by the builder in a fresh clone. Training arrays and their manifest stay local and are excluded from Git.
