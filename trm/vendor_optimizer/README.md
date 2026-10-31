# Adam-atan2 Fused Operator

## Usage

Drop-in replacement of `torch.optim.AdamW`.
 
 - Doesn't support `foreach`, `fused` argument, as the optimizer is already fused
 - Doesn't support `amsgrad`, `maximize`, `capturable`, `differentiable` argument yet

```bash
pip install adam-atan2
```

```python
from adam_atan2 import AdamATan2

# All supported arguments are listed below
optim = AdamATan2(model.parameters(),
    lr=1e-3,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)
```

## Consistency Tests

We tested the consistency against reference AdamW-atan2 PyTorch implementation. To run tests, clone this repository, run pytest:

```bash
pip install -e .
pytest
```

## References

 - https://arxiv.org/pdf/2407.05872
