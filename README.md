# learning-ml

Notes and practice code from working through PyTorch / machine learning fundamentals.
Everything here is small, self-contained, and meant for learning rather than reuse.

## Contents

### `ReLu/`

| File | What it does |
| --- | --- |
| `Buffer.py` | Generates the classic three-class spiral dataset with NumPy and plots it. |
| `hyerbolic tangent.py` | Implements `tanh` from its exponential definition and plots it against a range of inputs. |
| `model_0.py` | Binary classifier on `make_circles` — a small `nn.Module` trained with `BCEWithLogitsLoss`. |
| `model_1.py` | Binary classifier on `make_moons` — a 2×100-unit ReLU network, with decision boundaries plotted for train and test. |
| `example_0.py` | Multi-class classifier on the spiral dataset from `Buffer.py`. |
| `helper_functions.py` | Plotting / accuracy helpers from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning). |

## Requirements

```bash
pip install torch numpy pandas matplotlib scikit-learn requests
```

## Running

Each script has its own `main()` and runs standalone:

```bash
python ReLu/model_1.py
```

The scripts import `helper_functions.py` from the working directory, so run them from
inside `ReLu/` (or make sure that file is on your path).
