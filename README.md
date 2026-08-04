# learning-ml

Notes and practice code from working through PyTorch / machine learning fundamentals.
Everything here is small, self-contained, and meant for learning rather than reuse.

## Contents

### `pytorch_basics/`

| File | What it does |
| --- | --- |
| `spiral_dataset.py` | Generates the classic three-class spiral dataset with NumPy and plots it. |
| `tanh_from_scratch.py` | Implements `tanh` from its exponential definition and plots it against a range of inputs. |
| `circles_classifier.py` | Binary classifier on `make_circles` — a small `nn.Module` trained with `BCEWithLogitsLoss`. |
| `moons_classifier.py` | Binary classifier on `make_moons` — a 2×100-unit ReLU network, with decision boundaries plotted for train and test. |
| `spiral_classifier.py` | Multi-class classifier on the spiral dataset from `spiral_dataset.py`. |
| `helper_functions.py` | Plotting / accuracy helpers from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning). |

## Requirements

```bash
pip install torch numpy pandas matplotlib scikit-learn requests
```

## Running

Each script has its own `main()` and runs standalone:

```bash
python pytorch_basics/moons_classifier.py
```

The scripts import `helper_functions.py` from the working directory, so run them from
inside `pytorch_basics/` (or make sure that file is on your path).
