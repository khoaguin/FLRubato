# FLHHE

## Prerequisite

Set up the Python env:

```bash
just setup-venv
```

Make sure that you also have the [Go programming language](https://go.dev/doc/install) installed

## MNIST

Prepare MNIST dataset:

```bash
just prepare-mnist-data
```

This will create 3 datasets,

1. Partition 1: No labels 1, 3, 7
1. Partition 2: No labels 2, 5, 8
1. Partition 3: No labels 4, 6, 9

Train a simulated FL on 3 partitioned datasets
train-mnist-model-save-weights:
uv run -m flhhe.mnist.train --save-weights

train-mnist-model:
uv run -m flhhe.mnist.train

run-mnist-fed-avg:
uv run -m flhhe.mnist.fed_avg

run-mnist-fed-avg-he:
uv run -m flhhe.mnist.fed_avg_he
