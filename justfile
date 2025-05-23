# ---------------------------------------------------------------------------------------------------------------------
# Guidelines for new commands
# - Start with a verb
# - Keep it short (max. 3 words in a command)
# - Group commands by context. Include group name in the command name.
# - Mark things private that are util functions with [private] or _var
# - Don't over-engineer, keep it simple.
# - Don't break existing commands
# - Run just --fmt --unstable after adding new commands
# ---------------------------------------------------------------------------------------------------------------------
# Define color codes for terminal output in a Justfile
_red := '\033[1;31m'
_cyan := '\033[1;36m'
_green := '\033[1;32m'
_yellow := '\033[1;33m'
_nc := '\033[0m'

# ---------------------------------------------------------------------------------------------------------------------
# Aliases
# alias t := test
# ---------------------------------------------------------------------------------------------------------------------
# Commands

@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------
[group('reset')]
reset:
    rm -rf **/__pycache__/
    rm -rf logs/
    rm -rf weights/

### Python
# ---------------------------------------------------------------------------------------------------------------------
[group('venv')]
setup-venv:
    #!/bin/bash
    if [ ! -d ".venv" ]; then
        uv venv
        source .venv/bin/activate
        uv sync && uv pip install -e .
    fi
    source .venv/bin/activate

# ---------------------------------------------------------------------------------------------------------------------
[group('mnist-python')]
prepare-mnist-data:
    echo "{{ _cyan }}Preparing MNIST data {{ _nc }}"
    uv run src/flhhe/mnist/dataset.py

[group('mnist-python')]
train-mnist-model-save-weights:
    echo "{{ _cyan }}Training MNIST model and saving weights {{ _nc }}"
    uv run -m flhhe.mnist.train --save-weights

[group('mnist-python')]
train-mnist-model:
    echo "{{ _cyan }}Training MNIST model {{ _nc }}"
    uv run -m flhhe.mnist.train

[group('mnist-python')]
run-mnist-fed-avg:
    echo "{{ _cyan }}Running MNIST FedAvg {{ _nc }}"
    uv run -m flhhe.mnist.fed_avg

[group('mnist-python')]
run-mnist-fed-avg-he:
    echo "{{ _cyan }}Running MNIST FedAvg using plain HE {{ _nc }}"
    uv run -m flhhe.mnist.fed_avg_he

[group('mnist-python')]
run-mnist-fed-avg-hhe:
    echo "{{ _cyan }}Running MNIST FedAvg using HHE {{ _nc }}"
    uv run -m flhhe.mnist.fed_avg_hhe


### Go
# ---------------------------------------------------------------------------------------------------------------------
[group('mnist-go')]
run-he:
    echo "{{ _cyan }}Running HE FedAvg {{ _nc }}"
    go run src/he_fedavg/he_fedavg.go

# ---------------------------------------------------------------------------------------------------------------------
[group('mnist-go')]
run-hhe:
    echo "{{ _cyan }}Running HHE FedAvg {{ _nc }}"
    go run src/hhe_fedavg/hhe_fedavg.go
    echo "{{ _green }}HHE FedAvg completed {{ _nc }}"

# ---------------------------------------------------------------------------------------------------------------------
[group('mnist-go')]
test-hhe:
    echo "{{ _cyan }}Running HHE MNIST weight tests {{ _nc }}"
    go test src/hhe_fedavg/hhe_fedavg_test.go -v
    echo "{{ _green }}HHE MNIST weight tests completed {{ _nc }}"

# ---------------------------------------------------------------------------------------------------------------------
[group('mnist')]
run-mnist-e2e:
    just prepare-mnist-data
    just train-mnist-model-save-weights
    just run-mnist-fed-avg

    just run-he
    just run-mnist-fed-avg-he

    just run-hhe
    just test-hhe
    just run-mnist-fed-avg-hhe