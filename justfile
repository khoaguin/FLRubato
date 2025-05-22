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
reset-hhe:
    rm -rf **/__pycache__/
    rm -rf logs/
    rm -rf weights/MNIST/symmetric_encrypted/
    rm -rf weights/MNIST/he_encrypted/

reset-all:
    rm -rf **/__pycache__/
    rm -rf logs/
    rm -rf weights/MNIST/symmetric_encrypted/
    rm -rf weights/MNIST/he_encrypted/
    rm -rf weights/MNIST/plain/

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
[group('mnist')]
prepare-mnist-data:
    uv run src/flhhe/mnist/dataset.py

[group('mnist')]
train-mnist-model-save-weights:
    uv run -m flhhe.mnist.train --save-weights

train-mnist-model:
    uv run -m flhhe.mnist.train

run-mnist-fed-avg:
    uv run -m flhhe.mnist.fed_avg

run-mnist-fed-avg-he:
    uv run -m flhhe.mnist.fed_avg_he

run-mnist-e2e:
    just prepare-mnist-data
    just train-mnist-model-save-weights
    just run-mnist-fed-avg
    just run-he
    just run-mnist-fed-avg-he

    just run-hhe
    just test-hhe
    just run-mnist-fed-avg-hhe

### Go
# ---------------------------------------------------------------------------------------------------------------------
[group('he')]
run-he:
    go run src/he_fedavg/he_fedavg.go

# ---------------------------------------------------------------------------------------------------------------------
[group('hhe')]
run-hhe:
    echo "{{ _cyan }}Running HHE FedAvg {{ _nc }}"
    go run src/hhe_fedavg/hhe_fedavg.go
    echo "{{ _green }}HHE FedAvg completed {{ _nc }}"

# ---------------------------------------------------------------------------------------------------------------------
[group('test')]
test-hhe:
    echo "{{ _cyan }}Running end-to-end tests {{ _nc }}"
    go test src/hhe_fedavg/hhe_fedavg_test.go -v
    echo "{{ _green }}Test execution completed {{ _nc }}"
