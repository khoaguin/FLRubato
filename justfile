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
[group('deps')]
install-syftbox-dev:
    #!/bin/bash
    if [ -d "libs/SyftBox" ]; then
        echo "SyftBox directory already exists in libs/"
    else
        echo "Creating libs directory and cloning SyftBox..."
        mkdir -p libs/
        cd libs/
        git clone https://github.com/OpenMined/syft.git SyftBox
        cd SyftBox
        uv pip install -e .
        echo "Installation completed"
    fi

# ---------------------------------------------------------------------------------------------------------------------
[group('run')]
run:
    go run src/main.go

# ---------------------------------------------------------------------------------------------------------------------
[group('test')]
test:
    cd Rubato-server/ckks_fv
    go test -timeout=0s -bench=BenchmarkRtFRubato
