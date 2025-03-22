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
[group('run')]
run:
    go run src/main.go

[group('hhe')]
run-hhe:
    go run src/hhe_fedavg/hhe_fedavg.go

[group('hhe')]
build-hhe:
    go build -o /dev/null src/hhe_fedavg/hhe_fedavg.go

[group('hhe')]
lint-hhe:
    # cd src/hhe_fedavg && golangci-lint run
    go vet src/hhe_fedavg/hhe_fedavg.go


# ---------------------------------------------------------------------------------------------------------------------
[group('test')]
test:
    cd Rubato-server/ckks_fv
    go test -timeout=0s -bench=BenchmarkRtFRubato


# ---------------------------------------------------------------------------------------------------------------------
[group('reset')]
reset:
    just reset-ciphertexts

# Clean all directories in ciphertexts/ but preserve Ciphertexts.go file
[group('reset')]
reset-ciphertexts:
    find ciphertexts/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;

