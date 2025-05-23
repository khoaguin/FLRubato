# FLHHE

_Secure Federated Learning using [Hybrid Homomorphic Encryption](https://github.com/KAIST-CryptLab/RtF-Transciphering)._

## Prerequisite

Please make sure that you have these installed:

1. The `just` command runner. Installation guide [here](https://github.com/casey/just?tab=readme-ov-file#installation)
2. The `uv` Python package and project manager. Installation guide [here](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
3. The [Go programming language](https://go.dev/doc/install)

### Python Environment

```bash
just setup-venv
```

This will create a vertual Set up the Python env and install the Python packages needed.
When it's done, please run

```sh
source .venv/bin/activate
```

and you are ready.

## MNIST

### Prepare MNIST dataset

```bash
just prepare-mnist-data
```

This will create 3 partitions for training:

1. Partition 1: No labels 1, 3, 7
1. Partition 2: No labels 2, 5, 8
1. Partition 3: No labels 4, 6, 9

And 4 test sets for evaluation:

1. Test set that only includes label 1, 3, 7
1. Test set that only includes label 2, 5, 8
1. Test set that only includes label 4, 6, 9
1. Test set that only includes all labels

The datasets will be saved into `data/MNIST/processed`

### Train a simulated FL on 3 partitioned datasets

```sh
just train-mnist-model-save-weights
```

This will train 3 models locally on the partitioned MNIST datasets. Weights will be saved into `weights/MNIST/plain`

### Evaluate Plaintext FedAvg

```sh
just run-mnist-fed-avg
```

This will do plaintext averaging of the 3 trained model, and then evaluate them on all test sets

### Plain Homomorphic Encrypted FedAvg

```sh
just run-he
```

Each client encrypts the plaintext weights in CKKS, and then the server does HE encrypted FedAvg

### Evaluate HE FedAvg

```ssh
just run-mnist-fed-avg-he
```

Evaluate the decrypted average weights from plain HE FedAvg

### HHE FedAvg

```ssh
just run-hhe
```

This will run the FedAvg protocol using HHE. Each client encrypts the plaintext weights into symmetric ciphertexts and sends to the server. The server transciphers the symmetric ciphertexts into HE ciphertexts, and finally does HE encrypted FedAvg

```sh
just test-hhe
```

This decrypts the HE avg ciphertext weights from the HHE protocol, and compare it with the one outputs by plain HE FedAvg

### Evaluate HHE FedAvg

```sh
just run-mnist-fed-avg-hhe
```

Evaluate the decrypted average weights from FedAvg using HHE

### Everything in one go

```sh
just run-mnist-e2e
```

Will run all the above in one go
