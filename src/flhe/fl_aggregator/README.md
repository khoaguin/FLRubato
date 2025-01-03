# Fl Aggregator

Federated Learning (FL) Aggregator [SyftBox](https://syftbox-documentation.openmined.org) API.

The `fl_agregator` API is featured in the tutorial **Getting Started with FL on SyftBox**. 
Read more [here](https://syftbox-documentation.openmined.org/tutorials/federated-learning/getting-started/).

## Usage

**1. Install the API**

```bash
git clone https://github.com/OpenMined/fl_aggregator

cp -r ./fl_aggregator <SYFTBOX_DATADIR>/apis/  # default: ~/SyftBox
```

**Note**: `<SYFTBOX_DATADIR>` refers to the SyftBox data directory, according to your
SyftBox installation (default: `$HOME/SyftBox`).

**2. Agree on roles in the FL flow**

For example: Aggregator: `<a@openmined.org>`; Clients: `<b@openmined.org>`,`<c@openmined.org>`

- Aggregator must decide and share the model architecture with the clients.
- Aggregator will also provide a seed model weights which will be fine-tuned by each participant.

**3. Setup the FL config**

Create a `fl_config.json` configuration file including the following information:

- `"project_name"`: name assigned to the FL experiment.
- `"aggregator"`: designated aggregator datasite.
- `"participants"`: list of the designated client datasites.
- `"rounds"`: number of FL training rounds.
- `"model_arch"`: the Python module containing the ML model implementation.
- `"model_weight"`: model parameters file
- `"epoch"`: number of training epochs to run on each clients,
- `"learning_rate"`: learning rate of the optimizer.

Please see `samples/launch_config/fl_config.json` for an example configuration file:

```json
{
    "project_name": "MNIST_FL",
    "aggregator": "a@openmined.org",
    "participants": ["b@openmined.org","c@openmined.org"],
    "rounds": 3,
    "model_arch": "model.py",
    "model_weight": "global_model_weight.pt",
    "epoch": 10,
    "learning_rate": 0.1
}
```

**4. Start the FL experiment**

1. To start the FL experiment, the following files needs to be copied in 
`<SYFTBOX_DATADIR>/datasites/<aggregator_email>/api_data/fl_aggregator/launch` directory:

- `fl_config.json`
- `global_model_weight.pt`
- `model.py`

2. Copy the test dataset to `<SYFTBOX_DATADIR>/datasites/<aggregator_email>/private/fl_aggregator` 

An example test dataset could be found in `./samples/test_data`.

If this directory isn't available, either run the syftbox client with fl_aggregator API installed OR create it manually.

Once the files are in the `launch` folder, the API will create a folder named after the `project_name` specified in the `file_config.json` in the `running` folder.

Inside this folder the API will create a folder for each client datasite, where received updates in each round are gathered. 
Each of these folders will have a `._syftperm` file containing the appropriate permissions granting read/write access to the folder.

Finally, the API send a request to each participant to join the FL flow.

Please see below an example of the resulting folders structure during the FL execution.

```plaintext
api_data
└── fl_aggregator
    ├── launch
    │   ├── config.json
    │   ├── model_arch.py
    │   ├── global_model_weights.py
    └── running
        └── my_cool_fl_proj
            ├── fl_clients 
            │   ├── a@openmined.org
            │   ├── b@openmined.org
            │   ├── c@openmined.org
            ├── agg_weights  # to store aggregator's weights for each round
            ├── config.json  # moved from the launch folder after the app start
            ├── global_model_weights.pt
            ├── model_arch.py  # moved from the launch folder
            └── state.json
    └── done
        └── my_cool_fl_proj
            └── aggregated_model_weights.pt
```

**5. Monitoring the FL experiment**

- Visit the aggregator's dashboard to monitor the FL progress
which is available at `http://server_url/datasites/<aggregator_email>/fl/<project_name>/`, with `<project_name>` matches the name of the FL experiment setup in the `fl_config.json` file.

The server_url depends on which the server the client is running on.
- `syftbox.openmined.org` (for the public server)
- `localhost:5001` (when running locally, with default configuration)

## Running in dev mode

> ⚠️ Make sure you have the latest version of the SyftBox repository cloned on your local machine:
>
> ```bash
> git clone https://github.com/OpenMined/syft.git
> ```
> Also, make sure you have [just](https://github.com/casey/just) installed.

**Note**: The following commands need to be run within the `syft` directory (the repository root), unless otherwise indicated.

### **Set up**: 

First launch a local SyftBox server, and clients for all datasites participating in the FL flow:

- `just rs`: run SyftBox local dev server on port `5001`
- `just rc a`: creates a SyftBox client for `a@openmined.org`. (Repeat this for all the clients)

Each of those commands need to be run in a separate terminal session, or within the same terminal
using tools like [`tmux`](https://github.com/tmux/tmux/wiki) (recommended!).


### **Install the aggregator and the client APIs**: 

Aggregator:
- `git clone https://github.com/OpenMined/fl_aggregator` 
- `cp -R ./fl_aggregator ./clients/a@openmined.org/apis` 

Client(s):

- `git clone https://github.com/OpenMined/fl_client` 
- `cp -R ./fl_aggregator ./clients/b@openmined.org/apis`
- `cp -R ./fl_aggregator ./clients/c@openmined.org/apis`