# Recursive Reasoning Network (RRN)

This repository contains a PyTorch implementation of the Recursive Reasoning Network (RRN). I improved the original paper's KGE model with batched relation updates, which allows for maximal GPU utilization.

## Setting up the environment

I recommend using Python 3.9.20 - because of compatibility issues with certain packages in `reldata` (see later) - and have provided a `requirements.txt` file for installing the necessary dependencies.

From the root directory of this repository, run the following commands to create and activate a new conda environment, and install the required packages:

```bash
conda env create -f environment.yml
conda activate RRN
```

## Generating train and test data

To generate training and testing data, I make use of the source code provided by the original author of the RRN paper:

-   [Family Tree dataset generator](https://github.com/phohenecker/family-tree-data-gen)
-   [Country dataset generator](https://github.com/phohenecker/country-data-gen)

Both of these data generators use the [rel-data](https://github.com/phohenecker/rel-data) library for generating synthetic data, which is not so easy to use because of missing documentation and _outdated_ dependencies, but with the above mentioned Python version and the provided `requirements.txt` file, it should work fine.

---

### DLV

The family tree data generator makes use of the DLV system in order to perform symbolic reasoning over family trees by
means of the ontology mentioned above.
Therefore, you have to download the DLV executable for your platform from the
[official website](http://www.dlvsystem.com/dlv/#1).

After you have downloaded and extracted the DLV executable, copy the path to the executable file. Then, export this path as an environment variable:

```bash
echo "export DLV_PATH=/path/to/dlv/executable" >> ~/.zshrc
source ~/.bashrc
```

---

### Generating the _Family Tree_ data

To generate the _family tree_ datasets, navigate to the `data/family-tree` directory and make an `out` directory to store the generated data:

```bash
cd data/family-tree
mkdir -p out
```

Then, you can run the `generate.sh` script with the desired options:

```bash
./generate.sh [options] /path/to/dlv/executable
```

Because we set the `DLV_PATH` environment variable above, we can simply run:

```bash
./generate.sh [options] $DLV_PATH
```

#### Options

-   `--seed`: Random seed for data generation
-   `--max-branching-factor`: (default: `5`)
-   `--max-tree-depth`: (default: `5`)
-   `--max-tree-size`: (default: `26`)
-   `--negative-facts`: Boolean flag indicating whether to generate negative facts (default: `false`)
-   `--num-samples`: Number of individuals in the family tree
-   `--output-dir`: Directory to save the generated data (default: `./out`)
-   `--stop-prob`: (default: `0.0`)

#### Generating 200 train and 20 test samples with default parameters

Again, from the `data/family-tree` directory, run:

```bash
./generate.sh --num-samples 200 --output-dir out/train-200 --seed 1 $DLV_PATH
./generate.sh --num-samples 20 --output-dir out/test-20 --seed 2 $DLV_PATH
```

#### Generating 5000 train and 500 test samples like in the original paper

```bash
./generate.sh --num-samples 5000 --output-dir out/train-5000 --seed 3 $DLV_PATH
./generate.sh --num-samples 500 --output-dir out/test-500 --seed 4 $DLV_PATH
```

## Training the model

To train the model, make sure you have checked `train.py` to configure the number of epochs, learning rate, number of knowledge graphs, etc.
Then, run:

```bash
python3 train.py <checkpoint-subdir>
```

Where `<checkpoint-subdir>` is the name of the subdirectory inside the `checkpoints/` directory where you want to save the model checkpoints.
I recommend using the timestamp of the training as the subdirectory name, e.g., `2024-06-01_12-00-00`.

You can do this as follows:

```bash
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
python3 RRN/train.py $TIMESTAMP
```

## Testing the model

To load the checkpoint from the trained model and evaluate it on the test data, run:

```bash
python3 RRN/test.py <checkpoint-subdir>
```

Assuming you used the timestamp as the subdirectory name, we can first retreive the latest timestamp and then run the test script:

```bash
LATEST_CHKPT=$(ls -td ${REPO}/checkpoints/*/ | head -n 1)
python3 test.py $LATEST_CHKPT
```

## Running on HPC cluster

> _NOTE_:
> The DTU HPC clusters I use are **LSF** clusters.

To train the model on a High Performance Computing (HPC) cluster, you can use the provided `jobscript.sh`.

### Setup SSH

First, create an SSH key pair if you don't have one already:

```bash
ssh-keygen -t ed25519
```

Choose a password for the key when prompted. When the key is created, print the public key with:

```bash
cat ~/.ssh/id_ed25519.pub
```

Now, copy the public key to the HPC server. You can do this by logging into the HPC server using your credentials and adding the public key to the `~/.ssh/authorized_keys` file, or simply using `ssh-copy-id`:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <your-username>@<host>
```

Next, setup the SSH details of your `<host>` in `~/.ssh/config` (Unix), so that you don't have to type in the full details every time you want to connect:

```txt
Host <host>
  HostName <ip-address-or-domain>
  Port <port>
  User <your-username>
  IdentityFile id_ed25519
  AddKeysToAgent yes
  UseKeychain yes
```

### Connect to HPC

Connect to the HPC using the saved SSH configuration:

```bash
ssh <host>
```

Type in the SSH (and host) password(s) when prompted.
Switch to the assigned node if necessary:

```bash
linuxsh
```

### Create a project folder

Create a project folder on the HPC cluster to store the code and data, e.g.,

```bash
mkdir ~/rrn
```

... and clone this repository directly into the project folder:

```bash
git clone https://github.com/schependom/recursive-reasoning-network.git ~/rrn
```

### Python environment

First, make sure the **correct Python version is loaded**:

```bash
module load python3/3.9.19 # or 3.9.20
```

Now, inside the project folder, create a `venv` Python environment:

```bash
python3 -m venv .venv
```

To activate the environment, run:

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the beginning of the command line.
You can (but right now don't have to) deactivate the environment with:

```bash
deactivate
```

To install packages inside the `venv`, use:

```bash
python3 -m pip install <packages>
```

We are used to simply using `pip3`, but this is the recommended and correct way of installing packages. The `-m` flag in Python allows us to run modules as scripts. This way we ensure that the module is located in your current python environment, not the global python installation.

Now simply install from the `requirements.txt` file:

```bash
python3 -m pip install -r requirements.txt
```

To update the packages using the `requirements.txt` file:

```bash
python3 -m pip install --upgrade -r requirements.txt
```

### Submitting jobs

To submit a job to the HPC cluster, use the provided `jobscript.sh`.
E.g. `bsub < jobscript.sh` on LSF clusters.

## References

> **Ontology Reasoning with Deep Neural Networks.** \
> Patrick Hohenecker and Thomas Lukasiewicz.
> Preprint at [https://arxiv.org/abs/1808.07980](https://arxiv.org/abs/1808.07980) (2018).

-   [rel-data](https://github.com/phohenecker/rel-data)

-   [Family Tree dataset generator](https://github.com/phohenecker/family-tree-data-gen)

-   [Country dataset generator](https://github.com/phohenecker/country-data-gen)
