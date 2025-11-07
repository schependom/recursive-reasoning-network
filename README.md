# Recursive Reasoning Network (RRN)

This repository contains a PyTorch implementation of the Recursive Reasoning Network (RRN). I improved the original paper's KGE model with batched relation updates, which allows for maximal GPU utilization.

## Setting up the environment

I recommend using Python 3.9.20 - because of compatibility issues with certain packages in `reldata` (see later) - and have provided a `requirements.txt` file for installing the necessary dependencies.

From the root directory of this repository, run the following commands to create and activate a new conda environment, and install the required packages:

```bash
conda create -n rrn python=3.9.20
conda activate rrn
pip install -r requirements.txt
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
export DLV_PATH=/path/to/dlv/executable
```

---

### Generating the _Family Tree_ data

To generate the _family tree_ datasets, run the following scripts:

```bash
./data/family-tree/generate.sh [options] /path/to/dlv/executable
```

For example, use the following command to generate the data with the default parameters:

```bash
./data/family-tree/generate.sh $DLV_PATH
```

#### Options

-   `--seed`: Random seed for data generation
-   `--max_branching_factor`: (default: `5`)
-   `--max_tree_depth`: (default: `5`)
-   `--max_tree_size`: (default: `26`)
-   `--negative_facts`: Boolean flag indicating whether to generate negative facts (default: `false`)
-   `--num_samples`: Number of individuals in the family tree
-   `--output_dir`: Directory to save the generated data (default: `./out`)
-   `--stop_prob`: (default: `0.0`)

#### Generating 200 train and 20 test samples with default parameters

```bash
./data/family-tree/generate.sh --num_samples 200 --output_dir data/family-tree/out/train $DLV_PATH
./data/family-tree/generate.sh --num_samples 20 --output_dir data/family-tree/out/test $DLV_PATH
```

#### Generating 5000 train and 500 test samples like in the original paper

```bash
./data/family-tree/generate.sh --num_samples 5000 --output_dir data/family-tree/out/train $DLV_PATH
./data/family-tree/generate.sh --num_samples 500 --output_dir data/family-tree/out/test $DLV_PATH
```

## Training the model

Work in progress.

## Testing the model

Work in progress.

## References

> **Ontology Reasoning with Deep Neural Networks.** \
> Patrick Hohenecker and Thomas Lukasiewicz.
> Preprint at [https://arxiv.org/abs/1808.07980](https://arxiv.org/abs/1808.07980) (2018).

-   [rel-data](https://github.com/phohenecker/rel-data)

-   [Family Tree dataset generator](https://github.com/phohenecker/family-tree-data-gen)

-   [Country dataset generator](https://github.com/phohenecker/country-data-gen)
