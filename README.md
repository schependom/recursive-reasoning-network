# Recursive Reasoning Network (RRN)

This repository contains a PyTorch implementation of the Recursive Reasoning Network (RRN). I improved the original paper's KGE model with batched relation updates, which allows for maximal GPU utilization.

## Generating train and test data

To generate training and testing data, I make use of the source code provided by the original author of the RRN paper:

-   [Family Tree dataset generator](https://github.com/phohenecker/family-tree-data-gen)
-   [Country dataset generator](https://github.com/phohenecker/country-data-gen)

Both of these data generators use the [rel-data](https://github.com/phohenecker/rel-data) library for generating synthetic data, which is not so easy to use because of missing documentation and outdated dependencies.d

I recommend using Python 3.9.20 and have provided a `requirements.txt` file for installing the necessary dependencies.

### Setting up the environment

From the root directory of this repository, run the following commands to create and activate a new conda environment, and install the required packages:

```bash
conda create -n rrn python=3.9.20
conda activate rrn
pip install -r requirements.txt
```

### Generating the _Family Tree_ data

To generate the train-200, test-20, train-5000 and test-500 _family tree_ datasets, run the following script:

(WORK IN PROGRESS)

```bash
./data/family-tree/generate.sh
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
