# Installation

(In case `pip` is unavailable, use `pip3`)

## From PyPI

To install the base package for model prediction, we recommend you use `pip`:

```bash
pip install routee.powertrain
```

## From Source

To install the package from source, you can clone the repository and install the package using `pip`:

```bash
git clone https://github.com/NatLabRockies/routee-powertrain.git
cd routee-powertrain
pip install .
```

## Model Training

Model training requires a couple of extra dependencies that are not required for model prediction.
Each training pipeline has its own set of dependencies.

### Scikit-learn

To install the dependencies for the scikit-learn training pipeline, use the following command:

```bash
pip install routee.powertrain[scikit]
```

This should support usage of the following trainers:

- `SklearnRandomForestTrainer`

### NGBoost

To install the dependencies for the NGBoost training pipeline:

```bash
pip install routee.powertrain[ngboost]
```

This should support usage of the following trainers:

- `NGBoostTrainer`

### PyTorch (for CNN training)

To install the dependencies for the 1D-CNN training pipeline (PyTorch + ONNX export tooling):

```bash
pip install routee.powertrain[pytorch]
```

This should support usage of the following trainers:

- `CNNTrainer`

## Plotting

Visualization helpers like `pt.visualize_features` and `pt.contour_plot` require `matplotlib`. Install the `plot` extra to enable them:

```bash
pip install routee.powertrain[plot]
```

## Everything (development)

To install the package along with every optional extra and the development tooling (pytest, mypy, ruff, jupyter-book, etc.), use the `dev` extra:

```bash
pip install -e ".[dev]"
```
