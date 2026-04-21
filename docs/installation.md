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
git clone https://github.com/NREL/routee-powertrain.git
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
