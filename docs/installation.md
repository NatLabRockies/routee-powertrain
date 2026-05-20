# Installation
(In case `pip` is unavailable, use `pip3`)

## From PyPI

To install the base package for model prediction, we recommend you use `pip`:

```bash
pip install nrel.routee.powertrain
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

To install the dependencies for the scikit learn training pipeline, use the following command:

```bash
pip install nrel.routee.powertrain[scikit]
```

This should support usage of the following trainers:

- `SklearnRandomForestTrainer`

### Rust Smartcore

The rust smartcore training pipeline requires a rust compiler to be installed on your system.
One way to do this is to use [Anaconda](https://docs.anaconda.com/anaconda/install/):

Create a conda virtual environment as `routee`:

```bash
conda create --name routee
conda init
conda activate routee
```

```bash
conda install rust
```

Then, you'll have to build the python rust extension for powertrain:

```bash
pip install maturin

git clone https://github.com/NREL/routee-powertrain.git
cd routee-powertrain/rust
maturin develop --release
```

This will install the `powertrain_rust` extension and should support usage of the following trainers:

- `SmartcoreRandomForestTrainer`
