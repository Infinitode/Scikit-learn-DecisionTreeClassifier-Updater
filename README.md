# Scikit-learn Model Updater

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub top language](https://img.shields.io/github/languages/top/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub License](https://img.shields.io/github/license/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/infinitode/scikit-learn-decisiontreeclassifier-updater)
[![GitHub Repo stars](https://img.shields.io/github/stars/infinitode/scikit-learn-decisiontreeclassifier-updater)](https://github.com/infinitode/scikit-learn-decisiontreeclassifier-updater/stargazers)

This repository contains a comprehensive set of tools to upgrade a wide variety of Scikit-learn models from older versions (e.g., `<=1.3.x`) to be compatible with newer Scikit-learn environments.

## Features

- **Model Converter**: Extracts and upgrades the internal components of a Scikit-learn model.
- **Model Upgrader**: Reconstructs and updates the model using the upgraded components.
- **Extensive Model Support**: Handles a wide range of models, from simple scalers to complex pipelines.
- **Modular Architecture**: Uses a handler-based system that can be easily extended to support new model types.
- **Unified CLI**: A single, easy-to-use command-line interface for all operations.

## Supported Models

This tool supports a wide range of Scikit-learn models, including:

- **Tree-Based Models**:
  - `DecisionTreeClassifier`
  - `DecisionTreeRegressor`
  - `RandomForestClassifier`
  - `RandomForestRegressor`
- **Scalers**:
  - `StandardScaler`
  - `MinMaxScaler`
- **Nearest Neighbor Models**:
  - `KNeighborsClassifier`
  - `KNeighborsRegressor`
  - `NearestNeighbors`
- **Support Vector Machines (SVMs)**:
  - `SVC`
  - `SVR`
- **Linear Models**:
  - `LogisticRegression`
  - `Ridge`
  - `Lasso`
- **Decomposition Models**:
  - `PCA`
  - `IncrementalPCA`
  - `KernelPCA`
- **Preprocessing Models**:
  - `PolynomialFeatures`
  - `OneHotEncoder`
- **Pipeline Models**:
  - `Pipeline`
  - `FeatureUnion`
  - `ColumnTransformer`

## Installation

Clone this repository:
```bash
git clone https://github.com/Infinitode/repo-name.git
cd repo-name
```

## Usage

The model upgrade process is split into two main steps: `convert` and `upgrade`.

**1. Set up `old_environment`**

- Start with an environment that has the **older** versions of `scikit-learn` and `numpy` installed (the environment where your model was originally saved).
- Make sure to use the same Python version your model was saved in.

**2. Convert the Model**

Run the `convert` command to extract the model's components and prepare them for the upgrade. This step should be run in your `old_environment`.

```bash
python main.py convert \
    --model_path path/to/original_model.pkl \
    --shell_path path/to/model_shell.pkl \
    --upgraded_tree_state_path path/to/upgraded_state.pkl \
    --model_json_path path/to/model_metadata.json
```

**3. Set up `new_environment`**

- Create a new environment.
- Install the **latest/target** versions of `scikit-learn` and `numpy`.
- Use a modern/target Python version.

**4. Upgrade the Model**

Run the `upgrade` command to reconstruct the model using the files generated in the `convert` step. This step should be run in your `new_environment`.

```bash
python main.py upgrade \
    --shell_path path/to/model_shell.pkl \
    --upgraded_tree_state_path path/to/upgraded_state.pkl \
    --model_json_path path/to/model_metadata.json \
    --output_path path/to/upgraded_model.pkl
```

### Command-Line Arguments

You can use the `--help` flag to get more information about the commands and their arguments.

```bash
python main.py --help
python main.py convert --help
python main.py upgrade --help
```

#### `convert` arguments:
- `--model_path`: Path to your original saved model.
- `--shell_path`: Output path for the model shell.
- `--upgraded_tree_state_path`: Output path for the upgraded model state.
- `--model_json_path`: Output path for the model's metadata.

#### `upgrade` arguments:
- `--shell_path`: Path to the model shell file created during conversion.
- `--upgraded_tree_state_path`: Path to the upgraded model state file.
- `--model_json_path`: Path to the model's metadata file.
- `--output_path`: Output path for the final, upgraded model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## Acknowledgments

This project was inspired by the need to adapt Scikit-learn models to newer versions and ensure compatibility in various environments.