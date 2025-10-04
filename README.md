# Scikit-learn Model Updater

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub top language](https://img.shields.io/github/languages/top/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub License](https://img.shields.io/github/license/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/infinitode/scikit-learn-decisiontreeclassifier-updater)
[![GitHub Repo stars](https://img.shields.io/github/stars/infinitode/scikit-learn-decisiontreeclassifier-updater)](https://github.com/infinitode/scikit-learn-decisiontreeclassifier-updater/stargazers)

This repository contains a comprehensive set of tools to upgrade a wide variety of Scikit-learn models from older versions (e.g., `<=1.3.x`) to be compatible with newer Scikit-learn environments.

## ⚠️ Important Notice: Model Refitting

During the upgrade process, some models are **refit** with their original training data, while others have their internal components directly upgraded. It is crucial to understand which models are affected:

- **Models that ARE refit**:
  - `KNeighborsClassifier`
  - `KNeighborsRegressor`
  - `NearestNeighbors`

- **Why is this necessary?**
  - For these models, the internal data structures are not easily portable across Scikit-learn versions. To ensure compatibility, the tool extracts the original training data (`X` and `y`) and uses it to refit a new model instance in the target environment.

- **What does this mean for you?**
  - The refitting process ensures that your model is compatible with the new environment, but it is essentially a new model trained on the same data. The performance should be identical, but the model object itself is new.

- **Pipelines**:
  - If your `Pipeline`, `FeatureUnion`, or `ColumnTransformer` contains one of the models listed above, that specific step will be refit, while other steps will be upgraded directly.

All other supported models are upgraded by transferring their learned attributes to a new model instance without refitting.

> [!NOTE]
> Our internal saving logic uses `pickle` and not `joblib`. Refrain from using `joblib` to load converted models after they have been generated if not using the command-line interface directly.

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

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Infinitode/Scikit-learn-Model-Updater.git
   cd Scikit-learn-Model-Updater
   ```

2. **Set up your environments**:
   This tool requires two separate Python environments:
   - **`old_environment`**: Where your original model was trained and saved. It should have the **older** versions of `scikit-learn`, `numpy`, and other relevant libraries.
   - **`new_environment`**: Where you want to use the upgraded model. It should have the **newer/target** versions of `scikit-learn` and `numpy`.

   It is highly recommended to use virtual environments (e.g., `venv` or `conda`) to manage these dependencies.

## How It Works: The Two-Step Process

The model upgrade is a two-step process: `convert` and `upgrade`.

### Step 1: Convert the Model (in `old_environment`)

This step extracts the model's internal components and prepares them for the upgrade.

1. **Activate your `old_environment`**.
2. Run the `convert` command:
   ```bash
   python main.py convert \
       --model_path path/to/original_model.pkl \
       --shell_path path/to/model_shell.pkl \
       --upgraded_tree_state_path path/to/upgraded_state.pkl \
       --model_json_path path/to/model_metadata.json
   ```
   This will generate three files: a model "shell", the upgraded state, and a JSON metadata file.

### Step 2: Upgrade the Model (in `new_environment`)

This step reconstructs the model in the new environment using the files generated in the `convert` step.

1. **Activate your `new_environment`**.
2. Run the `upgrade` command:
   ```bash
   python main.py upgrade \
       --shell_path path/to/model_shell.pkl \
       --upgraded_tree_state_path path/to/upgraded_state.pkl \
       --model_json_path path/to/model_metadata.json \
       --output_path path/to/upgraded_model.pkl
   ```
   This will create the final, upgraded model file.

## Testing

To ensure the tool is working correctly, you can run the included tests. The tests are named as follows:
- `test_conversion.py`: Run in the old environment to test old model conversion.
- `test_upgrade.py`: Run in the new environment to test model upgrade.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## Acknowledgments

This project was inspired by the need to adapt Scikit-learn models to newer versions and ensure compatibility in various environments.
