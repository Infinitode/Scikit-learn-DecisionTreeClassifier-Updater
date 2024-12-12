# Scikit-learn `DecisionTreeClassifier` Updater

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub top language](https://img.shields.io/github/languages/top/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub License](https://img.shields.io/github/license/infinitode/scikit-learn-decisiontreeclassifier-updater)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/infinitode/scikit-learn-decisiontreeclassifier-updater)
[![GitHub Repo stars](https://img.shields.io/github/stars/infinitode/scikit-learn-decisiontreeclassifier-updater)](https://github.com/infinitode/scikit-learn-decisiontreeclassifier-updater/stargazers)

This repository contains the necessary code and tools to upgrade `DecisionTreeClassifier` models from `scikit-learn==1.3.x` or lower to the newer versions of `scikit-learn`.

## Features

- Model Converter: Extracts and upgrades components of a Scikit-learn model for compatibility with newer versions than `1.3.x`.
- Model Upgrader: Reconstructs and updates the model using the modified components.
- Open-source and easy-to-use CLI commands.

## Installation

Clone this repository:
```bash
git clone https://github.com/Infinitode/repo-name.git
cd repo-name
```

## Usage

### Python Script Files

You can modify the Python script files to convert and upgrade your models. These files are located in `new_environment` and `old_environment` respectively.

**1. Set up `old_environment`**

- Start with the `old_environment`.
- You'll need to install the older versions of both `scikit-learn` and `numpy` in this environment.
- Make sure to use the same Python version your model was saved in.

> [!NOTE]
> It needs to be similar to your `original` environment in which you saved your model.

**2. Modify the script under `old_environment` and run**

Modify the script file, and run it, it will convert your model and store the files that we need for later.

**3. Set up `new_environment`**

- Create a new environment for `new_environment`.
- Install the latest/target versions of `scikit-learn` and `numpy`.
- Use the latest/target Python version.

**4. Modify the script under `new_environment` and run**

Modify the script file and run it. It will use the files created in `step 2` to create a new model, that is compatible with your chosen version.

### CLI

You can use the CLI tools to quickly convert your model files.
- `model_converter.py` - Initial conversion. Extracts and upgrades components of the model for compatibility with newer versions than `1.3.x`. Works in `original environment` (older `scikit-learn`, etc.).
- `model_upgrader.py` - Final conversion. Reconstructs and updates the model using the modified components. Works in `new environment` (updated `scikit-learn`, etc.)

#### Model Converter
```bash
python model_converter.py \
    --model_path path/to/original_model.pkl \
    --shell_path path/to/classifier_shell.pkl \
    --tree_state_path path/to/tree_state.pkl \
    --upgraded_tree_state_path path/to/upgraded/tree_state.pkl \
    --model_txt_path path/to/model.txt
```

**Explanation of arguments:**
- `--model_path`: Path to your saved model.
- `--shell_path`: Output path for the classifier shell (e.g. shell.pkl).
- `--tree_state_path`: Output path for the tree state (e.g. tree_state.pkl).
- `--upgraded_tree_state_path`: Output path for the upgraded tree state (e.g. upgraded_tree_state.pkl).
- `--model_txt_path`: Output path for the model's inputs and outputs (e.g. model.txt).

#### Model Upgrader
```bash
python model_upgrader.py \
    --shell_path path/to/classifier_shell.pkl \
    --upgraded_tree_state_path path/to/upgraded/tree_state.pkl \
    --model_txt_path path/to/model.txt \
    --output_path path/to/updated_classifier.pkl
```

**Explanation of arguments:**
- `--shell_path`: Path to the classifier shell (e.g. shell.pkl).
- `--upgraded_tree_state_path`: Path to the upgraded tree state (e.g. upgraded_tree_state.pkl).
- `--model_txt_path`: Path to the model's inputs and outputs (e.g. model.txt).
- `--output_path`: Output path for the final model (e.g. upgraded_model.pkl).

> [!TIP]
> You can use `--help` in the commandline for usage instructions.
>
> ```bash
> python model_upgrader.py --help
> ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## Acknowledgments

This project was inspired by the need to adapt Scikit-learn models to newer versions and ensure compatibility in various environments.