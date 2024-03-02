# Italian VAT Judgement Prediction
## Installation
Python >= 3.9 is required. All the dependencies can be installed via pip:
```
pip install "vjp[all] @ git+https://github.com/Ball-Man/vjp-ita"
```
If you are only interested in one of the experiments, replace `all` with one of:
* zeroshot (includes one shot)
* fewshot

A minimal installation, able to run only the data exploration and cout based experiments:
```
pip install git+https://github.com/Ball-Man/vjp-ita
```

### Development
In order to run notebooks and display graphs, some extra dependencies are defined by `dev_requirements.txt` and can be installed with:
```
pip install -r dev_requirements.txt
```

### Notebooks
The given notebooks are set to run on a specific jupyter kernel having all of the above mentioned dependencies. The kernel is named `VJP`. When contributing to the project, this same kernel name shall be used. It can be crafted from scratch and added to a jupyter installation as follows.

On unix:
```
python -m venv venv
source venv/bin/activate
pip install [one of the variants above] -r dev_requirements.txt
python -m ipykernel install --user --name=VJP
```

On Windows (cmd):
```
python -m venv venv
venv\Scripts\activate.bat
pip install [one of the variants above] -r dev_requirements.txt
python -m ipykernel install --user --name=VJP
```

A user-wide jupyter installation is required. **On Windows**, `python` calls may be replaced by `py`. **On unix**, `python` may be replaced by `python3` and `pip` by `pip3`.

## Package structure
Results and analysis can be inspected directly opening the notebook. They can also be viewed directly on GitHub. Running them is only necessary if you want to reproduce or alter results. The notebooks:
* [data_exploration.ipynb](data_exploration.ipynb) contains an analysis of the dataset.
* [count_based.ipynb](count_based.ipynb) solves the problem and analyzes the results via a tf-idf encoding fed to a LinearSVC and a Random Forest.
* [zero_shot.ipynb](zero_shot.ipynb) explores zero-shot prompting approaches on GPT 3.5 turbo.
* [one_shot.ipynb](one_shot.ipynb) expands the zero-shot approaches with in-context learning.
* [few_shot.ipynb](few_shot.ipynb) explores few-shot methods via fine tuning and prompting of Masked Language Models. In particular, Umberto, a Roberta based Masked LM trained on an Italian corpus.

All the relevant code is contained in a well formed python package `vjp`:
* `vjp.data` provides all the tools to handle and preprocess data
* `vjp.dataset` is a data directory that ships the dataset used by these experiments
* `vjp.models` provides utilities for sklearn models.
* `vjp.folds` provides a balanced k-fold definition.
* `vjp.preprocess` provides the main preprocessing pipeline both as a function and as a CLI. The pipeline itself is described step by step in the notebooks.
* `vjp.text` provides text preprocessing pipelines.
* `vjp.lemmatization` provides italian lemmatization utilities.

## Preprocessing
The `vjp.preprocess` module contains an homonymous `preprocess(...)` function that can used to run all the data through the preprocessing pipeline. Even though it can be used this way, its structure is mainly thought for CLI usage. This script is mostly necessary for count based experiments (not shot based learning).

```
python -m vjp.preprocess -h
```

```
usage: preprocess.py [-h] [-l LEVEL] [-i [INPUT_FOLDERS ...]]
                     [-e [EDGE_RELATIONS ...]] [-c [CONNECTED_COMPONENT_TAGS ...]]
                     output_file

CLI script for data preprocessing. Dump to file preprocessed versions of the
dataset, choosing the level of preprocessing to meet.

positional arguments:
  output_file

options:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
  -i [INPUT_FOLDERS ...], --input-folder [INPUT_FOLDERS ...]
  -e [EDGE_RELATIONS ...], --edge-relation [EDGE_RELATIONS ...]
  -c [CONNECTED_COMPONENT_TAGS ...], --connected-component-tags [CONNECTED_COMPONENT_TAGS ...]
```
Data is dumped in parquet format. Output file is the only required argument. Input directories are expected to contain well formatted XML files (same structure of the ones found in `vjp.dataset`), but can be omitted. By default, all second instance samples found in `vjp.dataset` are used.

*(help descriptions for single options are missing)*

Example usage:
```
python -m vjp.preprocess preprocessed.parquet
```

Configuration as required by `baselines.ipynb`:
```
python -m vjp.preprocess -c req arg claim mot dec -l CONNECTED_COMPONENTS connected_components.parquet
```
