# Italian VAT Judgement Prediction
## Requirements
Python >= 3.8 is required. All dependencies can be installed with:
```
pip install -r requirements.txt
```

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
pip install -r requirements.txt -r dev_requirements.txt
python -m ipykernel install --user --name=VJP
```

On Windows (cmd):
```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt -r dev_requirements.txt
python -m ipykernel install --user --name=VJP
```

A user-wide jupyter installation is required. **On Windows**, `python` calls may be replaced by `py`. **On unix**, `python` may be replaced by `python3` and `pip` by `pip3`.

## Package structure
All the relevant code is contained in a well formed python package `vjp`:
* `vjp.data` provides all the tools to handle and preprocess data
* `vjp.dataset` is a data directory that ships the dataset used by these experiments
* `vjp.preprocess` provides the main preprocessing pipeline both as a function and as a CLI. The pipeline itself is described step by step in the notebooks.

## Preprocessing
The `vjp.preprocess` module contains an homonymous `preprocess(...)` function that can used to run all the data through the preprocessing pipeline. Even though it can be used this way, its structure is mainly thought for CLI usage.

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

*(help descriptions for single options are still missing)*

Example usage:
```
python -m vjp.preprocess preprocessed.parquet
```

Configuration as required by `baselines.ipynb`:
```
python -m vjp.preprocess -c req arg claim mot dec -l CONNECTED_COMPONENTS connected_components.parquet
```
