"""CLI script for data preprocessing.

Dump to file preprocessed versions of the dataset, choosing the
level of preprocessing to meet.
"""
import argparse
import enum
import pathlib
from typing import Optional, Sequence

import importlib_resources as resources
import networkx as nx
import pandas as pd

import vjp.data as data

DESCRIPTION = __doc__


class PreprocessingLevels(enum.Enum):
    """Levels of preprocessing for the data.

    Each level includes the previous ones.
    """
    CONNECTED_COMPONENTS = 0


class Namespace:
    """Custom namespace for argparse."""
    output_file: str
    level: PreprocessingLevels = PreprocessingLevels.CONNECTED_COMPONENTS
    input_folders: Sequence[resources.abc.Traversable] = ()

    def __init__(self):
        self.input_folders = []


def preprocess(namespace: Namespace) -> pd.DataFrame:
    """Apply preprocessing based on the given parameters."""
    level = namespace.level.value

    # Load data
    if namespace.input_folders:
        documents = sum((data.load_directory(directory)
                        for directory in namespace.input_folders), start=[])
    else:           # Default to existing provided data
        documents = data.load_second_instance()

    # Apply preprocessing pipeline based on given level
    documents = data.filter_other_outcomes(documents)
    triples_gen = (data.build_tag_triples(document) for document in documents)
    graphs = [nx.from_pandas_edgelist(triples, edge_attr='edge',
                                      create_using=nx.DiGraph())
              for triples in triples_gen]
    dataframe = data.dataframe_from_graphs(graphs, documents)

    # if level >= ...

    return dataframe


def dump_preprocess(namespace: Namespace):
    """Apply preprocessing and dump to file."""
    preprocess(namespace).to_parquet(namespace.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('output_file', type=str)
    parser.add_argument('-l', '--level', dest='level',
                        type=lambda s: PreprocessingLevels[s])
    parser.add_argument('-i', '--input-folder', nargs='*',
                        dest='input_folders', type=pathlib.Path,
                        action='append')

    namespace = parser.parse_args(namespace=Namespace())
    dump_preprocess(namespace)
