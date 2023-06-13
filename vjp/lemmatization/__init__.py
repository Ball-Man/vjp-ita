import os
from typing import Dict

import importlib_resources as resources

LEMMATIZATION_RESOURCES = 'vjp.lemmatization'
LEMMATIZATION_IT_FILE = (resources.files(LEMMATIZATION_RESOURCES)
                         / 'lemmatization-it.txt')


def load_lemmas(file: os.PathLike = LEMMATIZATION_IT_FILE
                ) -> Dict[str, str]:
    """Load dictionary of lemmas."""
    with open(file) as file:
        lines = file.read().splitlines()

    lemmas_dict = dict(reversed(line.split('\t')) for line in lines)
    keys_dict = dict(zip(lemmas_dict.values(), lemmas_dict.values()))
    lemmas_dict.update(keys_dict)

    return lemmas_dict
