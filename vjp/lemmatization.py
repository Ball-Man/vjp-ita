from typing import Dict

DEFAULT_FILENAME = 'lemmatization-it.txt'


def load_lemmas(filename: str = DEFAULT_FILENAME) -> Dict[str, str]:
    """Load dictionary of lemmas."""
    with open(filename) as file:
        lines = file.read().splitlines()

    lemmas_dict = dict(reversed(line.split('\t')) for line in lines)
    keys_dict = dict(zip(lemmas_dict.values(), lemmas_dict.values()))
    lemmas_dict.update(keys_dict)

    return lemmas_dict
