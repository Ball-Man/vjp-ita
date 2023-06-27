"""Text manipulation."""
import re
from typing import Callable, Dict
from functools import partial

import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords as rem_stopwords

import vjp.lemmatization as lemmatization

lemmatization_dict: Dict[str, str] = lemmatization.load_lemmas()

# Can be used in pipelines to substitute whitespaces (and more)
multiple_newlines_re = re.compile(r'\n+')
multiple_spaces_re = re.compile(r' +')
all_whites_re = re.compile(r'\s+')
weird_whites_re = re.compile(r'[\r\t\f\v]+')
trailing_line_spaces_re = re.compile(r'\n\s+|\n\s+')


def load_stopwords():
    """Download stopwords from NLTK."""
    nltk.download('stopwords')


def text_pipeline(*callables: Callable[[str], str]) -> Callable[[str], str]:
    """Build a text pipeline from a sequence of callables/pipelines."""
    def pipeline(text: str) -> str:
        for callable_ in callables:
            text = callable_(text)
        return text

    return pipeline


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text.

    Stopwords must be first loaded via :func:`load_stopwords`.
    """
    return rem_stopwords(text, stopwords=stopwords.words('italian'))


def remove_punctuation(text: str, replace_with=' ') -> str:
    """Remove punctuation from text.

    Everything that is not alphanumerical will be removed (replaced
    with ``replace_with``). By default, replace with a space in order
    to separate articles from nouns (e.g. ``l'ultimo -> l ultimo``).
    This helps since no sofisticated lemmatization for such cases is
    employed.
    """
    return ''.join(c if c.isalnum() else replace_with for c in text)


def lemmatize(text: str,
              lemmatization_dict: Dict[str, str] = lemmatization_dict,
              drop_missing=True) -> str:
    """Lemmatize text using the given lemmatization dict.

    Default lemmatization dict is provided by the
    :mod:`vjp.lemmatization` module and contains a simple set of italian
    lemmas.

    By default, missing words (words that are not in the lemmas
    dictionary) are completely removed.
    """
    if drop_missing:
        map_fun = lemmatization_dict.get                        # NOQA
    else:
        map_fun = lambda w: lemmatization_dict.get(w, w)        # NOQA

    lemmas = map(map_fun, text.split())
    lemmas = filter(bool, lemmas)

    return ' '.join(lemmas)


count_pipeline_head = text_pipeline(
    str.lower,
    remove_punctuation,
    remove_stopwords
)
"""Basic text preprocessing pipeline for count based encodings.

Punctuation and stopwords are removed. No lemmatization or stemming.
Designed to be used as head in more complex pipelines.
"""

count_drop_pipeline = text_pipeline(
    count_pipeline_head,
    lemmatize
)
"""Text preprocessing pipeline for count based encodings.

Designed for bag of words, tf-idf, etc.
All symbols and stopwords are removed. Words are lemmatized.
Unknown lemmas are removed (dropped).

Uses :attr:`count_pipeline_head` as head.
"""


count_keep_pipeline = text_pipeline(
    count_pipeline_head,
    partial(lemmatize, drop_missing=False)
)
"""Text preprocessing pipeline for count based encodings.

Designed for bag of words, tf-idf, etc.
All symbols and stopwords are removed. Words are lemmatized.
Unknown lemmas are kept unchanged.

Uses :attr:`count_pipeline_head` as head.
"""

# TODO: add a pipeline function to remove special characters only?
#       keep punctuation but remove stupid ascii characters.
shot_normalize_whites_pipeline = text_pipeline(
    partial(weird_whites_re.sub, ''),
    partial(multiple_newlines_re.sub, '\n'),
    partial(multiple_spaces_re.sub, ' '),
    partial(trailing_line_spaces_re.sub, ''),
    str.strip
)
"""Text preprocessing pipeline for x-shot learning (LLMs).

In order::

- Weird whitespaces (e.g. artifacts from XML structure) are removed.
- Multiple whitespaces are collapsed into one.
- Multiple newlines are collapsed into one.
- Trailing spaces are stripped.

Since some large language models will treat different spaces
differently, the most informative white characters (namely newline,
spaces) are kept, but normalized. Any other weird character is removed.

It could be interesting experimenting with different and/or more
simplistic setups (e.g. remove all white characters and replace them
with whitespaces only).
"""
