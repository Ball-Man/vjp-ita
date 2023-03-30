"""Model and evaluation utilities."""
from typing import Sequence, Iterable, Tuple

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

import vjp.data as data

CVIterator = Iterable[Tuple[Sequence[int], Sequence[int]]]
DEFAULT_TAGS = 'fact', 'req', 'claim', 'arg'
ALL_TAGS = 'fact', 'req', 'arg', 'claim', 'mot', 'dec'


def cross_validate(model: BaseEstimator, dataframe: pd.DataFrame,
                   cv: CVIterator,
                   tags: Sequence[str] = DEFAULT_TAGS,
                   scoring='f1_macro', n_jobs=-1) -> np.ndarray:
    """Cross validate given model on the specified data.

    Return a numpy array containing the scores (one per validation
    split).
    """
    X, y = data.count_based_X_y(dataframe, tags)
    return cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)
