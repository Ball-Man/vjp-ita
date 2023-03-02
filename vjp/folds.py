from typing import List, Tuple, Iterable
from itertools import chain
from operator import or_
from functools import reduce

import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, BINARY, INTEGER


def compute_folds(samples, num_folds=5, verbose=False,
                  max_seconds=10) -> Tuple[List[bool], ...]:
    """Retrieve balanced kfolds at document level.

    Expects as input a ``(N, 2)`` matrix where each sample represents a
    document: ``N`` is the number of documents. The two columns
    represent the number of positive and negative labels per each
    document. The function employs an integer programming model to
    provide ``num_folds`` partitions over the samples, as balanced as
    possible in terms of total number of positives and negatives.

    Output is a tuple in the form: ``(boolean_fold_0, ...)``.
    Each element of the tuple is a boolean list that can be used to
    select values of a dataframe like::
        documents_dataframe[boolean_fold_0]

    Length of the tuple is always ``num_folds``.
    """
    folds_range = range(num_folds)
    samples_range = range(len(samples))
    total_instances = sum(map(sum, samples))
    fold_ratio = total_instances / 10

    model = Model('balanced kfolds')
    model.verbose = verbose

    # folds_x[fold_index][sample_index]
    folds_x = [[model.add_var(var_type=BINARY) for _ in samples_range]
               for _ in folds_range]
    max_ = model.add_var(var_type=INTEGER)

    # Each sample is exclusive to one fold
    for sample_index in samples_range:
        model += xsum(folds_x[fold_index][sample_index]
                      for fold_index in folds_range) == 1

    # Compute counts for positive and negative labels,
    # minimize their distance from the target amounts
    positives = [xsum(folds_x[fold_index][sample_index]
                      * samples[sample_index][1]
                      for sample_index in samples_range)
                 for fold_index in folds_range]
    negatives = [xsum(folds_x[fold_index][sample_index]
                      * samples[sample_index][0]
                      for sample_index in samples_range)
                 for fold_index in folds_range]

    for value in chain(positives, negatives):
        model += value - fold_ratio <= max_
        model += - (value - fold_ratio) <= max_

    model.objective = minimize(max_)

    status = model.optimize(max_seconds=max_seconds)

    if verbose:
        output_folds = [[] for _ in folds_range]
        folds_values_positives = [0] * num_folds
        folds_values_negatives = [0] * num_folds
        for fold_index in folds_range:
            for sample_index in samples_range:
                if folds_x[fold_index][sample_index].x:
                    output_folds[fold_index].append(sample_index)
                    folds_values_positives[fold_index] += samples[
                        sample_index][1]
                    folds_values_negatives[fold_index] += samples[
                        sample_index][0]

        print('folds:', output_folds)
        print('folds positive counts:', folds_values_positives)
        print('folds negative counts:', folds_values_negatives)

        print('status', status)

    return tuple(
        list(map(lambda x: bool(round(x.x)), fold_vars))
        for fold_vars in folds_x
    )


def compute_decision_folds(dataframe: pd.DataFrame, num_folds=5, verbose=False,
                           max_seconds=10) -> Tuple[List[bool], ...]:
    """Use :func:`compute_folds` to compute a split at decision level.

    Meaning of optional arguments and output format are relatable to
    the ones of :func:`compute_folds`, but at decision level instead of
    document level. Note that the split still happens at document level,
    this function simply expands it at a finer granularity for practical
    reasons.
    """
    document_labels_df = (
        pd.get_dummies(dataframe[['document_index', 'label']],
                       columns=['label']).groupby('document_index').sum())
    document_folds = compute_folds(document_labels_df.values,
                                   num_folds=num_folds,
                                   verbose=verbose,
                                   max_seconds=max_seconds)

    return tuple(
        dataframe.document_index.isin(document_labels_df[document_fold].index)
        for document_fold in document_folds)


def split(dataframe: pd.DataFrame, num_folds=5, max_seconds=10
          ) -> Iterable[Tuple[List[int], List[int]]]:
    """Generator of train-test splits based on document level kfolds.

    Designed to be in a format similar to the one of ``scikit-learn``
    validators, and particularly in the right format to be used as
    ``cv`` parameter in ``sklearn.model_selection.GridSearchCV``.

    Yields a pair of list of indeces: ``(train_indeces, test_indeces)``
    """
    folds = compute_decision_folds(dataframe, num_folds=num_folds,
                                   max_seconds=max_seconds)
    for i, test_split in enumerate(folds):
        train_folds = folds[:i] + folds[i + 1:]
        train_split = reduce(or_, train_folds,
                             np.array(train_folds[0]))
        yield np.where(train_split)[0], np.where(test_split)[0]
