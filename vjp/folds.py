from itertools import chain
from typing import List, Tuple

from mip import Model, xsum, minimize, BINARY, INTEGER


def compute_folds(samples, num_folds=5,
                  verbose=False) -> Tuple[List[bool], ...]:
    """Retrieve balanced kfolds at document level.

    Expects as input a ``(N, 2)`` matrix where each sample represents a
    document: ``N`` is the number of documents. The two columns
    represent the number of positive and negative labels per each
    document. The function employs an integer programming model to
    provide ``num_folds`` partitions over the samples, as balanced as
    possible in terms of total number of positives and negatives.
    Sampling is not stratified.

    Output is a tuple in the form: ``(boolean_fold_0, ...)``.
    Each element of the tuple is a boolean list that can be used to
    select values of a dataframe like::
        documents_dataframe[boolean_fold_0]

    Length of the tuple is always ``num_folds``.
    """
    folds_range = range(num_folds)
    samples_range = range(len(samples))

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
    # minimize their maximum
    positives = [xsum(folds_x[fold_index][sample_index]
                      * samples[sample_index][1]
                      for sample_index in samples_range)
                 for fold_index in folds_range]
    negatives = [xsum(folds_x[fold_index][sample_index]
                      * samples[sample_index][0]
                      for sample_index in samples_range)
                 for fold_index in folds_range]

    for value in chain(positives, negatives):
        model += max_ >= value
    model.objective = minimize(max_)

    model.optimize()

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

    return tuple(
        list(map(lambda x: bool(round(x.x)), fold_vars))
        for fold_vars in folds_x
    )
