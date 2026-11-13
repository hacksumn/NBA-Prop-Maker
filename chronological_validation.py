"""Expanding validation folds that never share a slate across train and test."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def date_grouped_splits(game_dates, n_splits=5):
    """Yield positional indices with max(train date) < min(validation date).

    Splitting player-game rows directly permits same-game outcomes to train a
    model scored on another player in that game. Split unique dates instead.
    Inputs must already be chronologically sorted; invalid dates fail loudly.
    """
    dates = pd.Series(pd.to_datetime(game_dates, errors="raise")).reset_index(drop=True).dt.normalize()
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("Cross-validation requires known, chronologically sorted game dates")
    unique = dates.drop_duplicates().to_numpy()
    for train_dates, test_dates in TimeSeriesSplit(n_splits=n_splits).split(unique):
        yield (np.flatnonzero(dates.isin(unique[train_dates]).to_numpy()),
               np.flatnonzero(dates.isin(unique[test_dates]).to_numpy()))
