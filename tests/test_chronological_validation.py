import numpy as np
import pandas as pd
import pytest

from chronological_validation import date_grouped_splits


def test_no_same_slate_or_game_can_cross_validation_boundary():
    dates = pd.Series(np.repeat(pd.date_range("2026-01-01", periods=15), [3, 5, 9] * 5))
    folds = list(date_grouped_splits(dates, n_splits=4))
    assert len(folds) == 4
    for train, valid in folds:
        assert dates.iloc[train].max() < dates.iloc[valid].min()
        assert not set(train) & set(valid)
        for date in dates.iloc[valid].unique():
            assert set(np.flatnonzero(dates.eq(date))) <= set(valid)


@pytest.mark.parametrize("dates", [["2026-01-02", "2026-01-01"], ["2026-01-01", None]])
def test_invalid_or_unsorted_dates_fail_loudly(dates):
    with pytest.raises(ValueError, match="chronologically sorted"):
        list(date_grouped_splits(dates, 2))


def test_fewer_dates_than_folds_does_not_silently_fall_back_to_row_split():
    with pytest.raises(ValueError):
        list(date_grouped_splits(["2026-01-01"] * 100, 3))
