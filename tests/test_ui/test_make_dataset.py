import pandas as pd

from src.ui.make_dataset import make_dataset


def test_make_dataset():
    df = make_dataset()
    pd.testing.assert_series_equal(
        df,
        pd.Series(
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="timestamp"),
            data=[300.0, 100.0],
            name="value",
        ),
    )
