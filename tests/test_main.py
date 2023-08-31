import pandas as pd

from src import main


def test_main():
    df = main.main()
    pd.testing.assert_series_equal(
        df,
        pd.Series(
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="timestamp"),
            data=[300.0, 100.0],
            name="value",
        ),
    )
