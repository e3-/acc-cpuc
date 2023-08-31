import pandas as pd
from loguru import logger

def make_dataset() -> pd.Series:
    """Creates a simple example dataset.

    Returns:
        the example dataset
    """
    logger.info("making example data set")
    data = pd.Series(
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="timestamp"),
        data=[300.0, 100.0],
        name="value",
    )

    return data
