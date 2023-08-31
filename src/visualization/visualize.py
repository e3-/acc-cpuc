from typing import Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger


def plot_data(data: Union[pd.Series, pd.DataFrame]) -> go.Figure:
    """Creates a line plot of a series or dataframe.

    Args:
        data: the data to plot

    Returns:
        line plot of the data
    """
    logger.info("plotting data")
    fig = px.line(data)

    return fig
