import datetime
import pathlib

import toml
from loguru import logger

import src
from src.ui.make_dataset import make_dataset
from src.visualization.visualize import plot_data


def get_project_version() -> str:
    """
    Returns the version of the project as defined in pyproject.toml
    """
    with open(pathlib.Path(src.__file__) / ".." / ".." / "pyproject.toml", "r") as f:
        config = toml.load(f)

    version = config["project"]["version"]
    return version

def main():
    """Creates a simple example dataset and makes a line plot of it.

    Returns:
        The example data
    """
    version = get_project_version()
    logger.info(f"Code Version: {version}")
    logger.debug("making final data set from raw data")
    df = make_dataset()
    plot_data(df)
    return df


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    logger.add(pathlib.Path("..") / "results" / f"model_run_{start_time}.log", level="INFO")
    main()
