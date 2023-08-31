https://drivendata.github.io/cookiecutter-data-science/#opinions
```
├── .github                         <- Github actions
│   └── PULL_REQUEST_TEMPLATE.md    <- pull request template
│   └── ISSUE_TEMPLATE
│     ├── bug_report.md             <- bug report template
│     ├── doc_request.md            <- documentation request template
│     └── feature_request.md        <- feature request template
│   └── workflows
│     ├── bump-version.yml           <- automating incrementing version numbers
│     ├── pre-commit.yml             <- manage pre-commit hooks
│     ├── semantic-pull-request.yml  <- configure and enforce semantic guidelines for pull requests
│     └── testing.yml                <- ensure all pytests pass
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── case_settings  <- Define case input data.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── environment.yml   <- The requirements file for reproducing the environment
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`. (ex: cleaning client specific data)
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── results            <- Results folder
│  └──  reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── pyproject.toml     <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project. (ex: recap)
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── ui             <- Scripts to generate data inputs
│   │   └── scenario_tool.py
│   │
│   ├── main.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
├── tests              <- Pytests folder in same structure as src
│   ├── test_ui
│   │   └── test_scenario_tool.py
│   │
│   ├── test_main.py
│   │
│   └── test_visualization
│       └── test_visualize.py
```
