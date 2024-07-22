# CPUC ACC: Integrated Generation Capacity & GHG Avoided Cost Calculation

This repository contains the proposed methodology for calculating Generation Capacity & GHG Avoided Costs as part of 
the California Public Utilities Commission (CPUC) Integrated Distributed Energy Resources (IDER) 2024 
Avoided Cost Calculator (ACC). For more information, see 2024 ACC Electric Model and Documentation
[CPUC website](https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/energy-efficiency/idsm). 

## Quick Start

_Note: This quick start guide assumes you have already cloned or downloaded a copy of this repository._

### 0. Prepopulated Results
This repository comes pre-populated with results for 3 ACC cases, each of which can be found in the [results](./results/) folder.
* 2024ACC_TRC
* 2024ACC_SCT_Base
* 2024ACC_SCT_High

The following instructions are for users interested in running cases.

_Note: Parameters for each case are specified in the [acc.py file](./src/acc.py), inputs are in the [data folder](./data/processed/)._

### 1. Install Python via Anaconda 

We recommend using the [Anaconda](https://www.anaconda.com/download#downloads) Python distribution and package manager. 
During the installation process, we recommend selecting the **"Add Anaconda to my PATH environment variable"** option
so that we have easy access to the `conda` command from the command line.

> If you run into any `conda not recognized` or `command not found: conda` messages in the command line in the following steps,
> this means that you **did not** add Anaconda to your PATH. You can either rerun the installer (easiest) or manually
> add Anaconda to your PATH (see [these instructions](https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/) for some help).

### 2. Set Up a `conda` Environment 

We will use the `conda` command to create an isolated virtual environment for this example code to run within, without 
disturbing any other Python packages you may have already installed (see the [`conda` documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details on conda environments).

To create the `conda` environment, we will use the `environment.yml` file at the top level of the repository. 
Open a [Command Prompt window and navigate to this folder (the "acc-cpuc" folder)](https://www.wikihow.com/Open-a-Folder-in-Cmd). More experienced users may use a different terminal (e.g., PowerShell, Terminal on macOS).

Use the following command to create the `conda` environment:

```
conda env create -f environment.yml
```
<div style="page-break-after: always;"></div>

### 3. Activate the `conda` Environment

In the same Command Prompt window, use the following command to activate the `conda` environment:

```
conda activate e3-acc
```

#### macOS or Linux Users

If you are on a Mac or Linux computer, you will need to do one extra installation step. Run the following command
```commandline
conda install -c conda-forge coincbc
```

### 4. Run the ACC Calculation

In the same Command Prompt window, run the following command:
```commandline
e3-acc-cpuc
```

This will load the included input data from `./data/processed/{case_name}/` and report results to 
`./results/{case_name}/`.

Recall that parameters for each case are specified at the top of the [acc.py file](./src/acc.py), inputs are in the [data folder](./data/processed/). By default the code will generate results for the **2024ACC_TRC** case. 

These parameters and inputs may be updated and then the code may be re-run using the `e3-acc-cpuc` command. 