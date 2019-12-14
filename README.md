# Bankruptcy Prediction

For Data Preparation, Under src/data folder run python data_generation.py.
For Prediction, Under src folder run python bankruptcy_prediction.py.

Running Prediction will run Prediction using 6 different models and displays variour performance metrics graphs and Table.

## Setup for Coding Parts

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. Extract the zip file and run `conda env create -f environment.yml` from inside the extracted directory.
  - This creates a Conda environment called `cs229_project`
3. Run `source activate cs229_project`
  - This activates the `cs229` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229_project/bin/python`
  - Select `OK` then `Apply`