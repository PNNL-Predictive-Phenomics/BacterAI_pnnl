<p align="center">
    <img src="https://user-images.githubusercontent.com/4694385/225218387-ff83524a-bdcc-4751-b2c7-0fab567f50eb.png" alt="bacterai_logo_2000" style="width: 60%;">
</p>

## BacterAI
BacterAI was first developed by the Jensen Lab at the University of Michigan.
This repository represents an extension of BacterAI produced by Pacific Northwest National Laboratory.

### Installation
To install requirements using Conda, follow these steps to create a new environment named `bacterai` using the `bacterai_env.yml` file:

```bash
conda env create --name bacterai --file bacterai_env.yml
```

To activate the environment:
```bash
conda activate bacterai
```

To deactivate the environment:
```bash
conda deactivate
```

### Experimental Set-up

---

### Setting up an experiment

#### Create an Experiment Folder Structure
Place the necessary configuration files and optional ingredient files within the experiment folder. 
Customize the `config.json` file for the specific experiment you'd like to run (see Config section below). 
Ensure the experimental file paths in the `config.json` file are absolute paths.

Structure:
```plaintext
EXPT_FOLDER
  |-- config.json
  |-- ingredients.json
  |-- transfer_model_folder (optional)
  |   `-- transfer models
  |-- transfer_data_folder (optional)
  |   `-- transfer learning data
```

> **Note:** You can create your JSON files using R commands.
```R
library(jsonlite)
ingredients_sheet <- "path/to/experiment/ingredients.xlsx"

readxl::read_excel(ingredients_sheet, sheet = 1, na = 'NA') |>
  list(ingredients = _) |>
  toJSON(dataframe = 'rows', pretty = T, auto_unbox = T, null = 'null', na = 'null') |>
  write(file = file.path(getwd(), 'ingredients.json'))
```

---

### First Run

To start the first round (`Round 1`), use the following steps. 
> **Note:** The `tee` command in a Unix-like system allows you to view standard output while saving it to a log file.

*All commands assume that the conda environment is already loaded.*

#### Ensure `run.py` has execute permissions:
```bash
chmod u+x run.py
```

#### Commands:
```bash
cd /path/to/bacterai/code

EXPT=/path/to/experiment
N=1
python run.py $EXPT/config.json --round $N | tee $EXPT/stdout_R${N}.log
```

#### Output of Round 1

If training bagged neural nets:
```plaintext
EXPT_FOLDER
  |-- config.json
  |-- ingredients.json
  |-- Round1
  |   |-- random_train_kickstart*.csv (for round 1)
  |   |-- run_metrics.json
  |   |-- batch_meta_[datetime].csv
  |   |-- batch_dp_custom_ingredientsR1_[datetime].csv
  |   |-- nn_models
  |   |   |-- bag_model_1.pkl
  |   |   |-- bag_model_2.pkl
  |   |   |-- ...
  |   |   `-- bag_model_n.pkl
```

If training Gaussian process regression (GPR) models:
```plaintext
EXPT_FOLDER
  |-- config.json
  |-- ingredients.json
  |-- Round1
  |   |-- random_train_kickstart*.csv (for round 1)
  |   |-- run_metrics.json
  |   |-- batch_meta_[datetime].csv
  |   |-- batch_dp_custom_ingredientsR1_[datetime].csv
  |   |-- gpr_model
  |   |   |-- gpr_model.pth
  |   |   `-- gpr_likelihood.pth
```

#### Final Output after Full Round 1
The model-specific files will be utilized in the next round of BacterAI.
The most important file is the `batch_meta` CSV file.
This file contains the set of experiments that BacterAI has requested to be run.
These experiments will need to be converted into a more detailed protocol, either for an autonomous laboratory system, or for a human researcher.
The `Plateplan` application may be utilized for this but are not part of the BacterAI documentation.
`Plateplan` instructions are often embedded in larger custom Python script which can handle the idiosyncrasies of each experiment
(*e.g.,* which "ingredients" are liquid-dispensable reagents vs. which are conditions which might be controlled by other means).

---

### Runs 2+

For subsequent rounds (`Round 2` and beyond), you need to import raw data, extract relevant features, and generate a mapped data table that BacterAI requires.
The resulting CSV should have the term `mapped_data` somewhere in the filename, and it should go into the same round where the original request came from.

The example below uses the function `biotek_feature_extract.py` to process the data from a Biotek plate reader and find the difference between the
original and final optical density (OD).
In the example, this function placed the `mapped_data` file back into the `Round1` folder. 
The wavelength (`--signal` or `-s`) for almost any optical density measurement will be 600 nm.

#### Commands:
```bash
cd /path/to/bacterai/code

EXPT=/path/to/experiment
N=2
PREV_N=$(expr $N - 1)
run_date="YYYY-MM-DD"

python biotek_feature_extract.py $EXPT -d run_date -r $PREV_N -s 600 -f "delta_od"

python run.py $EXPT/config.json --round $N | tee $EXPT/stdout_R${N}.log
```

---

### Config settings
The config file represents the main way to control the experiment.

```json
{
  "experiment_path": "/path/to/experiment",
  "ingredients_file": "ingredients.json",
  "grow_threshold": 0.25,
  "nickname": "PpKT2440",
  "batch_size": 100,
  "timeout_min": 240,
  "model_type": 0,
  "direction": 0,
  "beyond_frontier": true,
  "use_unique": false,
  "n_rollouts": 10,
  "n_bags": 25,
  "transfer_model_folder": null,
  "transfer_data_dir": null,
  "redo_size": 0,
  "redo_threshold": [0, 1],
  "aas_only": false,
  "separate_redos": false,
  "simulation_types": [2],
  "random_walk_increment": 10
}
```

#### `grow_threshold`
- Used in `main()` function in `run.py`
- Default "growth/no-growth" threshold for model training. This is described as the grow threshold but it is more accurate to describe it as the fitness threshold because this represents the values of the fitness (or "y") variable after being normalized by each plate to a generic fitness score.
- 0.25 was the original default value; 0.1 - 0.12 seem good for M9 media

#### `nickname`
- What to call your experiment

#### `ingredients_file`
- The name of the file containing the list of ingredients to consider for BacterAI experiments
- BacterAI expects a JSON file
- Default: `"ingredients.json"`

#### `batch_size`
- Number of experiments to run during a single BacterAI "round"
- Determined by your plate size, liquid handling capacity, and sample measurement process

#### `timeout_min`
- Max time (minutes) to run a "round" in BacterAI before timing out
- Default: `4 * 60` (i.e., 4 hours)
- **Note:** There is some indication that this may actually be in seconds

#### `model_type`
- Used in `main()` function in `run.py`
- Selects the predictive model
  - GPR is a form of Bayesian optimization and will work best for continuous variables and where the full kernel of a density distribution can be explored across the possible range of values of a variable
  - Bagged neural nets may work best when confronted with binary or ordinal factors or a mix of categorical and continuous variables
- `0 = GPR (Gaussian process regression)`, `1 = NEURAL_NET`
- Default: `0`

#### `direction`
- Used in `main()` and `make_batch()` functions in `run.py`, but mainly in `perform_simulations()` function in `sim.py`
- Indicates whether to put all ingredients in and remove (`DOWN`), leave all out and add-in (`UP`), or split runs to do both
- `0 = DOWN`, `1 = UP`, `2 = BOTH`
- Default: `0`

#### `simulation_types`
- Used in `main()` and `make_batch()` functions in `run.py`; mainly in `perform_simulations()` function in `sim.py`
- Can have length > 1 if multiple types are specified. If so, new batch simulations are divided across types:
  - `random` performs random take-one-out actions
  - `greedy` takes all leave-one-out actions
  - `rollout` takes all leave-one-out actions, then calls `rollout_trajectory()` function to perform random walks of available actions and ranks those with the most viable "growth pathways"
- `0 = RANDOM`, `1 = GREEDY`, `2 = ROLLOUT`, `3 = ROLLOUT_PROB`
- Default: `2`

#### `beyond_frontier`
- Used in `perform_simulations()` function in `sim.py`
- When simulating new runs, determines whether to go beyond the growth frontier
  - If `True`, new untested conditions supporting predicted growth are included
- Default: `True`

#### `use_unique`
- Used in `perform_simulations()` function in `sim.py`
- Whether to take only unique states for the batch
- Default: `True`

#### `n_rollouts`
- Used in `perform_simulations()` in `sim.py` and within the `rollout_trajectory()` function (called by `perform_simulations()` when `simulation_types = ROLLOUT*`)
- Number of random-walk rollouts to assess with leave-one-out approaches
- Default: `1` (must be > 0 if using any rollout type of simulation)

#### `n_bags`
- Used in `train_bagged()` function in `net.py`
- Number of bagged models to run (relevant for NeuralNet model training)
  - Number of bagged models should match the number of transfer learning models, if applicable
- Default: `25`

#### `transfer_model_folder`
- Location of the transfer learning model
- Default: `None`

#### `transfer_data_dir`
- Location of the transfer learning data
- Default: `None`

#### `redo_size`
- Pre-specifies the number of experiments to redo from the previous round
- Used in `process_results()` function (`run.py`); in code, assigned to `N_REDOS`
  - Should likely be `0`, as setting to `None` may redo the entire previous round
- Use in conjunction with `redo_threshold`
- Default: `0`

#### `redo_threshold`
- Threshold range used to determine which experiments to redo
- Used in `process_results()` (`run.py`)
  - An intentional QA strategy or a mechanism to refine data for borderline growth conditions
- Default: `[0, 1]`

#### `aas_only`
- Determines whether ingredients consist only of amino acids
- Default: `false`

#### `separate_redos`
- Whether or not to separate redos into a dedicated batch of experiments (e.g., a separate plate)
- Default: `false`

#### `random_walk_increment`
- Number of increments to break a continuous variable into for simulation runs
- Used in `make_batch()`, `perform_simulations()`, and the `rollout_trajectory()` function (if using a rollout simulation type)
  - Applied only to continuous variables requiring exploration (`N_STATES = NA`)
  - Larger values increase exploration speed but decrease precision
  - Potential enhancement: make dynamic so increments decrease as rounds increase
- Default: `10`

---

### License
The software was originally released from the Jensen lab under the MIT license and is available for non-commercial use. 
Anyone interested in commercial use of BacterAI is encouraged to contact the authors at manager@jensenlab.net
