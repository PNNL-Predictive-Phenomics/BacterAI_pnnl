
# BacterAI
This repository represents an extension of BacterAI produced by Pacific Northwest National Laboratory.
This repository allows for a CLI to control BacterAI and allows users to set their own lists of experimental conditions.
BacterAI was first developed by the Jensen Lab at the University of Michigan.
Those who want more information about BacterAI are encouraged to contact the original authors
at manager@jensenlab.net, read through the original repository (https://github.com/jensenlab/BacterAI)
as well as the canonical paper:

Adam C. Dama, Kevin S. Kim, Danielle M. Leyva, Annamarie P. Lunkes, Noah S. Schmid, Kenan Jijakli & Paul A. Jensen.
BacterAI maps microbial metabolism without prior knowledge. *Nat Microbiol* **8**, 1018–1025 (2023). 
https://doi.org/10.1038/s41564-023-01376-0

See example.md for short vignette of our workflow.

This document summarizes:
1. How to clone the repo and run the CLI locally
2. What the current CLI does
3. Command reference and usage examples

---

## 0) Clone and Run Locally

### Prerequisites

- `git`
- Python `3.12+`
- One environment manager (`conda` or `venv`)

### Option A: Conda setup (recommended for this repo)

```bash
git clone https://github.com/PNNL-Predictive-Phenomics/BacterAI_pnnl.git
cd BacterAI_pnnl

conda env create --name bacterai --file bacterai_env.yml
conda activate bacterai

pip install -e .
bacterai -h
```

### Option B: Python venv setup

```bash
git clone https://github.com/PNNL-Predictive-Phenomics/BacterAI_pnnl.git
cd BacterAI_pnnl

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

bacterai -h
```

### Verify CLI install

```bash
bacterai -h
bacterai experiment -h
```

If your shell does not find `bacterai`, use:

```bash
python -m bacterai.main -h
```

---

## 1) What This CLI Does

The `bacterai` CLI is a unified interface for preparing, running, and post-processing BacterAI experiments.

At a high level it helps you:
- Convert experiment spreadsheets into `config.json` files
- Convert ingredient spreadsheets into `ingredients.json`
- Interactively scaffold experiment folders and write config/ingredient files
- Execute experiment rounds (`Round1`, `Round2`, ...)
- Process raw plate reader data (Biotek/Tecan) into `mapped_data_*.csv` for model training

### CLI entrypoint
- Console script: `bacterai`
- Backed by: `src/bacterai/main.py` (`bacterai.main:main` in `pyproject.toml`)

---

## 3) Command Reference

Use help anytime:

```bash
bacterai -h
bacterai <command> -h
```

## 3.1 `experiment`
Convert experiment setup sheets (`.csv`/`.xlsx`) into one or more `config.json` files.

```bash
bacterai experiment <input> [--sheet <name_or_index>] [--outfile-name config.json] [--force-format vertical|horizontal] [--verbose]
```

- Prompts for handling existing directories
- Writes one config per experiment directory
- `--force-format` helps when auto-detection is ambiguous

Example:
```bash
bacterai experiment ./inputs/experiment_design.xlsx --sheet 0 --verbose
```

## 3.2 `setup`
Interactive end-to-end setup for experiment configs and ingredients.

```bash
bacterai setup <input> [--sheet <name_or_index>] [--outfile-name config.json] [--force-format vertical|horizontal] [--verbose]
```

- Uses same experiment parsing as `experiment`
- If `experiment_path` is missing, prompts to use current working directory (then asks for experiment folder name) or provide a full experiment folder path
- Creates the experiment folder path if it does not exist
- If target folder already exists and contains files, prompts to overwrite or provide a different full path
- Prompts whether ingredients are shared across experiments or individual
- Writes both config and ingredient files through setup utilities

Example:
```bash
bacterai setup ./inputs/experiment_design.xlsx --sheet Planning
```

## 3.3 `ingredients`
Convert ingredient sheets (`.csv`/`.xlsx`) to JSON payload.

```bash
bacterai ingredients <input> [--sheet <name_or_index>] [--output <path.json>] [--id-map <map.csv>] [--verbose]
```

- If `--output` is omitted, JSON is printed to stdout
- `--id-map` can map ingredient names to IDs

Example:
```bash
bacterai ingredients ./inputs/ingredients.xlsx --sheet 1 --output ./exp1/ingredients.json
```

## 3.4 `run`
Execute a BacterAI experiment round from an experiment directory.

```bash
bacterai run <experiment_path> [--plot-only] [--verbose]
```

- Verifies experiment path exists
- Auto-detects next round number using existing `Round*` folders
- Executes full pipeline unless `--plot-only` is set

Example:
```bash
bacterai run /path/to/experiment --verbose
```

## 3.5 `process_data`
Process raw plate reader files and write mapped data CSV for training.

```bash
bacterai process_data {biotek|tecan} [path] --round <N> --feature <name> [--date <id>] [--signal <int>] [--output <csv>] [--verbose]
```

- Supports experiment request layouts at either:
  - `<experiment_path>/experiment_request/`
  - `<experiment_path>/RoundN/experiment_request/`
- `--signal` is required for `biotek` and ignored for `tecan`
- Default output name (if not provided):
  - `RoundN/mapped_data_<date_or_today>_<reader>_<feature>_data.csv`

Example (Biotek):
```bash
bacterai process_data biotek /path/to/experiment --date 2026-03-12 --round 1 --signal 600 --feature delta_od --verbose
```

Example (Tecan):
```bash
bacterai process_data tecan /path/to/experiment --round 1 --feature delta_od
```

---

## 3.6 Practical workflow (typical)

1. `bacterai experiment ...` or `bacterai setup ...`
2. `bacterai run <experiment_path>` for Round 1
3. Collect plate reader outputs externally
4. `bacterai process_data ... --round 1 ...`
5. `bacterai run <experiment_path>` for Round 2
6. Repeat processing and run for subsequent rounds

### 3.7 Config settings
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
