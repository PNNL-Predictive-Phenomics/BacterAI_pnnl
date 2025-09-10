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
Place the necessary configuration files and optional ingredient files within the experiment folder. Customize the `config.json` file for the specific experiment you'd like to run. Ensure the experimental file paths in the `config.json` file are absolute paths. If PNNL code is used, an ingredients list JSON file may be required.

Structure:
```plaintext
EXPT_FOLDER
  |-- config.json
  |-- ingredients.json
  |-- ingredients.xlsx (optional)
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
Note that the `tee` command in a Unix-like system allows you to view standard output while saving it to a log file.

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
  |-- ingredients.xlsx (optional)
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

If training GPR models:
```plaintext
EXPT_FOLDER
  |-- config.json
  |-- ingredients.json
  |-- ingredients.xlsx (optional)
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
After generating instrument data via PlatePlan, place those data into the `Round1` folder.
Look for files containing the term `"mapped_data"`, as these are what BacterAI will process.
For the default Biotek plate reader settings, the system focuses on `OD-600` signal, extracting the final OD values.

---

### Runs 2+

For subsequent rounds (`Round 2` and beyond), import raw data, extract relevant features, and generate a mapped data frame that BacterAI requires.
The example below uses the function `biotek_feature_extract.py` to process the data from a Biotek plate reader and find the difference between the
original and final optical density (OD).
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

### License
The software was originally released from the Jensen lab under the MIT license and is available for non-commercial use. 
Anyone interested in commercial use of BacterAI is encouraged to contact the authors at manager@jensenlab.net
