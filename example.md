# BacterAI Example Workflow

This example demonstrates the complete workflow for setting up and running a BacterAI experiment using sample data files.

## Prerequisites

Ensure BacterAI is installed and activated:

```bash
# Using conda
conda activate bacterai

# Or using venv
source env/bin/activate

# Verify installation
bacterai -h
```

---

## Step 1: Set Up Experiment from CSV Files

Convert experiment configuration and ingredients from CSV files into JSON format and create the experiment directory structure.

### Command

```bash
bacterai setup tests/test_files/inputs/demo_experiment.csv
```

### Expected Interaction

The CLI will prompt you interactively for your experiment path name and ingredients file (use `ingredients.csv` in the test files folder).
If these paths are configured in your `config.json` bacterai will set them up automatically for you.

### Expected Output Files

After running this command, your experiment directory will contain:

```
my_experiment/
├── config.json          # Experiment configuration
└── ingredients.json     # Ingredients list
```

### Important Considerations

If you would like to setup your `config.json` and `ingredients.json` separately, use the experiment and ingredients commands respectively.

### Generated config.json

```json
{
  "experiment_path": "/path/to/my_experiment",
  "ingredients_file": "ingredients.json",
  "grow_threshold": 0.25,
  "nickname": "Test",
  "batch_size": 90,
  "timeout_min": 60,
  "model_type": 0,
  "direction": 0,
  "beyond_frontier": true,
  "use_unique": false,
  "n_rollouts": 2,
  "n_bags": 25,
  "transfer_model_folder": null,
  "transfer_data_dir": null,
  "redo_size": 10,
  "redo_threshold": [0.0, 0.4],
  "aas_only": false,
  "separate_redos": false,
  "simulation_types": [2],
  "random_walk_increment": 10
}
```

### Generated ingredients.json (excerpt)

```json
{
  "ingredients": [
    {
      "ID": "o2",
      "INGREDIENT": "o2",
      "TYPE": "binary",
      "N_STATES": 2,
      "NOMINAL_VALUE": 0.0,
      "MIN_VALUE": 0.0,
      "MAX_VALUE": 1.0,
      "UNIT": "X",
      "C_ATOMS": 0,
      "N_ATOMS": 0,
      "NA_ATOMS": 0,
      "CL_ATOMS": 0
    },
    {
      "ID": "d_glucose",
      "INGREDIENT": "d_glucose",
      "TYPE": "quantitative",
      "NOMINAL_VALUE": 4.0,
      "MIN_VALUE": 0.0,
      "MAX_VALUE": 4.0,
      "UNIT": "mmol/L",
      "C_ATOMS": 6,
      "N_ATOMS": 0,
      "NA_ATOMS": 0,
      "CL_ATOMS": 0
    },
    {
      "ID": "sodium_citrate",
      "INGREDIENT": "sodium_citrate",
      "TYPE": "quantitative",
      "NOMINAL_VALUE": 4.0,
      "MIN_VALUE": 0.0,
      "MAX_VALUE": 4.0,
      "UNIT": "mmol/L",
      "C_ATOMS": 6,
      "N_ATOMS": 0,
      "NA_ATOMS": 3,
      "CL_ATOMS": 0
    },
    {
      "ID": "ammonium_chloride",
      "INGREDIENT": "ammonium_chloride",
      "TYPE": "quantitative",
      "NOMINAL_VALUE": 1.0,
      "MIN_VALUE": 0.0,
      "MAX_VALUE": 1.0,
      "UNIT": "mmol/L",
      "C_ATOMS": 0,
      "N_ATOMS": 1,
      "NA_ATOMS": 0,
      "CL_ATOMS": 1
    },
    {
      "ID": "pH",
      "INGREDIENT": "pH",
      "TYPE": "semi-quantitative",
      "N_STATES": 3,
      "NOMINAL_VALUE": 7.0,
      "MIN_VALUE": 5.0,
      "MAX_VALUE": 9.0,
      "UNIT": "pH",
      "C_ATOMS": 0,
      "N_ATOMS": 0,
      "NA_ATOMS": 0,
      "CL_ATOMS": 0
    },
    {
      "ID": "P_putida_AVS10",
      "INGREDIENT": "P_putida_AVS10",
      "TYPE": "strain",
      "NOMINAL_VALUE": 60.0,
      "MIN_VALUE": 60.0,
      "MAX_VALUE": 60.0,
      "UNIT": "mmol/L",
      "C_ATOMS": 0,
      "N_ATOMS": 0,
      "NA_ATOMS": 0,
      "CL_ATOMS": 0
    }
  ]
}
```
---

## Step 2: Run Experiment Round 1

Execute the first round of BacterAI simulations to generate initial experiment suggestions.

### Command

```bash
bacterai run /path/to/my_experiment --verbose
```

### Expected Output

```
Starting BacterAI run for round 1
Experiment directory: my/experiment/directory
Plot only: False
Generating random training data for cold start...
Generated 1000 random training examples
Training GPR model...
Iter 1/100 | Train loss: 1.0223
Iter 2/100 | Train loss: 0.9811
---
Iter 99/100 | Train loss: 0.5343
Iter 100/100 | Train loss: 0.5352
Final MSE: 0.1674
Creating Round 1 experimental design...
Using space-filling design for 90 experiments
Run completed successfully for round 1
```

### Generated Directory Structure

After Round 1 completes:

```
my_experiment/
├── config.json
├── ingredients.json
└── Round1/
    ├── batch_dp.csv
    ├── batch_meta.csv
    ├── random_train_kickstart_others.csv
    ├── run_metrics.json
    └── gpr_model/          
        ├── gpr_likelihood.pth
        └── gpr_model.pth
```

---

## Step 3: Process Plate Reader Data

After collecting plate reader data externally, process it to create the training data CSV.

The required file structure is:
```
my_experiment/
├── config.json
├── ingredients.json
├── experiment_request/ # will need to be in r1 as sub folder
    ├── data/
    ├── plate_maps/
    └── worklists/
└── Round1/
    ├── batch_dp.csv
    ├── batch_meta.csv
    ├── random_train_kickstart_others.csv
    ├── run_metrics.json
    └── gpr_model/          
        ├── gpr_likelihood.pth
        └── gpr_model.pth
```

### Command (BioTek Example)

```bash
bacterai process_data biotek /path/to/my_experiment --round 1 --signal 600 --feature delta_od --verbose
```

### Expected Output

```
Processing Biotek data:
  Experiment path: /path/to/my_experiment
  Date: Not specified (using direct experiment_request path)
  Round: 1
  Feature: delta_od
  Signal: 600nm

Successfully processed 60 experiments (60 unique experiments)
Output written to: my/path/experiment/Round1/mapped_data_{date}_biotek_delta_od_data.csv

Data preview (first 5 rows):
    feature  bad plate_control plate_blank       parent_plate  experiment_number       strain environment
0  0.227000    0          True       False  Test_Plate_Round1               9999  Test_strain         PH5
1  0.218560    0         False        True  Test_Plate_Round1               9999  Test_strain         PH5
2  0.223625    0         False       False  Test_Plate_Round1                 76  Test_strain         PH5
3  0.216506    0         False        True  Test_Plate_Round1               9999  Test_strain         PH5
4  0.212810    0         False       False  Test_Plate_Round1                 80  Test_strain         PH7

Columns: feature, bad, plate_control, plate_blank, parent_plate, experiment_number, strain, environment
Shape (incl. controls): (84, 8)
```

### Generated mapped_data_*.csv (excerpt)

```csv
feature,bad,plate_control,plate_blank,parent_plate,experiment_number,strain,environment
0.227,0,True,False,Test_Plate_Round1,9999,Test_strain,PH5
0.21856,0,False,True,Test_Plate_Round1,9999,Test_strain,PH5
0.223625,0,False,False,Test_Plate_Round1,76,Test_strain,PH5
...
```

### Alternative: Process Tecan Data

```bash
bacterai process_data tecan /path/to/my_experiment --round 1 --feature delta_od --verbose
```

Expected output is similar but from Tecan file format.

---

## Step 4: Run Experiment Round 2

After processing the plate reader data from Round 1, run Round 2 to explore the parameter space using the trained model.

### Command

```bash
bacterai run /path/to/my_experiment --verbose
```

### Expected Output

```
Starting BacterAI run for round 2
Experiment directory: /path/to/my_experiment
Reading configuration from: /path/to/my_experiment/config.json
Loading ingredients from: /path/to/my_experiment/ingredients.json

Found existing Round1 data
Loading training data from: /path/to/my_experiment/Round1/mapped_data_2026-04-16_biotek_delta_od_data.csv
Loaded 90 experiments for training

Training GPR model...
  - Model R²: 0.87
  - Cross-validation R²: 0.82

Running Round 2...
  - Batch size: 90 experiments
  - Simulation type: ROLLOUT
  - Beyond frontier: TRUE
  - N rollouts: 2

Simulating 90 new experiments...
  - High-fitness predictions found: 32
  - Conditions beyond frontier: 15

Round 2 batch generated
Output directory created: /path/to/my_experiment/Round2/
```

### Generated Files for Round 2

```
my_experiment/
├── config.json
├── ingredients.json
├── Round1/
│   ├── batch_config_round1.json
│   ├── mapped_data_2026-04-16_biotek_delta_od_data.csv
│   └── Round1_plots/
└── Round2/
    ├── batch_config_round2.json
    ├── experiment_designs.csv
    ├── experiment_request/
    │   ├── plate_maps/
    │   └── worklists/
    └── Round2_plots/
        ├── model_predictions_vs_observed.png
        ├── acquisition_function.png
        └── next_experiments.png
```

---

## Step 5: Repeat for Additional Rounds

Process data from Round 2 and continue iterating:

### Process Round 2 Data

```bash
bacterai process_data biotek /path/to/my_experiment --round 2 --signal 600 --feature delta_od --date 2026-04-18
```

### Run Round 3

```bash
bacterai run /path/to/my_experiment
```

Continue this cycle as many times as needed to converge on optimal conditions.

---

## Common Commands Reference

### Get Help on Any Command

```bash
bacterai -h
bacterai experiment -h
bacterai setup -h
bacterai ingredients -h
bacterai run -h
bacterai process_data -h
```

### Convert Only (Without Setup)

If you only want to generate config.json files without the interactive setup:

```bash
bacterai experiment tests/test_files/inputs/demo_experiment.csv
```

Expected output:
```
Wrote config file(s):
  - my_experiment/config.json
```

### Generate Ingredients JSON Only

```bash
bacterai ingredients tests/test_files/inputs/ingredients.csv --output my_experiment/ingredients.json
```

### Verbose Output for Debugging

Add `--verbose` flag to any command to see detailed processing information:

```bash
bacterai setup tests/test_files/inputs/demo_experiment.csv --verbose
```

---

## Troubleshooting

### No Rounds Found After Run

If `bacterai run` doesn't create Round1, check:
- Config file exists at `<experiment_path>/config.json`
- Ingredients file path is correct in config
- No permission issues in experiment directory

### Process Data Fails

For `bacterai process_data`:
- Ensure `experiment_request/` directory exists
- Check plate_maps and worklists are present
- For BioTek: ensure `--signal` is provided (e.g., 600)
- For Tecan: signal is ignored (optional flag)

### Import Errors

If commands fail with import errors:
```bash
python -m bacterai.main -h
```

This bypasses shell script and directly invokes the module.

---