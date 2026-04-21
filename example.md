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
      "ID": "P_putida_strain",
      "INGREDIENT": "P_putida_strain",
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

Because each BacterAI run contains a stochastic simulation element, please copy the following prepared folder for which data have already been generated for a run similar to the example above.

```bash
cp -R tests/vignette_experiment/ /path/to/vignette_experiment
```

The required file structure is:
```
my_experiment/
├── config.json
├── ingredients.json
└── Round1/
    ├── batch_dp.csv
    ├── batch_meta.csv
    ├── random_train_kickstart_others.csv
    ├── run_metrics.json
    ├── experiment_request/
    │   ├── data/
    │   │   └── testfid.xlsx*
    │   ├── plate_maps/
    │   │   ├── map.csv*
    │   │   └── plate_to_file_id.csv*
    │   └── worklists/
    │       └── testfid.csv*
    └── gpr_model/
        ├── gpr_likelihood.pth
        └── gpr_model.pth
```
\* These files are generated with a third party software. 

### Command (BioTek Example)

```bash
bacterai process_data biotek /path/to/vignette_experiment --round 1 --signal 600 --feature delta_od --verbose
```

### Expected Output

```
Processing Biotek data:
  Experiment path: /path/to/vignette_experiment
  Date: Not specified (using direct experiment_request path)
  Round: 1
  Feature: delta_od
  Signal: 600nm

Successfully processed 100 experiments (100 unique experiments)
Output written to: path/to/vignette_experiment/Round1/mapped_data_{date}_biotek_delta_od_data.csv

Data preview (first 5 rows):
   feature  bad plate_control plate_blank     parent_plate  experiment_number       strain environment
0    0.143    0          True       False  <plate-id-code>               9999  Test_strain         PH5
1    0.024    0         False        True  <plate-id-code>               9999  Test_strain         PH5
2   -0.003    1         False       False  <plate-id-code>                 76  Test_strain         PH5
3    0.013    0         False        True  <plate-id-code>               9999  Test_strain         PH5
4    0.003    1         False       False  <plate-id-code>                 90  Test_strain         PH5

Columns: feature, bad, plate_control, plate_blank, parent_plate, experiment_number, strain, environment
Shape (incl. controls): (133, 8)
```

### Generated mapped_data_*.csv (excerpt)

```csv
feature,bad,plate_control,plate_blank,parent_plate,experiment_number,strain,environment
0.143,0,True,False,<plate-id-code>,9999,Test_strain,PH5
0.024,0,False,True,<plate-id-code>,9999,Test_strain,PH5
-0.003,1,False,False,<plate-id-code>,76,Test_strain,PH5
0.013,0,False,True,<plate-id-code>,9999,Test_strain,PH5
0.003,1,False,False,<plate-id-code>,90,Test_strain,PH5
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
bacterai /path/to/vignette_experiment --verbose
```

### Expected Output

```
Starting BacterAI run for round 2
Experiment directory: path/to/experiment
Plot only: False
Redoing 10 experiments from previous round.
Media Results (Top 10):
 1. Fitness: 5.786, Depth: 1.16
	O2
 2. Fitness: 5.498, Depth: 0.99
	O2
 3. Fitness: 5.405, Depth: 0.97
	O2
 4. Fitness: 5.117, Depth: 1.47
	O2
 5. Fitness: 4.880, Depth: 1.28
	O2
 6. Fitness: 4.870, Depth: 0.36
	O2
 7. Fitness: 4.818, Depth: 0.29
	O2
 8. Fitness: 4.725, Depth: 0.87
	O2
 9. Fitness: 4.200, Depth: 1.44
	O2
10. Fitness: 4.200, Depth: 0.56
	O2
Total unique experiments: 100
Total redo experiments chosen: 10 (0 'bad' repeats)
Training GPR model...
Iter 1/100 | Train loss: 3.4354
Iter 2/100 | Train loss: 3.1748
...

Iter 99/100 | Train loss: 1.8823
Iter 100/100 | Train loss: 1.8823
Final MSE: 0.6771
Final R2: 0.7319
Creating simulation-based batch...
0 SimType.ROLLOUT 90 90 0
```
---

## Step 5: Repeat for Additional Rounds

Process data from Round 2 and continue iterating:

### Process Round 2 Data

```bash
bacterai process_data biotek /path/to/my_experiment --round 2 --signal 600 --feature delta_od
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