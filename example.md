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
├── experiment_request/
│   ├── data/
│   │   └── testfid.xlsx*
│   ├── plate_maps/
│   │   ├── map.csv*
│   │   └── plate_to_file_id.csv*
│   └── worklists/
│       └── testfid.csv*
└── Round1/
    ├── batch_dp.csv
    ├── batch_meta.csv
    ├── random_train_kickstart_others.csv
    ├── run_metrics.json
    └── gpr_model/          
        ├── gpr_likelihood.pth
        └── gpr_model.pth
```
* These files are generated with a third party software. 

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
Experiment directory: path/to/experiment
Plot only: False
Picking 10 more random experiments to redo.
Redoing 10 experiments from previous round.
Media Results (Top 10):
 1. Fitness: 1.031, Depth: 2.72
        o2
 2. Fitness: 1.024, Depth: 3.56
        o2
 3. Fitness: 1.024, Depth: 3.05
        o2
 4. Fitness: 1.023, Depth: 1.61
 5. Fitness: 1.023, Depth: 2.33
 6. Fitness: 1.019, Depth: 2.77
        o2
 7. Fitness: 1.017, Depth: 3.30
 8. Fitness: 1.017, Depth: 1.50
 9. Fitness: 1.014, Depth: 2.65
        o2
10. Fitness: 1.014, Depth: 2.51
Total unique experiments: 54
Total redo experiments chosen: 10 (0 'bad' repeats)
Training GPR model...
Iter 1/100 | Train loss: 1.2888
Iter 2/100 | Train loss: 1.2020
...

Iter 99/100 | Train loss: -2.5310
Iter 100/100 | Train loss: -2.5316
Final MSE: 0.0003
Final R2: 0.1484
Creating simulation-based batch...
0 SimType.ROLLOUT 90 90 0
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