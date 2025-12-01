import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import models, export, setup, paths, transfer_learning, data_preparation, model_training, batch
from ..analysis import plotting, processing as utils
from ..scripts import size_n_to_m_conversion
from ..utils import constants
from ..sim.core import SimType, SimDirection, perform_simulations

# Construct combined ingredients list
TEMPEST_INGREDIENTS = constants.AA_NAMES_TEMPEST + constants.BASE_NAMES_TEMPEST

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # Use IntelMKL - does this work?


def process_results(
    folder,
    prev_folder,
    new_folder,
    new_round_n,
    ingredient_names,
    threshold,
    n_redos=0,
    redo_threshold=[0, 1],
    redo_prev_round=False,
    plot_only=False,
    plot_redos=True,
    transfer_padding_needed=False,
):
    """Process the results of the previous round, generate plots, and
    return batch information to be used when generating the new round's
    batch.
    """

    folder_contents = os.listdir(folder)
    n_ingredients = len(ingredient_names)

    # Get paths for the necessary files in the current folder (new round # - 1)
    mapped_path = None
    batch_path = None
    dataset_path = None
    for filename in folder_contents:
        filename_low = filename.lower()
        if "mapped_data" in filename_low and "redo" not in filename_low:
            mapped_path = os.path.join(folder, filename)
        elif "batch_meta" in filename_low and "results" not in filename_low:
            batch_path = os.path.join(folder, filename)
        elif "train_pred" in filename_low and "orig" not in filename_low:
            dataset_path = os.path.join(folder, filename)

    if mapped_path is None:
        raise FileNotFoundError("file 'mapped_data' not found!")
    if batch_path is None:
        raise FileNotFoundError("file 'batch_meta' not found!")

    new_dataset_path = os.path.join(new_folder, "train_pred.csv")

    # Merge results (mapped data) with predictions (batch data)
    data, _, _ = utils.process_mapped_data(mapped_path, ingredient_names)
    batch_df = utils.normalize_ingredient_names(pd.read_csv(batch_path, index_col=None))

    batch_df['experiment_number'] = batch_df.index + 1
    
    results = pd.merge(
        batch_df,
        data,
        how="left",
        left_on='experiment_number',
        right_on='experiment_number',
        sort=True,
    )

    if transfer_padding_needed:
        # Expand dimensions of non-AA data by the length of AA data and pad with ones
        batch_df = size_n_to_m_conversion.fill_new_ingredients(
            batch_df,
            original_size=len(constants.BASE_NAMES),
            fill_column_names=constants.AA_SHORT,
            fill_on_right=False,
        )
        results = size_n_to_m_conversion.fill_new_ingredients(
            results,
            original_size=len(constants.BASE_NAMES),
            fill_column_names=constants.AA_SHORT,
            fill_on_right=False,
        )
        n_ingredients = len(constants.AA_SHORT) + len(constants.BASE_NAMES)

    # Plot the rescreen experiments if available
    if "is_redo" in results.columns:
        redo_results = results[results["is_redo"] == True]
        results = results[results["is_redo"] != True]

        if prev_folder != None and plot_redos and Path(prev_folder).exists():
            prev_result_path = os.path.join(prev_folder, "results_all.csv")
            prev_results = utils.normalize_ingredient_names(
                pd.read_csv(prev_result_path, index_col=None)
            )
            plotting.plot_redos(folder, prev_results, redo_results, ingredient_names)

    # Process results
    results["depth"] = results.iloc[:, :n_ingredients].sum(axis=1)
    results = results.sort_values(["fitness", "depth"], ascending=False)
    if "frontier_type" not in results.columns:
        results["frontier_type"] = "FRONTIER"
        print("Added 'frontier_type' column")
    if not plot_only:
        results.to_csv(os.path.join(folder, "results_all.csv"), index=None)

    # Generate results figure for current round
    plotting.plot_results(folder, results, threshold)
    if plot_only:
        quit()

    # Keep experiments where all replicate wells are not marked "bad"
    results_bad = results.loc[results["bad"] == 1, :].drop(columns="bad")
    results = results.loc[results["bad"] == 0, :].drop(columns="bad")

    # Assemble new training data set, either from scratch or appending to previous Round's set
    cols = list(results.columns[:n_ingredients]) + ["fitness", "growth_pred", "var"]
    cols_new = list(range(n_ingredients)) + ["y_true", "y_pred", "y_pred_var"]
    if dataset_path == None:
        new_dataset = pd.DataFrame(
            results.loc[:, cols].to_numpy(),
            columns=cols_new,
        )
    else:
        dataset = utils.normalize_ingredient_names(
            pd.read_csv(dataset_path, index_col=None)
        )

        data_batch = results.loc[:, cols]
        dataset.columns = data_batch.columns = cols_new
        new_dataset = pd.concat([dataset, data_batch], ignore_index=True)

    # Used experiments are the new dataset (old dataset plus "good" experiments from current round)
    used_experiments = set(map(tuple, new_dataset.to_numpy()[:, :n_ingredients]))
    new_dataset.to_csv(new_dataset_path, index=None)
    X_train = new_dataset.iloc[:, :n_ingredients].to_numpy()
    y_train = new_dataset.loc[:, "y_true"].to_numpy()

    # Assemble redo experiments, starting with bad experiments
    results_grow_only = results[results["fitness"] >= threshold]

    # Obtain the experiments from the current round to rescreen in the new round
    if redo_prev_round:
        redo_experiments = batch_df.loc[batch_df["type"] != "REDO", :]
        if "is_redo" in redo_experiments.columns:
            redo_experiments = redo_experiments[redo_experiments["is_redo"] == False]
        redo_experiments["is_redo"] = True
        redo_experiments["round"] = new_round_n - 1
        redo_experiments.columns = list(range(n_ingredients)) + list(
            redo_experiments.columns[n_ingredients:]
        )
        print(f"Redoing {len(redo_experiments)} experiments from previous round.")
    elif n_redos > 0:
        redo_experiments = results[results["type"] != "REDO"]
        if "is_redo" in redo_experiments.columns:
            redo_experiments = redo_experiments[redo_experiments["is_redo"] == False]

        if isinstance(redo_threshold, list) or isinstance(redo_threshold, tuple):
            if len(redo_threshold) != 2:
                raise Exception("Length of redo_threshold must be 2.")

            thresholded_indexes = redo_experiments[
                (redo_experiments["fitness"] >= redo_threshold[0])
                & (redo_experiments["fitness"] < redo_threshold[1])
            ].index

            n_needed = max(n_redos - len(thresholded_indexes), 0)
            if n_needed > 0:
                # Need more to fill out redos
                print(f"Picking {n_needed} more random experiments to redo.")
                unused_indexes = redo_experiments.index.difference(thresholded_indexes)
                additional_indexes = np.random.choice(
                    unused_indexes,
                    size=min(n_needed, len(unused_indexes)),
                    replace=False,
                )
                thresholded_indexes = thresholded_indexes.union(additional_indexes)

            redo_experiments = redo_experiments.loc[thresholded_indexes, :]

        redo_experiments = redo_experiments.sample(
            min(n_redos, len(redo_experiments)), replace=False
        )

        redo_experiments = redo_experiments.loc[:, batch_df.columns]
        redo_experiments["is_redo"] = True
        redo_experiments["round"] = new_round_n - 1
        redo_experiments.columns = list(range(n_ingredients)) + list(
            redo_experiments.columns[n_ingredients:]
        )
        print(f"Redoing {len(redo_experiments)} experiments from previous round.")
    else:
        redo_experiments = None

    # Save and output successful results
    results_grow_only.to_csv(os.path.join(folder, "results_grow_only.csv"), index=False)

    # Print some metrics/results
    top_10 = results_grow_only.iloc[:10, :]
    print("Media Results (Top 10):")
    for idx, (_, row) in enumerate(top_10.iterrows()):
        print(f"{idx+1:2}. Fitness: {row['fitness']:.3f}, Depth: {row['depth']:2}")
        for l in row[:n_ingredients][row[:n_ingredients] == 1].index:
            print(f"\t{l}")

    print(f"Total unique experiments: {len(used_experiments)}")
    if redo_experiments:
        print(
            f"Total redo experiments chosen: {len(redo_experiments)} ({len(results_bad)} 'bad' repeats)"
        )

    return X_train, y_train, used_experiments, redo_experiments


def execute_experiment(experiment_path: str, plot_only: bool = False):
    """
    Execute a BacterAI experiment round.
    
    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory (containing config.json)
    plot_only : bool, optional
        If True, only generate plots without running experiments
    """
    # Setup experiment configuration and data
    settings, ingredients_pd, ingredients_list = setup.setup_experiment(experiment_path)
    n_ingredients = len(ingredients_list)
    
    # Create ingredients mapping for later use
    ingredients_map = dict(zip(range(len(ingredients_list)), ingredients_list)) 
    transfer_model = transfer_learning.load_pretrained_model(settings)
    
    date = datetime.datetime.now().isoformat().replace(":", ".")
    prev_round_folder, current_round_folder, new_round_folder = paths.setup_round_folders(
    settings.experiment_path, settings.round_number
) 
    if not new_round_folder.exists():
        new_round_folder.mkdir(parents=True)

    if settings.round_number > 1:
        # Calculate whether padding is needed for transfer learning data
        transfer_padding_needed = (settings.transfer_data_dir is not None and 
                         not settings.aas_only and 
                          settings.round_number == 2)
        # Continue the experiment (for all rounds except the first)
        redo_entire_round = False if settings.redo_size is not None else True
        X_train, y_train, used_experiments, redo_experiments = process_results(
            current_round_folder,
            prev_round_folder,
            new_round_folder,
            settings.round_number,
            ingredients_list,
            settings.grow_threshold,
            n_redos=settings.redo_size,
            redo_threshold=settings.redo_threshold,
            redo_prev_round=redo_entire_round,
            plot_only=plot_only,
            plot_redos=not settings.separate_redos,
            transfer_padding_needed=transfer_padding_needed,
        )
    elif settings.transfer_model_folder:
        # Skip any initial random training if using a pre-trained model
        X_train, y_train, used_experiments, redo_experiments = None, None, None, None
    else:
        # Generate random training data for cold start
        X_train, y_train, used_experiments, redo_experiments = data_preparation.generate_random_training_data(
            ingredients_pd, ingredients_map, settings, new_round_folder
        )
    
    # When doing transfer learning (data dir method), Round 1 is a special case
    # using the 'new' ingredients only so that we can kickstart that side of the NN,
    # to prevent those weights from collapsing.
    #
    # So, for the second round for 20+19 CDM, we have to combine round 1 data
    # (which has only non-AA ingredient inputs) with the transfer data of the AA-only
    # experiment (file located at settings.transfer_data_dir) in the following way:
    tl_X_train, tl_y_train = transfer_learning.handle_data_dir_transition(settings, new_round_folder, n_ingredients)
    if tl_X_train is not None:
        X_train, y_train = tl_X_train, tl_y_train
    
    # Train the model
    model = model_training.train_experiment_model(
        X_train, y_train, settings, new_round_folder, transfer_model
    )
    
    # Export redos into separate file if requested
    if settings.separate_redos:
        export.export_to_dp_batch(
            new_round_folder,
            redo_experiments,
            TEMPEST_INGREDIENTS,
            date,
            settings.nickname,
            is_redo=True,
        )
        redo_experiments = None
        
    
    # CREATE THE BATCHES
    if settings.round_number == 1 and settings.transfer_data_dir is None and settings.transfer_model_folder is None:
        # Round 1 without transfer learning - use experimental design
        batch_df, batch_used, all_metrics = batch.create_round1_experimental_design(
            settings, ingredients_pd, ingredients_list
        )
    else:
        # Other rounds or with transfer learning - use simulation-based approach
        batch_df, all_metrics = batch.create_simulation_based_batch(
            model, settings, ingredients_pd, ingredients_list, 
            used_experiments, redo_experiments, new_round_folder
        )
        model.close()
    
    # Output run metrics
    run_metrics_path = os.path.join(new_round_folder, "run_metrics.json")
    with open(run_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    export.export_to_dp_batch(new_round_folder, batch_df, ingredients_list, date, settings.nickname)