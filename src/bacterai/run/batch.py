"""Batch creation utilities for BacterAI."""

import numpy as np
import pandas as pd
from pyDOE3 import pbdesign
from scipy.stats import qmc
from ..sim.core import SimType, SimDirection, perform_simulations


def make_batch(
    model,
    media,
    ingredients_pd,
    new_round_n,
    batch_size,
    sim_types,
    rollout_trajectories,
    threshold,
    timeout=60 * 15,
    unique=True,
    direction=SimDirection.DOWN,
    go_beyond_frontier=True,
    used_experiments=None,
    redo_experiments=None,
):
    """Make a new BacterAI batch; the main function that calls the simulation loops."""
    n_types = len(sim_types)
    n_exps = batch_size // n_types
    batch_set = used_experiments
    sub_batches = []
    all_metrics = {}
    
    for idx, sim_type in enumerate(sim_types):
        if idx == n_types - 1:
            n_exps = batch_size - sum([len(x) for x in sub_batches])
        print(idx, sim_type, batch_size, n_exps, sum([len(x) for x in sub_batches]))
        
        batch, batch_set, metrics = perform_simulations(
            model,
            media,
            ingredients_pd,
            n_exps,
            threshold,
            sim_type,
            direction,
            new_round_n,
            unique=unique,
            timeout=timeout,
            batch_set=batch_set,
            n_rollout_trajectories=rollout_trajectories,
            go_beyond_frontier=go_beyond_frontier,
        )
        sub_batches.append(batch)
        all_metrics[sim_type.name] = metrics
    
    batch = pd.concat([redo_experiments] + sub_batches, ignore_index=True)
    return batch, batch_set, all_metrics


def create_round1_experimental_design(settings, ingredients_pd, ingredients_list):
    """
    Create Round 1 experimental design using Plackett-Burman or space-filling design.
    
    Parameters
    ----------
    settings : Settings
        Experiment settings
    ingredients_pd : pd.DataFrame
        Ingredients dataframe
    ingredients_list : list
        List of ingredient names
        
    Returns
    -------
    tuple
        (batch, batch_used, all_metrics)
    """
    print("Creating Round 1 experimental design...")
    
    n_ingredients = len(ingredients_list)
    
    # Set up ingredients for experimental design
    ingredients_pd.loc[ingredients_pd.TYPE == "quantitative", "N_STATES"] = 3
    # total_runs = np.sum(ingredients_pd.astype({"N_STATES": "int64"})["N_STATES"])

    if settings.batch_size < 40:
        # Use Plackett-Burman design
        print(f"Using Plackett-Burman design for {settings.batch_size} experiments")
        batch = create_plackett_burman_design(settings, ingredients_pd, ingredients_list, n_ingredients)
    else:
        # Use space-filling random design
        print(f"Using space-filling design for {settings.batch_size} experiments")
        batch = create_space_filling_design(settings, ingredients_pd, ingredients_list, n_ingredients)
    
    # Add metadata columns
    batch["type"] = "n/a"
    batch["direction"] = 2
    batch["frontier_type"] = 'FRONTIER'
    batch["growth_pred"] = 1
    batch["var"] = 0
    batch["is_redo"] = False
    batch["round"] = 1
    
    batch_used = set()
    
    # Create dummy metrics for Round 1
    metrics = {
        "k_history": "n/a",
        "count_history": "n/a", 
        "k_avg": "n/a",
        "count_avg": "n/a",
        "total_loops_count": "n/a",
        "time_to_finish_sec": "n/a",
    }
    
    all_metrics = {}
    direction = SimDirection.BOTH
    all_metrics[direction.name] = metrics
    
    return batch, batch_used, all_metrics


def create_plackett_burman_design(settings, ingredients_pd, ingredients_list, n_ingredients):
    """Create Plackett-Burman experimental design."""
    batch = pbdesign(n_ingredients)
    batch = pd.DataFrame(batch, columns=ingredients_pd["INGREDIENT"])
    batch = (batch + 1) / 2

    # Modify 0/1 values to concentrations
    for ingt in ingredients_list:
        ingt_type = ingredients_pd[ingredients_pd.INGREDIENT == ingt].TYPE.iloc[0]
        
        if ingt_type == "quantitative":
            # Quantitative ingredients get either 0 or the nominal value
            nominal_val = ingredients_pd[ingredients_pd.INGREDIENT == ingt].NOMINAL_VALUE.iloc[0]
            if nominal_val == 0:
                nominal_val = ingredients_pd[ingredients_pd.INGREDIENT == ingt].MAX_VALUE.iloc[0]
            batch[ingt] = batch[ingt] * nominal_val
            
        elif ingt_type == "semi-quantitative":
            # Semi-quantitative ingredients get either the min or max value
            min_val = ingredients_pd[ingredients_pd.INGREDIENT == ingt].MIN_VALUE.iloc[0]
            max_val = ingredients_pd[ingredients_pd.INGREDIENT == ingt].MAX_VALUE.iloc[0]
            batch.loc[batch[ingt] == 0, ingt] = min_val
            batch.loc[batch[ingt] == 1, ingt] = max_val
    
    return batch


def create_space_filling_design(settings, ingredients_pd, ingredients_list, n_ingredients):
    """Create space-filling experimental design using Sobol sequences."""
    # Create 2^m experiments, likely creating more than necessary at first
    ingt_sampler = qmc.Sobol(d=n_ingredients)
    m_to_use = np.ceil(np.log2(settings.batch_size))
    batch = ingt_sampler.random_base2(m=np.int64(m_to_use))
    
    # Rescale values to upper and lower bounds
    l_bounds = ingredients_pd.MIN_VALUE.to_numpy(copy=True)
    u_bounds = ingredients_pd.MAX_VALUE.to_numpy(copy=True)
    mask = (ingredients_pd["TYPE"].astype(str) == "binary").to_numpy()
    u_bounds[mask] = 1

    batch = qmc.scale(batch, l_bounds, u_bounds)
    batch = pd.DataFrame(batch, columns=ingredients_pd["INGREDIENT"])
    
    # Convert binary values to 0/1 and match semi-quantitative values to closest match
    for ingt in ingredients_list:
        ingt_type = ingredients_pd[ingredients_pd.INGREDIENT == ingt].TYPE.iloc[0]
        
        if ingt_type == "binary":
            batch[ingt] = batch[ingt].round()
            
        elif ingt_type == "semi-quantitative":
            # Assumes a min, nominal, and max semi-quant value scheme
            sq_vals = np.array([
                ingredients_pd[ingredients_pd.INGREDIENT == ingt].MIN_VALUE.iloc[0],
                ingredients_pd[ingredients_pd.INGREDIENT == ingt].NOMINAL_VALUE.iloc[0],
                ingredients_pd[ingredients_pd.INGREDIENT == ingt].MAX_VALUE.iloc[0]
            ])
            col = batch[ingt].values
            col_match = np.array([np.argmin(np.abs(val - sq_vals)) for val in col])
            batch[ingt] = sq_vals[col_match]
    
    # Limit to number of experiments in plate
    batch = batch.loc[0:(settings.batch_size - 1)]
    
    return batch


def create_simulation_based_batch(model, settings, ingredients_pd, ingredients_list, 
                                used_experiments, redo_experiments, new_round_folder):
    """
    Create simulation-based batch for rounds > 1 or with transfer learning.
    
    Returns
    -------
    tuple
        (batch, all_metrics)
    """
    print("Creating simulation-based batch...")
    
    # Prepare ingredients data for simulation
    ingredients_pd['N_STATES'] = ingredients_pd['N_STATES'].fillna(settings.random_walk_increment)
    ingredients_pd['N_STATES'] = pd.to_numeric(ingredients_pd['N_STATES'])
    
    # Reverse min and max values if min == nominal (stress condition)
    ingredients_pd['IS_STRESS'] = ingredients_pd['MIN_VALUE'] == ingredients_pd['NOMINAL_VALUE']
    ingredients_pd.loc[ingredients_pd['IS_STRESS'], ['MIN_VALUE', 'MAX_VALUE']] = \
        ingredients_pd.loc[ingredients_pd['IS_STRESS'], ['MAX_VALUE', 'MIN_VALUE']].values
    
    # Build starting media based on ingredient type
    starting_media_down, starting_media_up = build_starting_media(ingredients_pd, ingredients_list)

    # Determine simulation parameters
    if settings.direction == SimDirection.DOWN:
        starting_media = starting_media_down
        direction = SimDirection.DOWN
        batch_size = settings.batch_size
    elif settings.direction == SimDirection.UP:
        starting_media = starting_media_up
        direction = SimDirection.UP
        batch_size = settings.batch_size
    elif settings.direction == SimDirection.BOTH:
        starting_media = starting_media_down
        direction = SimDirection.DOWN
        batch_size = settings.batch_size // 2
    
    all_metrics = {}
    
    # Create first batch
    batch, batch_used, metrics = make_batch(
        model,
        starting_media,
        ingredients_pd,
        new_round_n=settings.round_number,
        batch_size=batch_size,
        sim_types=settings.simulation_types,
        rollout_trajectories=settings.n_rollouts,
        threshold=settings.grow_threshold,
        timeout=60 * settings.timeout_min,
        unique=settings.use_unique,
        direction=direction,
        go_beyond_frontier=settings.beyond_frontier,
        used_experiments=used_experiments,
        redo_experiments=redo_experiments,
    )
    all_metrics[direction.name] = metrics
    
    # Create second batch for UP direction if needed
    if settings.direction == SimDirection.BOTH:
        direction = SimDirection.UP
        starting_media = starting_media_up
        batch2, _, metrics = make_batch(
            model,
            starting_media,
            ingredients_pd,
            new_round_n=settings.round_number,
            batch_size=batch_size,
            sim_types=settings.simulation_types,
            rollout_trajectories=settings.n_rollouts,
            threshold=settings.grow_threshold,
            timeout=60 * settings.timeout_min,
            unique=settings.use_unique,
            direction=direction,
            go_beyond_frontier=settings.beyond_frontier,
            used_experiments=batch_used,
        )
        batch = pd.concat((batch, batch2), ignore_index=True)
        all_metrics[direction.name] = metrics
    
    return batch, all_metrics


def build_starting_media(ingredients_pd, ingredients_list):
    """Build starting media arrays for DOWN and UP directions."""
    starting_media_down = []
    starting_media_up = []
    
    for ingt in ingredients_list:
        ingt_type = ingredients_pd[ingredients_pd.INGREDIENT == ingt].TYPE.iloc[0]
        
        if ingt_type == "binary":
            starting_media_down.append(1.0)
            starting_media_up.append(0.0)
        elif ingt_type in ["semi-quantitative", "quantitative"]:
            min_value = ingredients_pd[ingredients_pd.INGREDIENT == ingt].MIN_VALUE.iloc[0]
            max_value = ingredients_pd[ingredients_pd.INGREDIENT == ingt].MAX_VALUE.iloc[0]
            n_states = int(ingredients_pd[ingredients_pd.INGREDIENT == ingt].N_STATES.iloc[0])
            levels = np.linspace(min_value, max_value, n_states)
            starting_media_down.append(levels[-1])
            starting_media_up.append(levels[0])
        else:
            raise ValueError(f"Unknown ingredient type: {ingt_type}")
    
    return np.array(starting_media_down), np.array(starting_media_up)