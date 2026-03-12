from enum import Enum
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.constants import *
from ..analysis import processing as utils


class SimType(Enum):
    """The supported simulation types when performing simulations.

    RANDOM:
        Takes a random action given a set of available actions.
    GREEDY:
        Predicts the growth at for the L1Os at a given state. Chooses the best
        performing action.
    ROLLOUT:
        Perform rollout simulations for the L1Os at a given state. Chooses the
        best performing action.
    ROLLOUT_PROB:
        Perform rollout simulations for the L1Os at a given state. Samples a random
        action using the softmax(rollout rewards) as the probability distribution.
    """

    RANDOM = 0
    GREEDY = 1
    ROLLOUT = 2
    ROLLOUT_PROB = 3


class SimDirection(Enum):
    """The supported simulation directions
    Down is working from the top down, Up is working from the bottom up.
    """

    DOWN = 0
    UP = 1
    BOTH = 2

    def action_value(self):
        """The value of the action to be taken.

        Returns
        -------
        int
            0 for DOWN and 1 for UP.
        """
        if self.value == 0:
            return 0
        return 1

    def target_value(self):
        """The value of the available target actions to be removed .

        Returns
        -------
        int
            1 for DOWN and 0 for UP.
        """
        if self.value == 0:
            return 1
        return 0


def rollout_trajectory(model, states, ingredients_pd, n_trajectories, threshold, sim_direction):
    """Performs a randomized rollout simulation. The random walk looks for all available
    actions at a current state, then chooses a random one. This process is repeated until no
    more actions can be taken, or if the actions results in no growth above the threshold,
    at which point we store the number of steps take (# of ingredients removed before no
    growth). The results are then averaged, to give a reward for each of the tested input
    states.

    Parameters
    ----------
    model : models.Model
        The model used when running the simulation.
    states : np.ndarray
        A 2D array of the states to run the rollouts on.
    ingredients_pd : pd.DataFrame
        The ingredients DataFrame containing the media information.
    n_trajectories : int
        The number of rollouts to perform, which the rewards are averaged over.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    sim_direction : SimDirection
        The directions in which the simulations run.

    Returns
    -------
    np.ndarray
        The averaged rewards of the trajectories performed on each of the input states.
    """
    trajectory_states = np.repeat(states, n_trajectories, axis=0)
    rewards = {i: [] for i in range(len(states))}

    # State boundaries keeps track of the start and end row indexes for each state in the
    # trajectory_states 2D-array
    states_boundaries = np.arange(0, n_trajectories * len(states) + 1, n_trajectories)

    reward_idx = 0
    step = 0
    # Random walk to remove 'n_trajectories' ingredients
    while trajectory_states.size > 0:
        available_actions = []
        available_values = []
        for row_idx, row in enumerate(trajectory_states):
            row_actions = []
            next_values = dict()
            for col_idx, val in enumerate(row):
                ingt_type = ingredients_pd.iloc[col_idx].TYPE
                if ingt_type == "binary":
                    if val == sim_direction.target_value():
                        row_actions.append(col_idx)
                        next_values[col_idx] = sim_direction.action_value()
                else:
                    min_value = ingredients_pd.iloc[col_idx].MIN_VALUE
                    max_value = ingredients_pd.iloc[col_idx].MAX_VALUE
                    n_states = int(ingredients_pd.iloc[col_idx].N_STATES)
                    if ingt_type == "quantitative":
                        n_states += 1
                    levels = np.linspace(min_value, max_value, n_states)
                    idx_level = np.where(np.isclose(levels, val))[0]
                    if not len(idx_level):
                        idx_level = [np.argmin(np.abs(levels - val))]
                    idx_level = idx_level[0]
                    if sim_direction == SimDirection.DOWN and idx_level > 0:
                        row_actions.append(col_idx)
                        next_values[col_idx] = levels[idx_level - 1]
                    elif sim_direction == SimDirection.UP and idx_level < n_states - 1:
                        row_actions.append(col_idx)
                        next_values[col_idx] = levels[idx_level + 1]
            available_actions.append(row_actions)
            available_values.append(next_values)

        if all(len(row_actions) == 0 for row_actions in available_actions):
            for k, v in rewards.items():
                remaining = n_trajectories - len(v)
                if remaining > 0:
                    rewards[k] = v + [step] * remaining
            break

        for row_idx, row_actions in enumerate(available_actions):
            if not row_actions:
                continue
            chosen_action = np.random.choice(row_actions)
            trajectory_states[row_idx, chosen_action] = available_values[row_idx][chosen_action]

        # Obtain predicted fitnesses for action taken for each tracjectory
        results, _ = model.evaluate(trajectory_states)

        # Obtain results below threshold
        no_grows = np.argwhere(results < threshold)[:, 0]

        # Add reward for finished trajectories to the proper state
        new_state_boundaries = states_boundaries.copy()
        for result_idx in no_grows:
            for i in range(len(states_boundaries) - 1):
                lb = states_boundaries[i]
                ub = states_boundaries[i + 1]
                if lb <= result_idx < ub:
                    rewards[i].append(step)
                    new_state_boundaries[i + 1 :] -= 1
                    break
        states_boundaries = new_state_boundaries

        # Remove finished trajectories
        trajectory_states = np.delete(trajectory_states, no_grows, axis=0)
        step += 1

    rewards = np.array(list(rewards.values()))
    return rewards.mean(axis=1)


def compute_adaptive_choice_const(state, direction, n_already_exists):
    """
    Compute the K for softmax, defaults to 100, where softmax() acts as ~max().
    Then as the number of duplicate media increases, K decreases. At K=1, it is
    the standard softmax(). At K=0, the choice becomes random. The percent of
    actions remaining is also calculated to contribute a depth decay effect as
    well.
    """
    # Hyperparams
    A = 100
    B = 30
    C = 3

    n_actions_remaining = (state == direction.target_value()).sum()
    percent_remaining = n_actions_remaining / len(state)
    k = -A * (1 - np.exp(-np.power(n_already_exists / B, C))) + A
    k = (k * 0.50) * (1 + percent_remaining)
    return k


@utils.decoratortimer(2)
def perform_simulations(
    model,
    state,
    ingredients_pd,
    n,
    threshold,
    sim_type,
    sim_direction,
    new_round_n,
    unique=False,
    batch_set=None,
    timeout=None,
    n_rollout_trajectories=1,
    go_beyond_frontier=True,
):
    """
    Performs simulations and generates a batch of experiments to determine the
    'growth frontier' of a bacteria. The simulations determine available actions
    and choose the next best action to take from the current state. Depending on the
    simulation type, this method differs. If there are no actions that result in a
    predicted growth, the simulation terminates and adds the desired state to the
    batch to test.

    Parameters
    ----------
    model : models.Model
        The model used when running the simulation.
    state : np.ndarray()
        The starting state of the media.
    ingredients_pd : pd.DataFrame
        The ingredients DataFrame containing the media information.
    n : Int
        The number of simulations to perform for this batch.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    sim_type : SimType
        The type of simulations to run.
        # Track all unique accepted frontier states
        accepted_frontier_states = []
        accepted_frontier_keys = set()
        frontier_state_counter = 0
    sim_direction : SimDirection
        The directions in which the simulations run.
    new_round_n : int
        The number of the new round.
    unique : bool, optional
        Take only unique states for the batch, by default False
    batch_set : set(tuple), optional
        The current states already in the batch, by default None
    timeout : int, optional
        The timeout length before forced temination of the simulations
        in seconds, by default None
    n_rollout_trajectories : int, optional
        The number of simulations to perform per state in the rollouts,
        by default 1
    go_beyond_frontier : bool, optional
        Add the state one step beyond the determined 'growth frontier',
        by default True

    Returns
    -------
    pd.DataFrame
        The batch of experiments to perform, where each row is a state
        to test and their associated metadata (simulation parameters, predicted
        growth, etc.)
    """
    if batch_set == None:
        batch_set = set()
    batch = []
    batch_frontier_types = []
    terminating_growths = []
    terminating_variances = []
    frontiers = []

    desc = f"Performing {sim_type.name} Sims ({sim_direction.name})"
    tq = tqdm(total=n, desc=desc)
    not_timed_out = True
    start_time = time.time()
    loops = 1
    n_found_but_exists = 0
    adaptive_choice_history = []
    
    starting_state_rounded = state.copy()
    starting_state_rounded = tuple(np.round(state, decimals=6))
    batch_set.add(starting_state_rounded)

    while len(batch) < n and not_timed_out:
        tq.desc = f"{desc} ({loops} loops)"

        frontier_for_start = 1

        current_state = state.copy()

        current_grow_pred = 0
        current_grow_var = 0
        while True:
            # print(f"Current state: {current_state}")

            choices = []
            next_values = dict()
            for i, val in enumerate(current_state):
                ingt_type = ingredients_pd.iloc[i].TYPE
                if ingt_type == "binary":
                    if val == sim_direction.target_value():
                        choices.append(i)
                        next_values[i] = sim_direction.action_value()
                else:
                    min_value = ingredients_pd.iloc[i].MIN_VALUE
                    max_value = ingredients_pd.iloc[i].MAX_VALUE
                    n_states = int(ingredients_pd.iloc[i].N_STATES)
                    if ingt_type == "quantitative":
                        n_states += 1
                    levels = np.linspace(min_value, max_value, n_states)
                    idx_level = np.where(np.isclose(levels, val))[0]
                    if not len(idx_level):
                        idx_level = [np.argmin(np.abs(levels - val))]
                    idx_level = idx_level[0]
                    if sim_direction == SimDirection.DOWN and idx_level > 0:
                        candidate = levels[idx_level - 1]
                        candidate_state = current_state.copy()
                        candidate_state[i] = candidate
                        candidate_key = tuple(np.round(candidate_state, decimals=6))
                        if candidate_key not in batch_set or not unique:
                            choices.append(i)
                            next_values[i] = candidate
                    elif sim_direction == SimDirection.UP and idx_level < n_states - 1:
                        candidate = levels[idx_level + 1]
                        candidate_state = current_state.copy()
                        candidate_state[i] = candidate
                        candidate_key = tuple(np.round(candidate_state, decimals=6))
                        if candidate_key not in batch_set or not unique:
                            choices.append(i)
                            next_values[i] = candidate

            choices = np.array(choices)

            # if no available actions, check if we can start from a frontier state (only if going beyond frontier is allowed)
            # terminate if not going beyond frontier or if exhausted all viable frontier states
            if choices.size == 0:
                #print(f"\n[DEBUG] No available actions from current state: {current_state}")
                if go_beyond_frontier:
                    # print(f"\n[DEBUG] Attempting to use frontier states as starting points...")
                    if len(frontiers) > 0 and frontier_for_start <= len(frontiers):
                        current_state = frontiers[frontier_for_start - 1]
                        frontier_for_start += 1
                        continue
                    elif len(frontiers) == 0 or frontier_for_start > len(frontiers):
                        # print(f"\n[DEBUG] No viable frontier states available. Terminating simulation.")
                        break
                else:
                    break
            
            candidate_states = np.tile(current_state, (choices.size, 1))
            if sim_type == SimType.RANDOM:
                action = np.random.choice(choices, 1, False)[0]  # Random one-step action
                candidate_states[0, action] = next_values[action]  # Take action
                candidate_states = candidate_states[0].reshape((1, -1))  # Reshape to 2D
                choices = [action]

            if sim_type == SimType.GREEDY:
                # Take all possible one-step actions
                for j, idx in enumerate(choices):
                    candidate_states[j, idx] = next_values[idx]

            elif sim_type == SimType.ROLLOUT or sim_type == SimType.ROLLOUT_PROB:
                # Generate all possible one-step actions
                for j, idx in enumerate(choices):
                    candidate_states[j, idx] = next_values[idx]

                # Perform rollouts
                rollout_results = rollout_trajectory(
                    model,
                    candidate_states,
                    ingredients_pd,
                    n_rollout_trajectories,
                    threshold,
                    sim_direction,
                )
                if sim_type == SimType.ROLLOUT_PROB:
                    # Pick an action idx from a distribution based on softmax of rollout results
                    k = compute_adaptive_choice_const(
                        current_state, sim_direction, n_found_but_exists
                    )
                    adaptive_choice_history.append((k, n_found_but_exists))

                    # Weighted softmax
                    p = utils.softmax(rollout_results, k=k)
                    action_idx = np.random.choice(choices.size, 1, p=p)[0]
                else:
                    # Pick highest predicted reward (mean # removed)
                    action_idx = np.argsort(rollout_results)[-1]

                action = choices[action_idx]
                candidate_states = candidate_states[action_idx].reshape((1, -1))
                choices = [action]

            # Get growth prediction of candidate states
            results, results_vars = model.evaluate(candidate_states)

            # Pick highest predicted growth as best action
            best_action_idx = np.argsort(results)[-1]
            best_action = choices[best_action_idx]

            # Keep track of prev state values
            old_state = current_state.copy()
            old_growth_result = current_grow_pred
            old_growth_var = current_grow_var


            # Set new state values
            new_state = candidate_states[best_action_idx]  # Take best action
            new_growth_result = float(results[best_action_idx])
            new_growth_var = float(results_vars[best_action_idx])
            
            # print(f"Best new state: {new_state}.  pred: {new_growth_result}, pred_var: {new_growth_var}")

            is_down = sim_direction == SimDirection.DOWN
            grows_present = (results >= threshold).sum() > 0

            if (is_down and grows_present) or (not is_down and not grows_present):
                # Keep going if grows are present and DOWN direction, or
                # Keep going if no grows are present and UP direction

                # Update state values
                current_state = new_state
                current_grow_pred = new_growth_result
                current_grow_var = new_growth_var

            elif (is_down and not grows_present) or (not is_down and grows_present):
                # Terminate if going DOWN and more grows present
                # Terminate if going UP terminate and there are grows present
                # NOTE: we need to add a third condition: terminate if we reach the end where ALL ingredients are opposite from their starting points
                # for that, we will need to define the endpoint states as an array
                if is_down:
                    f_state, b_state = old_state, new_state
                    f_grow_result, b_grow_result = old_growth_result, new_growth_result
                    f_grow_var, b_grow_var = old_growth_var, new_growth_var
                else:
                    f_state, b_state = new_state, old_state
                    f_grow_result, b_grow_result = new_growth_result, old_growth_result
                    f_grow_var, b_grow_var = new_growth_var, old_growth_var

                frontiers.append(f_state)

                if go_beyond_frontier:
                    # Add both the "frontier" and "beyond frontier" states
                    states = [f_state, b_state]
                    growth_preds = [f_grow_result, b_grow_result]
                    var_preds = [f_grow_var, b_grow_var]
                    frontier_types = ["FRONTIER", "BEYOND"]
                else:
                    states = [f_state]
                    growth_preds = [f_grow_result]
                    var_preds = [f_grow_var]
                    frontier_types = ["FRONTIER"]

                for st, gr, va, ft in zip(
                    states, growth_preds, var_preds, frontier_types
                ):
                    # Round state to avoid floating-point precision issues
                    st_rounded = np.round(st, decimals=6)
                    key = tuple(st_rounded)
                    if not unique or key not in batch_set:
                        # print(f"[DEBUG] Adding state: unique={unique}, key in batch_set={key in batch_set}")
                        batch.append(st_rounded)
                        terminating_growths.append(gr)
                        terminating_variances.append(va)
                        batch_frontier_types.append(ft)
                        batch_set.add(key)
                        tq.update()
                        st_print = np.round(st, decimals=2)  # round decimals for printing
                        print(f"\n\tADDED: {st_rounded} - {ft}")
                        if sim_type == SimType.ROLLOUT_PROB:
                            n_found_but_exists -= 1
                            n_found_but_exists = max(n_found_but_exists, 0)
                    else:
                        if sim_type == SimType.ROLLOUT_PROB:
                            n_found_but_exists += 1
                        print(f"\n\tEXISTS: {st_rounded} - {ft}")

                    if len(batch) >= n:
                        break
                break

        if timeout is not None:
            not_timed_out = (time.time() - start_time) <= timeout
        loops += 1

    duration = time.time() - start_time

    tq.close()
    if batch:
        batch = pd.DataFrame(np.vstack(batch))
        batch["type"] = sim_type.name
        batch["direction"] = sim_direction.name
        batch["frontier_type"] = batch_frontier_types
        batch["growth_pred"] = terminating_growths
        batch["var"] = terminating_variances
        batch["is_redo"] = False
        batch["round"] = new_round_n
    else:
        batch = pd.DataFrame()

    k_history = [a[0] for a in adaptive_choice_history]
    count_history = [a[1] for a in adaptive_choice_history]

    if len(k_history):
        k_avg = sum(k_history) / len(k_history)
        count_avg = sum(count_history) / len(count_history)
    else:
        k_avg = "n/a"
        count_avg = "n/a"

    metrics = {
        "k_history": k_history,
        "count_history": count_history,
        "k_avg": k_avg,
        "count_avg": count_avg,
        "total_loops_count": loops - 1,
        "time_to_finish_sec": round(duration, 2),
    }

    return batch, batch_set, metrics