"""
Bayesian optimisation-style experiment recommendation for omicsTL models.

Works with any trained TransferVAE, TransferMLP, or TransferForest and any
dataset — the interface is fully domain-agnostic.  The main entry point is
``recommend_next_batch``.

Uncertainty sources
-------------------
RF  : variance across individual transfer trees (pred_0 … pred_N columns).
DL  : Monte Carlo Dropout — the model is temporarily set to training mode so
      dropout remains active; n_mc_samples forward passes are averaged.

Acquisition functions
---------------------
EI  : Expected Improvement  — recommended default.
UCB : Upper Confidence Bound — robust when uncertainty is poorly calibrated.
POI : Probability of Improvement — included for completeness; EI is strictly
      better in almost all practical cases.

Batch diversity
---------------
Two-stage selection: (1) shortlist the top ``shortlist_pct`` fraction of
candidates by acquisition score (default top 10 %); (2) from that shortlist,
greedily pick ``batch_size`` points that maximise the minimum pairwise
Euclidean distance (greedy max-min, a 2-approximation of the NP-hard optimal).
This is strictly stronger than greedy exclusion (min_dist_pct) because it
actively optimises spread within the high-EI region rather than just preventing
the worst clustering.

Candidate grid resolution
--------------------------
By default all features are sampled as continuous floats.  Pass ``step_sizes``
to snap individual features to experimental grid levels — integer step gives
integer values; 0.5 gives one decimal place; 0.25 gives two; and so on.
Mixed precision is supported: each feature declares its own granularity.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.stats import qmc as _qmc

logger = logging.getLogger(__name__)

AcquisitionName = Literal["EI", "UCB", "POI"]


# ── Sobol / candidate helpers ─────────────────────────────────────────────────

def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 that is >= n (required by Sobol sampler)."""
    return 1 << max(n - 1, 0).bit_length()


def _maximin_select(
    norm_candidates: np.ndarray,
    acq_scores: np.ndarray,
    batch_size: int,
    shortlist_pct: float = 0.10,
) -> np.ndarray:
    """Top-EI shortlist + greedy max-min diversity batch selection.

    Parameters
    ----------
    norm_candidates : feature matrix normalised to [0, 1], shape (n, d).
    acq_scores      : acquisition score per candidate, shape (n,).
    batch_size      : number of points to select.
    shortlist_pct   : fraction of candidates to keep before applying max-min.
                      Default 0.10 (top 10 %).  Floor is always >= batch_size.

    Returns
    -------
    np.ndarray of shape (batch_size,) — indices into norm_candidates / acq_scores,
    ordered by selection (first = highest EI seed).

    Algorithm
    ---------
    1. Shortlist k = max(batch_size, ceil(n * shortlist_pct)) candidates by EI.
    2. Precompute all k×k pairwise Euclidean distances within the shortlist.
    3. Seed with the highest-EI shortlisted point.
    4. Greedily add the point whose minimum distance to the already-selected
       set is largest (max of min-distances).  Update min-distances after each
       pick in O(k) time.
    Total complexity: O(n log k) shortlisting + O(k² d) distance precomputation
    + O(batch_size × k) greedy loop — fast for typical k ≈ 800, d ≤ 20.
    """
    n = len(acq_scores)
    k = min(n, max(batch_size, int(np.ceil(n * shortlist_pct))))

    # ── 1. Shortlist top-k by acquisition score ────────────────────────────
    shortlist_idx = (
        np.arange(n) if k >= n
        else np.argpartition(acq_scores, -k)[-k:]
    )
    shortlist  = norm_candidates[shortlist_idx]   # (k, d)
    sh_scores  = acq_scores[shortlist_idx]        # (k,)

    logger.debug(
        "max-min shortlist: top %.0f%% → k=%d from n=%d candidates.",
        shortlist_pct * 100, k, n,
    )

    # ── 2. Precompute all pairwise distances in shortlist ──────────────────
    diff     = shortlist[:, None, :] - shortlist[None, :, :]  # (k, k, d)
    pairwise = np.sqrt((diff ** 2).sum(axis=2))               # (k, k)

    # ── 3. Greedy max-min selection ────────────────────────────────────────
    first    = int(np.argmax(sh_scores))    # seed: highest-EI point
    selected = [first]
    min_dist = pairwise[:, first].copy()   # distance to the only selected point
    min_dist[first] = -np.inf             # mark as selected

    for _ in range(batch_size - 1):
        nxt = int(np.argmax(min_dist))     # point farthest from selected set
        selected.append(nxt)
        min_dist = np.minimum(min_dist, pairwise[:, nxt])
        min_dist[nxt] = -np.inf

    return shortlist_idx[np.array(selected)]


# ── Discrete candidate helpers ─────────────────────────────────────────────────

def _infer_decimals(step: float) -> int:
    """Return the number of decimal places implied by step.

    Examples: 1 → 0,  25 → 0,  0.5 → 1,  0.25 → 2,  0.1 → 1,  0.05 → 2.
    Uses the string representation of step to avoid floating-point ambiguity.
    """
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _make_allowed_values(lo: float, hi: float, step: float) -> np.ndarray:
    """Enumerate all discrete allowed values in [lo, hi] at step increments.

    Parameters
    ----------
    lo, hi : range bounds (inclusive).
    step   : grid spacing — e.g. 1 for integers, 0.5 for half-steps.

    Returns a 1-D array with values rounded to the precision of step.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step!r}")
    n_steps = max(1, round((hi - lo) / step))
    vals = lo + np.arange(n_steps + 1) * step
    vals = vals[vals <= hi + step * 1e-6]   # trim float overshoot
    return np.round(vals, _infer_decimals(step))


# ── Acquisition functions ──────────────────────────────────────────────────────

def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Expected Improvement over the current best observation f_best.

    EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - f_best - xi) / sigma.

    The first term is exploitation (reward for being above f_best); the second
    is exploration (reward proportional to uncertainty).  No hyperparameter
    needs tuning — the balance self-regulates as more data are collected.

    Parameters
    ----------
    mu, sigma : predicted mean and standard deviation for each candidate.
    f_best    : best response value observed so far.
    xi        : small positive jitter to encourage exploration (default 0).
    """
    sigma = np.clip(sigma, 1e-9, None)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return np.clip(ei, 0.0, None)


def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Upper Confidence Bound: mu + kappa * sigma.

    kappa controls the explore/exploit trade-off.  Higher kappa = more
    exploration.  More robust than EI when uncertainty estimates are poorly
    calibrated, but requires a sensible kappa.

    kappa schedule
    --------------
    If kappa is None (the default), the theoretically-motivated schedule
    ``kappa = sqrt(2 * ln(n_obs))`` is used, where n_obs is the number of
    existing observations.  This grows with the dataset so exploration is
    maintained as more rounds are collected.  Passing a fixed float overrides
    this.

    Parameters
    ----------
    n_obs : number of existing observations (required when kappa is None).
    """
    if kappa is None:
        if n_obs is None or n_obs < 1:
            raise ValueError(
                "UCB requires either a kappa value or n_obs (number of existing "
                "observations) to auto-compute kappa = sqrt(2 * ln(n_obs))."
            )
        kappa = float(np.sqrt(2.0 * np.log(max(n_obs, 2))))
        logger.debug("UCB: auto-computed kappa = %.4f (n_obs=%d)", kappa, n_obs)
    return mu + kappa * np.clip(sigma, 0.0, None)


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement over f_best + xi.

    POI(x) = Phi((mu - f_best - xi) / sigma).

    xi is a required improvement buffer.  With xi=0 POI degenerates: any point
    infinitesimally above f_best gets probability ~1 regardless of magnitude,
    so it clusters around the current best and stops exploring.  The default
    xi=0.01 forces it to look for points that are meaningfully better.

    EI is generally preferred because it accounts for the magnitude of
    improvement automatically, but POI can be useful when you want a simple
    probability threshold.
    """
    if xi <= 0.0:
        logger.warning(
            "POI called with xi=%.4f.  With xi<=0, POI is degenerate — it "
            "treats a tiny improvement identically to a large one and clusters "
            "near the current best.  Consider xi=0.01 or use EI instead.", xi
        )
    sigma = np.clip(sigma, 1e-9, None)
    Z = (mu - f_best - xi) / sigma
    return norm.cdf(Z)


def compute_acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    method: AcquisitionName = "EI",
    kappa: float | None = None,
    xi: float | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Dispatch to the requested acquisition function.

    Parameters
    ----------
    method : 'EI' (default), 'UCB', or 'POI'.
    kappa  : UCB exploration weight.  If None, auto-computed as
             sqrt(2*ln(n_obs)).  Ignored for EI and POI.
    xi     : Improvement buffer for EI and POI.  Defaults: EI→0.0, POI→0.01.
             Ignored for UCB.
    n_obs  : Number of existing observations.  Required by UCB when kappa=None.

    Notes
    -----
    * EI   — xi=0 is the standard default and needs no tuning.
    * UCB  — kappa should be set explicitly or n_obs provided for auto-schedule.
    * POI  — xi must be > 0 for non-degenerate behaviour (default 0.01).
    """
    if method == "EI":
        return expected_improvement(mu, sigma, f_best, xi if xi is not None else 0.0)
    if method == "UCB":
        return upper_confidence_bound(mu, sigma, kappa=kappa, n_obs=n_obs)
    if method == "POI":
        return probability_of_improvement(mu, sigma, f_best, xi if xi is not None else 0.01)
    raise ValueError(f"Unknown acquisition method '{method}'. Choose EI, UCB, or POI.")


# ── Uncertainty estimation ─────────────────────────────────────────────────────

def _predict_rf_with_uncertainty(
    rf_model,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) for an RF model using per-tree prediction variance."""
    from omicstl.simulation_utils.model_utils import predict_rf_model

    preds_dict = predict_rf_model(rf_model, X)
    pred_df = pd.DataFrame(preds_dict)

    # Individual transfer-tree columns: pred_0, pred_1, … (not source, ensemble, val)
    tree_cols = [
        c for c in pred_df.columns
        if c.startswith("pred_")
        and c not in ("pred_source", "pred_ensemble")
        and not c.endswith("_val")
        and not c.endswith("_prob_1")   # classification probabilities
        and not c.endswith("_prob_2")
        and not c.endswith("_prob_3")
    ]

    if "pred_ensemble" in pred_df.columns:
        mu = pred_df["pred_ensemble"].values.astype(float)
    elif tree_cols:
        mu = pred_df[tree_cols].mean(axis=1).values.astype(float)
    else:
        mu = pred_df.iloc[:, -1].values.astype(float)

    if len(tree_cols) >= 2:
        sigma = pred_df[tree_cols].std(axis=1).values.astype(float)
    else:
        # Fallback: no variance available — use a small constant
        logger.warning("RF has fewer than 2 tree columns; setting sigma = 0.01.")
        sigma = np.full(len(mu), 0.01)

    return mu, sigma


def _predict_dl_with_uncertainty(
    dl_model,
    X: pd.DataFrame,
    model_id: str = "target",
    n_mc_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) for a DL model using Monte Carlo Dropout.

    The model is temporarily set to training mode so dropout remains active
    during inference.  n_mc_samples stochastic forward passes are averaged.
    """
    model_obj = getattr(dl_model, model_id)
    net = model_obj.model

    # Convert input once
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    if net.device is not None:
        X_tensor = X_tensor.to(net.device)

    # MC Dropout: keep network in train mode
    net.train()
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            out = net([X_tensor])
            yhat = out[0]  # first output is always the prediction
            samples.append(yhat.detach().cpu().numpy().flatten())
    net.eval()

    arr = np.stack(samples, axis=0)   # (n_mc_samples, n_points)
    mu    = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    return mu, sigma


def predict_with_uncertainty(
    model_info: dict,
    X: pd.DataFrame,
    feature_cols: list[str],
    n_mc_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) arrays for each row of X.

    Parameters
    ----------
    model_info : dict with keys 'model' and 'type' ('dl' or 'rf').
                 Matches the format returned by ``run_scenario`` / ``fit_dl_model``.
    X          : candidate DataFrame (must contain feature_cols).
    feature_cols : feature columns to pass to the model.
    n_mc_samples : number of stochastic passes for DL MC Dropout.
    """
    X_in = X[feature_cols].copy()

    if model_info["type"] == "rf":
        return _predict_rf_with_uncertainty(model_info["model"], X_in)

    if model_info["type"] == "dl":
        return _predict_dl_with_uncertainty(
            model_info["model"], X_in, n_mc_samples=n_mc_samples
        )

    raise ValueError(f"Unknown model type '{model_info['type']}'. Expected 'dl' or 'rf'.")


# ── Batch recommendation ───────────────────────────────────────────────────────

def recommend_next_batch(
    model_info: dict,
    existing_data: pd.DataFrame,
    response_col: str,
    feature_cols: list[str],
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    expansion_pct: float = 0.0,
    step_sizes: dict[str, float] | None = None,
    n_candidates: int = 5_000,
    batch_size: int = 20,
    acquisition: AcquisitionName = "EI",
    kappa: float | None = None,
    xi: float | None = None,
    n_mc_samples: int = 50,
    shortlist_pct: float = 0.10,
    seed: int = 42,
    return_candidates: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Recommend the next batch of experiments using Bayesian optimisation.

    This is a domain-agnostic function that works with any omicsTL model and
    any dataset.  It accounts for all existing observations and uses an
    acquisition function to balance exploitation and exploration.

    Parameters
    ----------
    model_info : dict
        ``{'model': <trained model>, 'type': 'dl' | 'rf'}``.
    existing_data : pd.DataFrame
        All observations collected so far (feature columns + response column).
        The response column is used to determine the current best value f*.
    response_col : str
        Name of the response column in existing_data.
    feature_cols : list[str]
        Names of the feature columns.  Must match what the model expects.
    feature_ranges : dict or None
        ``{col: (min, max)}`` search bounds representing **physically feasible**
        limits for each feature.  This is the recommended way to control the
        search space — define it based on what is experimentally possible, not
        just what has been observed so far.  If None, the observed range in
        existing_data is used and a warning is logged reminding you to set
        explicit bounds.
    expansion_pct : float
        Fractional expansion applied symmetrically to each feature's range
        *after* the range is resolved (whether from feature_ranges or from the
        observed data).  0.0 (default) means no expansion.  0.1 expands each
        side by 10 % of the span, so a range of [0, 100] becomes [-10, 110].
        Useful as a lightweight exploration knob when physical bounds are not
        known in advance.  Ignored when expansion_pct=0.0.
    step_sizes : dict or None
        ``{col: step}`` per-feature grid spacing.  Omit a feature to keep it
        continuous.  Examples::

            step_sizes = {
                "concentration_mM": 25,   # 0, 25, 50, 75, 100
                "pH": 0.5,                # 6.0, 6.5, 7.0 …
                "temperature": 0.25,      # 30.0, 30.25, 30.5 …
            }

        The step value also determines output precision: integer step → integer
        values, 0.5 → one decimal place, 0.25 → two decimal places.
        Features absent from this dict are sampled as continuous floats.
    n_candidates : int
        Number of random candidate points to score.
    batch_size : int
        Number of experiments to recommend.
    acquisition : str
        Acquisition function: ``'EI'`` (default), ``'UCB'``, or ``'POI'``.
        EI is recommended — it self-balances explore/exploit with no tuning.
    kappa : float or None
        UCB exploration weight (ignored for EI and POI).  If None (default),
        auto-computed as ``sqrt(2 * ln(n_obs))`` — the theoretically optimal
        schedule that grows with the dataset so exploration is maintained.
        Pass a fixed float to override (e.g. ``kappa=2.0``).
    xi : float or None
        Improvement buffer for EI and POI (ignored for UCB).
        If None: EI uses 0.0 (standard), POI uses 0.01 (non-degenerate default).
        With xi=0, POI is degenerate — it treats any improvement, however tiny,
        identically.  EI does not have this problem.
    n_mc_samples : int
        Number of Monte Carlo Dropout passes for DL models.
    shortlist_pct : float
        Fraction of Sobol candidates to shortlist by acquisition score before
        applying max-min diversity selection.  Default 0.10 (top 10 %).
        The shortlist size is floored at batch_size so selection always succeeds.
        Smaller values keep quality high but reduce room for diversity; larger
        values spread points more but include lower-EI candidates.
    seed : int
        Random seed for reproducible candidate sampling.
    return_candidates : bool
        If True, return ``(batch, candidates)`` — a tuple where the second
        element is the full scored candidate pool (all n_candidates rows with
        'predicted_mean', 'predicted_std', 'acquisition_score').  Useful for
        diagnostic plots.  Default False (return batch only).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        When return_candidates=False (default): the recommended batch only.
        When return_candidates=True: ``(batch, candidates)`` tuple.
        Batch columns: feature_cols + ['predicted_mean', 'predicted_std',
        'acquisition_score', 'batch_rank'].  Sorted by batch_rank.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Feature ranges ──────────────────────────────────────────────────
    if feature_ranges is None:
        logger.warning(
            "feature_ranges was not provided.  Candidate search space is being "
            "inferred from the observed data range, which restricts the optimizer "
            "to interpolation only — it cannot suggest experiments outside the "
            "values already seen.  Pass feature_ranges={col: (min, max)} with "
            "physically meaningful bounds (like BacterAI's pre-specified "
            "concentration levels) to allow exploration of the full feasible "
            "space.  Use expansion_pct to expand the observed range by a fixed "
            "fraction if explicit bounds are not available."
        )
        feature_ranges = {
            col: (float(existing_data[col].min()), float(existing_data[col].max()))
            for col in feature_cols
        }

    # Apply symmetric expansion if requested
    if expansion_pct > 0.0:
        expanded = {}
        for col, (lo, hi) in feature_ranges.items():
            margin = (hi - lo) * expansion_pct
            expanded[col] = (lo - margin, hi + margin)
            logger.debug(
                "Feature '%s': expanded [%.4g, %.4g] → [%.4g, %.4g] "
                "(expansion_pct=%.2f)",
                col, lo, hi, lo - margin, hi + margin, expansion_pct,
            )
        feature_ranges = expanded

    for col in feature_cols:
        lo, hi = feature_ranges[col]
        if lo == hi:
            feature_ranges[col] = (lo - 1e-6, hi + 1e-6)

    # ── 2. Generate candidates (Sobol low-discrepancy sequences) ──────────
    # Sobol requires power-of-2 sample sizes for optimal coverage.
    # Discrete features (step_sizes) are mapped from the Sobol [0,1) dimension
    # to allowed levels proportionally, ensuring uniform level distribution.
    step_sizes = step_sizes or {}
    _allowed: dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col in step_sizes:
            lo, hi = feature_ranges[col]
            lvls = _make_allowed_values(lo, hi, step_sizes[col])
            if len(lvls) == 0:
                raise ValueError(
                    f"step_sizes['{col}']={step_sizes[col]} produces no allowed "
                    f"values in range [{lo}, {hi}]."
                )
            _allowed[col] = lvls
            logger.debug(
                "Feature '%s': %d discrete levels (step=%s, range=[%s, %s])",
                col, len(lvls), step_sizes[col], lo, hi,
            )

    n_sobol = _next_power_of_2(n_candidates)
    if n_sobol != n_candidates:
        logger.info(
            "Sobol requires power-of-2 sample size; rounding n_candidates %d → %d.",
            n_candidates, n_sobol,
        )

    # scramble=True randomises the sequence while preserving low discrepancy;
    # seed ensures reproducibility across calls.
    sampler    = _qmc.Sobol(d=len(feature_cols), scramble=True, seed=seed)
    sobol_unit = sampler.random(n_sobol)   # shape (n_sobol, n_features), values in [0, 1)

    cand_dict: dict[str, np.ndarray] = {}
    for j, col in enumerate(feature_cols):
        lo, hi = feature_ranges[col]
        u = sobol_unit[:, j]
        if col in _allowed:
            # Discrete: map [0, 1) uniformly to allowed levels — gives exactly
            # n_sobol / n_levels candidates per level (perfectly stratified).
            lvls = _allowed[col]
            idx  = np.minimum((u * len(lvls)).astype(int), len(lvls) - 1)
            cand_dict[col] = lvls[idx]
        else:
            # Continuous: linear scale from [0, 1) to [lo, hi).
            cand_dict[col] = lo + u * (hi - lo)

    candidates = pd.DataFrame(cand_dict)

    # ── 3. Predict mean and std ────────────────────────────────────────────
    logger.info(
        "Scoring %d Sobol candidates with %s acquisition (model type: %s)",
        n_sobol, acquisition, model_info["type"],
    )
    mu, sigma = predict_with_uncertainty(
        model_info, candidates, feature_cols, n_mc_samples
    )
    candidates["predicted_mean"] = mu
    candidates["predicted_std"]  = sigma

    # ── 4. Acquisition score ───────────────────────────────────────────────
    f_best = float(existing_data[response_col].max())
    logger.info("Current best observed response: %.6f", f_best)

    n_obs = len(existing_data)
    scores = compute_acquisition(
        mu, sigma, f_best,
        method=acquisition, kappa=kappa, xi=xi, n_obs=n_obs,
    )
    candidates["acquisition_score"] = scores

    # ── 5. Select batch: top-EI shortlist + greedy max-min diversity ──────
    norm_candidates = np.column_stack([
        (candidates[col].values - feature_ranges[col][0])
        / (feature_ranges[col][1] - feature_ranges[col][0])
        for col in feature_cols
    ])

    acq_vals    = candidates["acquisition_score"].values
    selected_idx = _maximin_select(
        norm_candidates, acq_vals, batch_size, shortlist_pct,
    )

    logger.info(
        "Selected %d candidates via top-%.0f%% EI shortlist + max-min diversity.",
        len(selected_idx), shortlist_pct * 100,
    )

    # ── 6. Build output DataFrame ──────────────────────────────────────────
    empty_batch = pd.DataFrame(
        columns=feature_cols
        + ["predicted_mean", "predicted_std", "acquisition_score", "batch_rank"]
    )
    if len(selected_idx) == 0:
        logger.error("No candidates were selected.")
        return (empty_batch, candidates) if return_candidates else empty_batch

    # Rank within the batch by acquisition score (rank 1 = highest EI)
    batch_acq  = acq_vals[selected_idx]
    rank_order = np.argsort(-batch_acq)          # descending EI
    sorted_idx = selected_idx[rank_order]

    result = candidates.iloc[sorted_idx][
        feature_cols + ["predicted_mean", "predicted_std", "acquisition_score"]
    ].copy()
    result["batch_rank"] = np.arange(1, len(sorted_idx) + 1)
    result = result.reset_index(drop=True)

    logger.info(
        "Best acquisition score: %.6f   Best predicted mean: %.6f",
        result["acquisition_score"].iloc[0],
        result["predicted_mean"].max(),
    )

    return (result, candidates) if return_candidates else result
