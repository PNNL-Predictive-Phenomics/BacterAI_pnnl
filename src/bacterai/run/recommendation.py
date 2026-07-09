from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0) -> np.ndarray:
    sigma = np.clip(sigma, 1e-12, None)
    z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return np.clip(ei, 0.0, None)


def _infer_decimals(step: float) -> int:
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _quantize_column(values: np.ndarray, lo: float, hi: float, step: float) -> np.ndarray:
    if step <= 0:
        return values
    n = np.round((values - lo) / step)
    out = lo + n * step
    return np.clip(np.round(out, _infer_decimals(step)), lo, hi)


def _maximin_select(norm_candidates: np.ndarray, scores: np.ndarray, batch_size: int, shortlist_pct: float = 0.05):
    n = len(scores)
    k = min(n, max(batch_size, int(np.ceil(n * shortlist_pct))))
    if k <= batch_size:
        return np.argpartition(scores, -batch_size)[-batch_size:]

    shortlist_idx = np.argpartition(scores, -k)[-k:]
    shortlist = norm_candidates[shortlist_idx]
    shortlist_scores = scores[shortlist_idx]

    diff = shortlist[:, None, :] - shortlist[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    first = int(np.argmax(shortlist_scores))
    selected = [first]
    min_dist = dist[:, first].copy()
    min_dist[first] = -np.inf

    for _ in range(batch_size - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist[:, nxt])
        min_dist[nxt] = -np.inf

    return shortlist_idx[np.array(selected)]


def _predict_rf_mu_sigma(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pred_dict = model.generate_predictions([X])[0]

    preferred = [
        "pred_ensemble_full",
        "pred_ensemble",
        "pred_0",
        "pred_1",
        "pred_2",
        "pred_3",
    ]

    mu = None
    for key in preferred:
        if key in pred_dict:
            mu = np.asarray(pred_dict[key], dtype=float)
            break
    if mu is None:
        first = next(iter(pred_dict.values()))
        mu = np.asarray(first, dtype=float)

    component_keys = [k for k in ("pred_0", "pred_1", "pred_2", "pred_3") if k in pred_dict]
    if len(component_keys) >= 2:
        stacked = np.column_stack([np.asarray(pred_dict[k], dtype=float) for k in component_keys])
        sigma = np.std(stacked, axis=1)
    else:
        sigma = np.full_like(mu, 0.01, dtype=float)

    return mu, sigma


def recommend_next_batch(
    model,
    existing_data: pd.DataFrame,
    response_col: str,
    feature_cols: List[str],
    feature_ranges: Dict[str, Tuple[float, float]],
    step_sizes: Dict[str, float],
    batch_size: int,
    acquisition: str = "EI",
    n_candidates: int = 8192,
    n_mc_samples: int = 50,
    shortlist_pct: float = 0.05,
    seed: int = 42,
    return_candidates: bool = False,
):
    _ = n_mc_samples
    if acquisition != "EI":
        raise ValueError("Only EI acquisition is currently supported in the native transfer RF path.")

    sampler = qmc.Sobol(d=len(feature_cols), scramble=True, seed=seed)
    power = int(np.ceil(np.log2(max(2, n_candidates))))
    u = sampler.random_base2(m=power)
    candidates = u[:n_candidates, :]

    lows = np.asarray([feature_ranges[c][0] for c in feature_cols], dtype=float)
    highs = np.asarray([feature_ranges[c][1] for c in feature_cols], dtype=float)
    scaled = qmc.scale(candidates, lows, highs)

    cand_df = pd.DataFrame(scaled, columns=feature_cols)
    for col in feature_cols:
        step = step_sizes.get(col)
        if step is None:
            continue
        lo, hi = feature_ranges[col]
        cand_df[col] = _quantize_column(cand_df[col].to_numpy(dtype=float), lo, hi, float(step))

    existing = existing_data[feature_cols].drop_duplicates()
    merged = cand_df.merge(existing.assign(__seen=1), on=feature_cols, how="left")
    cand_df = merged[merged["__seen"].isna()].drop(columns=["__seen"])

    if cand_df.empty:
        raise RuntimeError("Candidate generation produced no novel experiments.")

    mu, sigma = _predict_rf_mu_sigma(model, cand_df[feature_cols])
    f_best = float(existing_data[response_col].max())
    acq = expected_improvement(mu, sigma, f_best=f_best, xi=0.0)

    norm = (cand_df[feature_cols].to_numpy(dtype=float) - lows) / np.clip(highs - lows, 1e-12, None)
    selected_idx = _maximin_select(norm, acq, batch_size=batch_size, shortlist_pct=shortlist_pct)

    out = cand_df.iloc[selected_idx].copy().reset_index(drop=True)
    out["predicted_mean"] = mu[selected_idx]
    out["predicted_std"] = sigma[selected_idx]
    out["acquisition_score"] = acq[selected_idx]
    out["batch_rank"] = np.arange(1, len(out) + 1)
    out = out.sort_values("batch_rank", ascending=True).reset_index(drop=True)

    if return_candidates:
        cand_out = cand_df.copy()
        cand_out["predicted_mean"] = mu
        cand_out["predicted_std"] = sigma
        cand_out["acquisition_score"] = acq
        return out, cand_out

    return out
