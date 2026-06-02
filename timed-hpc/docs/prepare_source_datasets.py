"""
Prepare three taxonomically stratified source datasets from the P. putida Fitness Browser.

Inputs (Stone_adaption/):
    fb_fdata_cleaned.csv  — experimental conditions + taxonomy (1,494 experiments,
                            all Gammaproteobacteria, already curated)
    fitness_e_data.csv    — per-gene fitness values, wide format:
                            rows = genes (locusIdPutida), columns = experiments
    ingredients.json      — biologist-defined search space: MIN/MAX for the 11
                            predictor variables and fixed values for media components

Outputs (docs/data/):
    pputida_source_putida.csv              — 261 rows  (Pseudomonas putida only)
    pputida_source_pseudomonas.csv         — 1,079 rows (all Pseudomonas spp.)
    pputida_source_gammaproteobacteria.csv — 1,494 rows (all Gammaproteobacteria)
    feature_config.json                    — feature_ranges, step_sizes, units, and
                                             fixed variable values for use in
                                             recommend_next_batch

Response variable construction (Resp, range 0–1):
    1. Extract fitness values for 3 key metabolic genes:
           pykA → PP_1362,  pykF → PP_4301,  ppc → PP_1505
       These are P. putida locus IDs used as a common ortholog reference across
       all organisms in the fitness browser.
    2. Mean across the 3 genes per experiment (NaN-safe)
    3. Robust z-score: (x − median) / IQR  — computed *within each subset*
       so the response is relative to the organism group being used as source
    4. Sigmoid: 1 / (1 + exp(−z))  — maps to (0, 1)

Usage:
    cd timed-hpc/docs
    python prepare_source_datasets.py
"""

import csv
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE            = pathlib.Path(__file__).parent          # timed-hpc/docs/
_PPI_DIR         = _HERE.parent.parent                    # PPI TIMED/
STONE_DIR        = _PPI_DIR / "Stone_adaption"
OUT_DIR          = _HERE / "data"

FB_PATH          = STONE_DIR / "fb_fdata_cleaned.csv"
FITNESS_PATH     = STONE_DIR / "fitness_e_data.csv"
INGREDIENTS_PATH = STONE_DIR / "ingredients.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Column definitions ─────────────────────────────────────────────────────────
CONDITIONS = [
    "d_glucose", "sodium_citrate", "sodium_octanoate", "sodium_acetate",
    "sodium_benzoate", "d_xylose", "l_arabinose", "sodium_chloride",
    "urea", "ammonium_chloride", "pH",
]

# fitness_e_data.csv locus IDs → output column names
GENE_MAP = {
    "PP_1362": "fitness_pykA_PP_1362",
    "PP_4301": "fitness_pykF_PP_4301",
    "PP_1505": "fitness_ppc_PP_1505",
}
GENE_COLS = list(GENE_MAP.values())

# ── Step 0: Parse ingredients.json → feature_config.json ─────────────────────
print("=" * 60)
print("Step 0 — Extracting feature config from ingredients.json")
print("=" * 60)

with open(INGREDIENTS_PATH, encoding="utf-8") as fh:
    ingredients = json.load(fh)["ingredients"]

PREDICTOR_TYPES = {"quantitative", "semi-quantitative"}

feature_ranges: dict[str, list[float]] = {}
step_sizes:     dict[str, float]       = {}
units:          dict[str, str]         = {}
fixed_variables: dict[str, dict]       = {}

for ing in ingredients:
    name  = ing["INGREDIENT"]
    itype = ing["TYPE"]
    lo    = ing["MIN_VALUE"]
    hi    = ing["MAX_VALUE"]
    nom   = ing["NOMINAL_VALUE"]
    unit  = ing["UNIT"]

    if itype in PREDICTOR_TYPES:
        # Only include the 11 known condition columns
        if name in CONDITIONS:
            feature_ranges[name] = [lo, hi]
            units[name] = unit

            # Semi-quantitative: compute step from N_STATES
            # e.g. pH N_STATES=3, range [5,9] → step = (9-5)/(3-1) = 2.0
            n_states = ing.get("N_STATES")
            if n_states and n_states > 1:
                step = (hi - lo) / (n_states - 1)
                step_sizes[name] = step

    else:
        # Fixed media component or strain — not varied in experiments
        fixed_variables[name] = {
            "value": nom,
            "unit":  unit,
            "type":  itype,
            "id":    ing["ID"],
        }

# Ensure order matches CONDITIONS list
feature_ranges = {c: feature_ranges[c] for c in CONDITIONS if c in feature_ranges}
units          = {c: units[c]          for c in CONDITIONS if c in units}

print(f"\nPredictor variables ({len(feature_ranges)}):")
print(f"  {'Condition':<22}  {'Min':>8}  {'Max':>8}  {'Unit':<12}  Step")
print("  " + "-" * 60)
for col in CONDITIONS:
    lo, hi = feature_ranges[col]
    step   = step_sizes.get(col, "continuous")
    print(f"  {col:<22}  {lo:>8.3g}  {hi:>8.3g}  {units[col]:<12}  {step}")

print(f"\nFixed variables ({len(fixed_variables)}):")
for name, info in fixed_variables.items():
    print(f"  {name:<40}  {info['value']}  {info['unit']}  [{info['type']}]")

# Save feature_config.json
feature_config = {
    "feature_ranges": feature_ranges,
    "step_sizes":     step_sizes,
    "units":          units,
    "fixed_variables": fixed_variables,
}
config_path = OUT_DIR / "feature_config.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(config_path, "w", encoding="utf-8") as fh:
    json.dump(feature_config, fh, indent=2)
print(f"\nSaved: {config_path.name}")


# ── Step 1: Stream 3-gene fitness rows from wide fitness_e_data ────────────────
print("=" * 60)
print("Step 1 — Extracting gene fitness (streaming fitness_e_data.csv)")
print("=" * 60)

gene_rows: dict[str, dict[str, float]] = {}   # {locusId: {expId: value}}
exp_ids: list[str] = []

with open(FITNESS_PATH, newline="", encoding="utf-8") as fh:
    reader = csv.reader(fh)
    for line_num, row in enumerate(reader):
        if line_num == 0:
            # First column is locusIdPutida; remainder are experiment IDs
            exp_ids = [c.strip('"') for c in row[1:]]
            continue

        locus = row[0].strip('"')
        if locus in GENE_MAP:
            gene_rows[locus] = {}
            for j, val in enumerate(row[1:]):
                v = val.strip('"').strip()
                gene_rows[locus][exp_ids[j]] = (
                    float(v) if v not in ("", "NA", "NaN", "na", "nan") else np.nan
                )
            print(f"  Found {locus:10s}  (line {line_num + 1:>5d})")

        if len(gene_rows) == len(GENE_MAP):
            print(f"  All {len(GENE_MAP)} genes found — stopping early.")
            break

if len(gene_rows) < len(GENE_MAP):
    missing = set(GENE_MAP) - set(gene_rows)
    raise RuntimeError(
        f"Could not find gene(s) {missing} in {FITNESS_PATH.name}. "
        "Check that the locus IDs match."
    )

# Build tidy DataFrame: rows = experiments, columns = gene locus IDs
fitness_wide = (
    pd.DataFrame(gene_rows)           # index = experiment IDs, cols = locus IDs
    .rename(columns=GENE_MAP)         # PP_1362 → fitness_pykA_PP_1362
)
fitness_wide.index.name = "expName"

print(f"\nGene fitness matrix: {len(fitness_wide)} experiments × {len(GENE_COLS)} genes")
for col in GENE_COLS:
    n_nan = fitness_wide[col].isna().sum()
    print(f"  {col:<30s}  NaN: {n_nan:>4d} ({100*n_nan/len(fitness_wide):.1f}%)")


# ── Step 2: Load conditions + taxonomy and merge ───────────────────────────────
print("\n" + "=" * 60)
print("Step 2 — Loading conditions and merging")
print("=" * 60)

fb = pd.read_csv(FB_PATH)
print(f"fb_fdata_cleaned: {len(fb)} rows")
print("\nTaxonomy breakdown:")
print(
    fb.groupby(["division", "genus", "species"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .to_string(index=False)
)

merged = fb.merge(
    fitness_wide.reset_index(),
    on="expName",
    how="inner",
)
print(f"\nAfter merge: {len(merged)} rows  (expected {len(fb)})")
if len(merged) != len(fb):
    print(f"  WARNING: {len(fb) - len(merged)} rows lost — check expName alignment.")


# ── Step 3: Response variable ─────────────────────────────────────────────────
def compute_response(subset: pd.DataFrame) -> pd.Series:
    """
    Mean gene fitness (3 genes, NaN-safe) → robust z-score → sigmoid → (0, 1).

    The robust z-score uses median and IQR computed *within the subset*, so
    the response is calibrated to the organism group, not the full dataset.
    Falls back to std-based z-score if IQR is degenerate (< 1e-9).
    """
    mean_fit = subset[GENE_COLS].astype(float).mean(axis=1, skipna=True)
    med      = mean_fit.median()
    iqr_val  = scipy_iqr(mean_fit.dropna())
    if iqr_val < 1e-9:
        iqr_val = float(mean_fit.std())
    z = (mean_fit - med) / max(iqr_val, 1e-9)
    return (1.0 / (1.0 + np.exp(-z))).rename("Resp")


# ── Step 4: Build and save each source dataset ────────────────────────────────
print("\n" + "=" * 60)
print("Step 4 — Building and saving source datasets")
print("=" * 60)

SUBSETS = {
    "putida": merged[merged["species"] == "putida"].copy(),
    "pseudomonas": merged[merged["genus"] == "Pseudomonas"].copy(),
    "gammaproteobacteria": merged.copy(),
}

OUTPUT_COLS = ["SampleID", "Resp"] + CONDITIONS + GENE_COLS
summary_rows = []

for label, subset in SUBSETS.items():
    subset = subset.reset_index(drop=True)
    subset["Resp"] = compute_response(subset)
    subset = subset.rename(columns={"expName": "SampleID"})

    out = subset[OUTPUT_COLS]
    out_path = OUT_DIR / f"pputida_source_{label}.csv"
    out.to_csv(out_path, index=False)

    resp = subset["Resp"]
    genus_counts = subset["genus"].value_counts().to_dict()
    print(f"\n  [{label}]")
    print(f"    Rows    : {len(subset)}")
    print(f"    Genera  : {genus_counts}")
    print(f"    Resp    : min={resp.min():.4f}  max={resp.max():.4f}  "
          f"mean={resp.mean():.4f}  std={resp.std():.4f}")
    print(f"    Saved   : {out_path.name}")

    summary_rows.append({
        "dataset": label,
        "n_rows": len(subset),
        "genera": str(genus_counts),
        "resp_min": round(resp.min(), 4),
        "resp_max": round(resp.max(), 4),
        "resp_mean": round(resp.mean(), 4),
        "resp_std": round(resp.std(), 4),
        "path": str(out_path),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
summary_df = pd.DataFrame(summary_rows).set_index("dataset")
print(summary_df[["n_rows", "resp_min", "resp_max", "resp_mean", "resp_std"]].to_string())
print(f"\nAll files written to: {OUT_DIR}")
