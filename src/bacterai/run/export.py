"""Export utilities for BacterAI batches."""

import csv
import os
from typing import List
import pandas as pd


def export_to_dp_batch(
    parent_path, batch, ingredient_names, date, nickname=None, is_redo=False
):
    """Export the BacterAI batch to a Deep Phenotyping-compatible file."""
    if len(batch) == 0:
        print("Empty Batch: No files generated.")
        return
        
    n_ingredients = len(ingredient_names)
    batch = batch.rename(
        columns={a: b for a, b in zip(range(n_ingredients), ingredient_names)}
    )
    batch = batch.sort_values(by=["growth_pred", "var"], ascending=[False, True])
    
    meta_file_name = (
        f"batch_redo_meta_{date}.csv" if is_redo else f"batch_meta_{date}.csv"
    )
    batch.to_csv(os.path.join(parent_path, meta_file_name), index=None)
    
    # DeepPhenotyping compatible list
    batch = batch.drop(columns=batch.columns[n_ingredients:])
    nickname = f"_{nickname}" if nickname != None else ""
    dp_file_name = (
        f"batch_redo_dp{nickname}_{date}.csv"
        if is_redo
        else f"batch_dp{nickname}_{date}.csv"
    )
    
    with open(os.path.join(parent_path, dp_file_name), "w") as f:
        writer = csv.writer(f, delimiter=",")
        for _, row_data in batch.iterrows():
            row_data = row_data[row_data == 0]
            removed_ingredients = list(row_data.index.to_numpy())
            writer.writerow(removed_ingredients)