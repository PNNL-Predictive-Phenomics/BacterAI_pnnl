"""
CLI-facing functions for the unified configuration tool.
All user interaction (printing, file writing decisions) happens here.
"""
import json
import os
import sys
from typing import Optional, Union, List

from bacterai.configuration import ExperimentConfig, IngredientsConfig, to_json_string


def experiment(
    input_path: str,
    sheet: Optional[Union[str, int]],
    outfile_name: str,
    force_format: Optional[str],
    dry_run: bool,
) -> None:
    """
    Convert experiment settings (CSV/XLSX) to JSON file(s).
    Prints status messages and writes files unless dry_run is True.
    """
    cfgs = ExperimentConfig.parse_configs(
        input_path=input_path,
        sheet=sheet,
        force_format=force_format,
    )
    written: List[str] = []

    for cfg in cfgs:
        exp_path = cfg.get("experiment_path")
        if not exp_path:
            print(f"Skipping entry without experiment_path: {cfg}", file=sys.stderr)
            continue
        os.makedirs(exp_path, exist_ok=True)
        out_path = os.path.join(exp_path, outfile_name)
        if dry_run:
            print(f"[DRY-RUN] Would write: {out_path}")
            print(to_json_string(cfg))
        else:
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=2)
            written.append(out_path)

    if not dry_run:
        if written:
            print("Wrote config file(s):")
            for p in written:
                print(f"  - {p}")
        else:
            print("No config files written.", file=sys.stderr)


def ingredients(
    input_path: str,
    sheet: Optional[Union[str, int]],
    output: Optional[str],
    id_map_path: Optional[str],
) -> None:
    """
    Convert ingredient sheets (Format 1 or 2; CSV/XLSX) to a JSON payload.
    Writes to output if provided, otherwise prints to stdout.
    """
    result = IngredientsConfig.parse_ingredients(
        input_path=input_path,
        sheet=sheet,
        id_map_path=id_map_path,
    )
    out_str = to_json_string(result)
    if output:
        with open(output, "w") as f:
            f.write(out_str)
        print(f"Wrote: {output}")
    else:
        print(out_str)