"""
Argument parser and command entry points for the unified configuration tool.
"""
import argparse
import sys
from typing import Optional, Union

from bacterai.cli import ingredients as cli_ingredients, setup as cli_setup, experiment as cli_experiment, run as cli_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified configuration tool for experiments and ingredients.\n"
            "Use `<program> <command> -h` for command-specific help."
        )
    )
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    # Experiment subcommand
    exp = subparsers.add_parser("experiment", help="Convert experiment settings (CSV/XLSX) to JSON file(s).")
    exp.add_argument("input", help="Path to input .csv or .xlsx file")
    exp.add_argument(
        "-s", "--sheet",
        help="Excel sheet name or index (for xlsx); defaults to first sheet",
        default=None,
    )
    exp.add_argument(
        "--outfile-name",
        help="Output JSON filename to write in each experiment folder",
        default="config.json",
    )
    exp.add_argument(
        "--force-format",
        choices=["vertical", "horizontal"],
        help="Force input format if auto-detection fails",
    )
    exp.add_argument(
        "--verbose",
        action="store_true",
        help="Print file contents to console in addition to writing files",
    )
    exp.set_defaults(func=experiment_wrapper)

    # Setup subcommand (interactive experiment setup)
    setup = subparsers.add_parser("setup", help="Interactive setup of experiment with prompts for ingredients.")
    setup.add_argument("input", help="Path to input .csv or .xlsx file")
    setup.add_argument(
        "-s", "--sheet",
        help="Excel sheet name or index (for xlsx); defaults to first sheet",
        default=None,
    )
    setup.add_argument(
        "--outfile-name",
        help="Output JSON filename to write in each experiment folder",
        default="config.json",
    )
    setup.add_argument(
        "--force-format",
        choices=["vertical", "horizontal"],
        help="Force input format if auto-detection fails",
    )
    setup.add_argument(
        "--verbose",
        action="store_true",
        help="Print file contents to console in addition to writing files",
    )
    setup.set_defaults(func=setup_wrapper)

    # Ingredients subcommand remains as-is
    ing = subparsers.add_parser("ingredients", help="Convert ingredient sheets (Format 1 or 2; CSV/XLSX) to JSON.")
    ing.add_argument("input", help="Path to input .csv or .xlsx file")
    ing.add_argument(
        "-s", "--sheet",
        help="Excel sheet name (if xlsx); defaults to first sheet",
        default=None,
    )
    ing.add_argument(
        "-o", "--output",
        help="Path to output .json (default: stdout)",
        default=None,
    )
    ing.add_argument(
        "--id-map",
        help="Optional CSV mapping of INGREDIENT to ID (columns: INGREDIENT,ID)",
        default=None,
    )
    ing.add_argument(
        "--verbose",
        action="store_true",
        help="Print file contents to console in addition to writing files",
    )
    ing.set_defaults(func=ingredients_wrapper)

    # Run subcommand (execute experiment rounds)
    run = subparsers.add_parser("run", help="Execute an experiment round using BacterAI models and simulations.")
    run.add_argument("experiment_path", help="Path to experiment directory.")
    run.add_argument(
        "-p", "--plot-only",
        action="store_true",
        help="Only generate plots, do not run experiments",
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output during execution",
    )
    run.set_defaults(func=run_wrapper)
    
    return parser


def _coerce_sheet_arg(raw: Optional[str]) -> Optional[Union[str, int]]:
    if raw is None:
        return None
    s = raw.strip()
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return s
    return s


def experiment_wrapper(args) -> None:
    sheet = _coerce_sheet_arg(args.sheet)
    cli_experiment(
        input_path=args.input,
        sheet=sheet,
        outfile_name=args.outfile_name,
        force_format=args.force_format,
        verbose=args.verbose,
    )


def setup_wrapper(args) -> None:
    sheet = _coerce_sheet_arg(args.sheet)
    cli_setup(
        input_path=args.input,
        sheet=sheet,
        force_format=args.force_format,
        outfile_name=args.outfile_name,
        verbose=args.verbose,
    )


def ingredients_wrapper(args) -> None:
    sheet = _coerce_sheet_arg(args.sheet)
    cli_ingredients(
        input_path=args.input,
        sheet=sheet,
        output=args.output,
        id_map_path=args.id_map,
        verbose=args.verbose,
    )


def run_wrapper(args) -> None:
    cli_run(
        experiment_path=args.experiment_path,
        plot_only=args.plot_only,
        verbose=args.verbose,
    )


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()