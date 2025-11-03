"""
Argument parser and command entry points for the unified configuration tool.
"""
import argparse
import sys
from typing import Optional, Union

from bacterai.cli import ingredients as cli_ingredients
from bacterai.setup import setup_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified configuration tool for experiments and ingredients.\n"
            "Use `<program> <command> -h` for command-specific help."
        )
    )
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    # Experiment subcommand (now routed to interactive setup when needed)
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
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing files",
    )
    exp.set_defaults(func=experiment_wrapper)

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
    ing.set_defaults(func=ingredients_wrapper)
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
    setup_experiments(
        input_path=args.input,
        sheet=sheet,
        force_format=args.force_format,
        outfile_name=args.outfile_name,
        dry_run=args.dry_run,
    )


def ingredients_wrapper(args) -> None:
    sheet = _coerce_sheet_arg(args.sheet)
    cli_ingredients(
        input_path=args.input,
        sheet=sheet,
        output=args.output,
        id_map_path=args.id_map,
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