import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PAIR_CHECKER_SCRIPT = Path("pair-checker.py")
READY_SIGNALS_SCRIPT = Path("paper-trading-ready.py")
ALPACA_SCRIPT = Path("alpaca-paper-trading.py")


def run_step(command: List[str], label: str) -> None:
    """Run one pipeline step and stop on failure."""
    print(f"\n=== {label} ===")
    print(" ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pair trading research and execution pipeline.")
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip pair-checker.py and reuse existing research outputs.",
    )
    parser.add_argument(
        "--skip-ready",
        action="store_true",
        help="Skip paper-trading-ready.py and reuse the existing ready-signals file.",
    )
    parser.add_argument(
        "--skip-alpaca",
        action="store_true",
        help="Skip the Alpaca preview/execution step.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Pass --execute to alpaca-paper-trading.py.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Pass --allow-stale to alpaca-paper-trading.py.",
    )
    args = parser.parse_args()

    python_executable = sys.executable

    if not args.skip_research:
        run_step([python_executable, str(PAIR_CHECKER_SCRIPT)], "Research")

    if not args.skip_ready:
        run_step([python_executable, str(READY_SIGNALS_SCRIPT)], "Ready Signals")

    if not args.skip_alpaca:
        alpaca_command = [python_executable, str(ALPACA_SCRIPT)]
        if args.execute:
            alpaca_command.append("--execute")
        if args.allow_stale:
            alpaca_command.append("--allow-stale")
        run_step(alpaca_command, "Alpaca")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
