"""Aggregate RoboCasa-Kitchen per-task eval results into suite + overall SR.

Reads `<run_dir>/<TASK>/simulation_results.csv` for the 24 RoboCasa-Kitchen tasks (each
CSV has one row per episode with a `success` column), prints PnP / Open-Close / Other
suite averages and the overall SR, and writes it to `<run_dir>/model_summary.txt`.

Usage:
    python gr00t/eval/sim/robocasa/aggregate_eval_summary.py <run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TASKS = [
    "TurnSinkSpout", "TurnOnStove", "TurnOnSinkFaucet", "TurnOnMicrowave",
    "TurnOffStove", "TurnOffSinkFaucet", "TurnOffMicrowave", "PnPStoveToCounter",
    "PnPSinkToCounter", "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPCounterToSink",
    "PnPCounterToMicrowave", "PnPCounterToCab", "PnPCabToCounter", "OpenSingleDoor",
    "OpenDrawer", "OpenDoubleDoor", "CoffeeSetupMug", "CoffeeServeMug",
    "CoffeePressButton", "CloseSingleDoor", "CloseDrawer", "CloseDoubleDoor",
]
SUITES = {
    "PnP": [t for t in TASKS if "PnP" in t],
    "OpenClose": [t for t in TASKS if ("Open" in t or "Close" in t) and "PnP" not in t],
    "Other": [t for t in TASKS if "PnP" not in t and "Open" not in t and "Close" not in t],
}


def _task_sr(run_dir: Path, task: str) -> float | None:
    csv = run_dir / task / "simulation_results.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    return float(df["success"].mean()) if len(df) else None


def main(run_dir: str) -> None:
    run = Path(run_dir)
    per_task = {t: _task_sr(run, t) for t in TASKS}
    done = {t: v for t, v in per_task.items() if v is not None}

    lines = [f"RoboCasa-Kitchen eval summary: {run}", ""]
    for suite, tasks in SUITES.items():
        vals = [per_task[t] for t in tasks if per_task[t] is not None]
        m = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"[{suite}] {100 * m:.2f}%  ({len(vals)}/{len(tasks)} tasks)")
    lines.append("")
    for t in TASKS:
        v = per_task[t]
        lines.append(f"    {t:<22} {'--' if v is None else f'{100 * v:.2f}%'}")
    overall = sum(done.values()) / len(done) if done else float("nan")
    lines += ["", f"Total avg ({len(done)}/24 tasks): {100 * overall:.2f}%"]

    summary = "\n".join(lines)
    print(summary)
    (run / "model_summary.txt").write_text(summary + "\n")
    print(f"\n[i] wrote {run / 'model_summary.txt'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
