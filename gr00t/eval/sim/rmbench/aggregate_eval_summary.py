#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate RMBench per-task eval results into suite + overall success rates.

Usage: python gr00t/eval/sim/rmbench/aggregate_eval_summary.py runs/eval/rmbench/<MODEL_TAG>

Reads <root>/<task>/_result.txt (each holds a `Success Rate: <float in [0,1]>` line written by
RMBench's eval_policy.py) and reports the official 9 tasks grouped by memory-complexity tier:
  M(1) single memory event (5): observe_and_pickup, rearrange_blocks, put_back_block, swap_blocks, swap_T
  M(n) repeated/multi memory (4): battery_try, blocks_ranking_try, cover_blocks, press_button
For the HAMLET memory-lift, run this on the vanilla and the HAMLET output dirs and diff the OVERALL rows.
"""
import glob
import os
import re
import sys

import numpy as np

M1 = ["observe_and_pickup", "rearrange_blocks", "put_back_block", "swap_blocks", "swap_T"]
MN = ["battery_try", "blocks_ranking_try", "cover_blocks", "press_button"]
TASKS = M1 + MN


def read_sr(task_dir):
    cands = sorted(glob.glob(os.path.join(task_dir, "**", "_result.txt"), recursive=True))
    direct = os.path.join(task_dir, "_result.txt")
    if os.path.exists(direct):
        cands.append(direct)
    if not cands:
        return None
    m = re.search(r"Success Rate:\s*([0-9.]+)", open(cands[-1]).read())
    return float(m.group(1)) if m else None


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate_eval_summary.py <runs/eval/rmbench/MODEL_TAG>")
        sys.exit(1)
    root = sys.argv[1]
    srs = {t: read_sr(os.path.join(root, t)) for t in TASKS}

    print(f"\n=== RMBench eval summary: {os.path.basename(root.rstrip('/'))} ===\n")
    print(f"  {'Task':24s} {'SR':>7s}   tier")
    print(f"  {'-' * 42}")
    for t in TASKS:
        v = srs[t]
        tier = "M(1)" if t in M1 else "M(n)"
        print(f"  {t:24s} {(v*100 if v is not None else float('nan')):6.1f}%   {tier}" if v is not None
              else f"  {t:24s} {'MISS':>7s}   {tier}")

    def avg(keys):
        vals = [srs[t] for t in keys if srs[t] is not None]
        return (np.mean(vals) * 100) if vals else float("nan"), len(vals), len(keys)

    print(f"  {'-' * 42}")
    for name, keys in [("M(1) suite", M1), ("M(n) suite", MN), ("OVERALL (macro)", TASKS)]:
        a, n, k = avg(keys)
        print(f"  {name:24s} {a:6.1f}%   ({n}/{k} tasks)")
    print()


if __name__ == "__main__":
    main()
