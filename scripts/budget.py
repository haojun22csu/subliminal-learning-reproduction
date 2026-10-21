"""Manual CU planning aid; this is not a billing limit or runtime monitor."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def check(initial, current, rate, minutes, phase_budget, phase_spent, remaining_hours):
    values = (initial, current, rate, minutes, phase_budget, phase_spent, remaining_hours)
    if any(not math.isfinite(x) or x < 0 for x in values) or rate == 0 or minutes == 0:
        raise ValueError("Inputs must be finite nonnegative numbers; rate and duration must be positive")
    if current > initial:
        raise ValueError("Balance increased; reconcile top-ups/other sessions before planning")
    ceiling = max(0, min(160, initial - 40))
    spent = initial - current
    cost = rate * minutes / 60
    reasons = []
    if cost > ceiling - spent:
        reasons.append("Task exceeds the remaining global CU allowance")
    if cost > phase_budget - phase_spent:
        reasons.append("Task exceeds the remaining phase CU allowance")
    if current - cost < 40:
        reasons.append("Task would breach the 40 CU reserve")
    if minutes / 60 > remaining_hours:
        reasons.append("Task exceeds the remaining wall-clock allowance")
    return {"allowed_by_estimate": not reasons, "reasons": reasons,
            "initial_balance": initial, "current_balance": current,
            "spent_since_start": spent, "global_ceiling": ceiling,
            "conservative_task_minutes": minutes, "estimated_cu": cost,
            "remaining_global_allowance": max(0, ceiling - spent),
            "warning": "Manual estimate only. Check UI every 20-30 minutes and terminate runtime manually."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    gate = subs.add_parser("check")
    for name in ("initial", "current", "rate", "minutes", "phase-budget", "phase-spent", "remaining-hours"):
        gate.add_argument("--" + name, type=float, required=True)
    record = subs.add_parser("record")
    record.add_argument("--ledger", type=Path, required=True)
    record.add_argument("--phase", required=True)
    record.add_argument("--task", required=True)
    for name in ("before", "after", "minutes"):
        record.add_argument("--" + name, type=float, required=True)
    args = parser.parse_args()
    if args.action == "check":
        result = check(args.initial, args.current, args.rate, args.minutes,
                       args.phase_budget, args.phase_spent, args.remaining_hours)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["allowed_by_estimate"] else 2)
    if any(not math.isfinite(x) or x < 0 for x in (args.before, args.after, args.minutes)) or args.after > args.before:
        raise ValueError("Reconcile invalid or increased balances before recording")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    exists = args.ledger.exists()
    with args.ledger.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(["UTC", "Phase", "Task", "Balance Before", "Balance After", "CU Used", "Minutes"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), args.phase, args.task,
                         args.before, args.after, args.before - args.after, args.minutes])
    print(args.ledger)


if __name__ == "__main__":
    main()
