"""Reads today's 4-pick betslip and outputs it as a Claude Code systemMessage."""
import json
import sys
from pathlib import Path
from datetime import datetime

CSV = Path(__file__).parent / "output" / "betslips_history.csv"
today = datetime.now().strftime("%Y-%m-%d")

try:
    import csv
    if not CSV.exists():
        sys.exit(0)

    rows = []
    with open(CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    slip = next(
        (r for r in rows if r["game_date"] == today and "4-pick" in r["slip_type"]),
        None
    )

    if not slip:
        sys.exit(0)

    picks = slip["picks"].split("|")
    payout = slip["potential_payout"]
    wager  = slip["wager"]

    lines = [
        f"TODAY'S 4-PICK POWER PLAY  ({today})",
        f"${ wager } to win ${ payout }",
        "─" * 40,
    ]
    for i, p in enumerate(picks, 1):
        lines.append(f"  {i}. {p.strip()}")
    lines.append("─" * 40)

    print(json.dumps({"systemMessage": "\n".join(lines)}))

except Exception:
    sys.exit(0)
