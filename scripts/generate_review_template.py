from pathlib import Path
import pandas as pd


def generate(path: str | Path = "outputs/review-template.xlsx") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # META sheet with placeholder review_run_id key
    meta = pd.DataFrame({"key": ["review_run_id"], "value": [""]})

    # BETS sheet header template
    bets_cols = [
        "game_id",
        "market_type",
        "selection",
        "line",
        "odds",
        "stake",
        "book",
        "opportunity_id",
        "log_status",
        "bet_id",
        "logged_at",
    ]
    bets = pd.DataFrame(columns=bets_cols)

    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="META", index=False)
        bets.to_excel(writer, sheet_name="BETS", index=False)

    return str(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/review-template.xlsx")
    args = parser.parse_args()
    out = generate(args.output)
    print(f"Wrote template -> {out}")
