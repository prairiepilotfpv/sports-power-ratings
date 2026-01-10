from src.utils.review_helpers import populate_game_ids
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    n = populate_game_ids(args.workbook, args.db)
    print(f"Populated {n} game_id(s)")


if __name__ == "__main__":
    main()
