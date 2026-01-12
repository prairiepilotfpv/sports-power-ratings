from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sqlite3

import openpyxl
import pandas as pd
import pytest

from ingest.schema import GameResult
from pipelines import schedule as schedule_pipeline
from src.data import repository as repo
from src.data import betting_repository as br
from src.pipelines import bets as bets_pipeline


def _seed_schedule_db(db_path: Path, scheduled_date: date) -> None:
    games = [
        GameResult(
            date=date(2024, 1, 1),
            home_team="Team A",
            away_team="Team B",
            home_score=100,
            away_score=90,
            sport="nba",
            season="2024-25",
            game_id="2024-01-01-team-a-team-b",
        ),
        GameResult(
            date=date(2024, 1, 2),
            home_team="Team C",
            away_team="Team A",
            home_score=95,
            away_score=110,
            sport="nba",
            season="2024-25",
            game_id="2024-01-02-team-c-team-a",
        ),
        GameResult(
            date=scheduled_date,
            home_team="Team B",
            away_team="Team C",
            home_score=None,
            away_score=None,
            sport="nba",
            season="2024-25",
            game_id="2024-01-05-team-b-team-c",
        ),
    ]
    repo.save_games(db_path, games)


def test_schedule_workbook_includes_bets_and_meta(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    repo.init_db(db_path)
    br.init_db(db_path)
    scheduled_date = date(2024, 1, 5)
    _seed_schedule_db(db_path, scheduled_date)

    workbook_path = schedule_pipeline.build_schedule_excel_report(
        db_path,
        sport="nba",
        season="2024-25",
        model="bradley-terry",
        output_path=tmp_path / "schedule.xlsx",
        as_of_date=scheduled_date,
    )

    wb = openpyxl.load_workbook(workbook_path)
    assert "BETS" in wb.sheetnames
    assert "META" in wb.sheetnames
    assert wb["META"].sheet_state == "hidden"

    dashboard_df = pd.read_excel(workbook_path, sheet_name="dashboard")
    assert list(dashboard_df.columns) == schedule_pipeline.DASHBOARD_COLUMNS

    meta_df = pd.read_excel(workbook_path, sheet_name="META")
    meta = dict(zip(meta_df["key"], meta_df["value"]))
    assert "ml_bradley-terry" in meta["review_run_id"]
    assert "spread_direct" in meta["review_run_id"]
    assert "total_direct" in meta["review_run_id"]


def test_schedule_bets_rows_and_log_bets(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    repo.init_db(db_path)
    br.init_db(db_path)
    scheduled_date = date(2024, 1, 5)
    _seed_schedule_db(db_path, scheduled_date)

    workbook_path = schedule_pipeline.build_schedule_excel_report(
        db_path,
        sport="nba",
        season="2024-25",
        model="bradley-terry",
        output_path=tmp_path / "schedule.xlsx",
        as_of_date=scheduled_date,
    )

    bets_df = pd.read_excel(workbook_path, sheet_name="BETS")
    assert len(bets_df) == 6
    assert set(bets_df["market_type"]) == {"ML", "spread", "total"}
    assert {"Over", "Under"} <= set(bets_df["selection"])

    bets_df.loc[0, "stake"] = 1
    bets_df.loc[0, "odds"] = -110
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="a") as writer:
        writer.book.remove(writer.book["BETS"])
        bets_df.to_excel(writer, sheet_name="BETS", index=False)

    inserted = bets_pipeline.log_bets(
        str(workbook_path),
        review_run_id=None,
        db_path=db_path,
        dry_run=False,
        writeback=True,
    )
    assert inserted == 1

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        assert count == 1

    updated_df = pd.read_excel(workbook_path, sheet_name="BETS")
    staked = updated_df[updated_df["stake"].notna() & (updated_df["stake"] != "")]
    assert not staked["bet_id"].isna().all()


def test_schedule_ensemble_uses_tuned_weights(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        schedule_pipeline,
        "list_models",
        lambda: ["bradley-terry", "elo"],
    )

    db_path = tmp_path / "games.db"
    repo.init_db(db_path)
    br.init_db(db_path)
    scheduled_date = date(2024, 1, 5)
    _seed_schedule_db(db_path, scheduled_date)

    weights_payload = {
        "ensemble_id": "ensemble_ml_v1",
        "market": "ML",
        "objective": "log_loss",
        "train_window": {"start": "2024-01-01", "end": "2024-01-04"},
        "models": ["bradley-terry", "elo"],
        "weights": {"bradley-terry": 0.9, "elo": 0.1},
        "created_at": "2024-01-05T00:00:00Z",
    }
    weights_path = (
        Path("outputs")
        / "ensembles"
        / "nba"
        / "2024-25"
        / "ML"
        / "ensemble_ml_v1.json"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
    from ensemble.io import load_ml_weights

    assert load_ml_weights("nba", "2024-25", "ensemble_ml_v1") == {
        "bradley-terry": 0.9,
        "elo": 0.1,
    }

    workbook_path = schedule_pipeline.build_schedule_excel_report(
        db_path,
        sport="nba",
        season="2024-25",
        model=None,
        output_path=tmp_path / "schedule.xlsx",
        as_of_date=scheduled_date,
    )

    bets_df = pd.read_excel(workbook_path, sheet_name="BETS")
    ml_rows = bets_df[bets_df["market_type"] == "ML"]
    home_row = ml_rows[ml_rows["selection"] == "Team B"].iloc[0]
    components = json.loads(home_row["ml_ensemble_components_json"])
    weights = {comp["model"]: comp["weight"] for comp in components}
    assert weights["bradley-terry"] == pytest.approx(0.9, rel=1e-6)
    assert weights["elo"] == pytest.approx(0.1, rel=1e-6)
    combined = sum(comp["prob"] * comp["weight"] for comp in components if comp["prob"] is not None)
    assert home_row["home_win_prob"] == pytest.approx(combined, rel=1e-6)
    assert "ensemble_ml_v1" in str(home_row["win_prob_source"])
    assert "ml_ensemble_components_json" in ml_rows.columns

    meta_df = pd.read_excel(workbook_path, sheet_name="META")
    meta = dict(zip(meta_df["key"], meta_df["value"]))
    assert "ml_ensemble_ml_v1" in meta["review_run_id"]
    assert "spread_ensemble_spread_v1" in meta["review_run_id"]
