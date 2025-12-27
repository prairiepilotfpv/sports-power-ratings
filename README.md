# Sports Power Ratings

Pipeline to:
1) OCR screenshots / parse SR tables → CSV
2) Normalize with OpenAI Structured Outputs
3) Run Bradley-Terry rankings (modular)

Run:
1) python -m venv .pyenv
2) .\.pyenv\Scripts\Activate.ps1
3) pip install -r requirements.txt
4) python src\pipelines\ingest_game_results.py --help

Human-friendly pipeline (CSV in, CSV out):
1) python -m venv .pyenv
2) .\.pyenv\Scripts\Activate.ps1
3) pip install -r requirements.txt
4) python -m src.cli.pipeline input --sport nba --season 2024-25 --source sports-reference --input schedule.csv
5) python -m src.cli.pipeline run_model --sport nba --season 2024-25 --model bradley-terry --output rankings.csv

Command shortcuts:
- input (alias for import)
- run_model (alias for rank)

CSV output columns:
- team
- rating (raw Bradley-Terry strength)
- points (Bradley-Terry strength scaled to point-spread units)
- games

Paste Sports-Reference CSV text:
1) python -m src.cli.pipeline import --sport nba --season 2024-25 --source sports-reference --input-text "Date,Visitor/Neutral,PTS,Home/Neutral,PTS,,,LOG,Notes\nFri Nov 1 2024,Boston Celtics,124,Charlotte Hornets,109,Box Score,,2:17,"
