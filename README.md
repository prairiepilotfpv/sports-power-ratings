# Sports Power Ratings

Pipeline to:
1) OCR screenshots / parse SR tables → CSV
2) Normalize with OpenAI Structured Outputs
3) Run Elo(+)

Run:
1) python -m venv .pyenv
2) .\.pyenv\Scripts\Activate.ps1
3) pip install -r requirements.txt
4) python src\pipelines\ingest_game_results.py --help
