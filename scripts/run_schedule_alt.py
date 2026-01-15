import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.pipelines.schedule import build_schedule_excel_report

p = build_schedule_excel_report(
	'data/db/nba/2025-26.db',
	sport='nba',
	season='2025-26',
	output_path=Path('data/processed/nba/2025-26/schedule_with_projections.new.xlsx'),
)
print('wrote', p)
