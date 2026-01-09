from eval.validation import validate_prediction_row, get_validation_config

row = {
    'projected_home_score': 3,
    'projected_away_score': 2,
    'margin_sd': 2.0,
    'total_sd': 2.5,
    'projected_total': 5,
    'model': 'toor',
    'game_id': 'nhl|2025-10-18|FLA|BUF'
}
print('Using get_validation_config("nhl")')
config = get_validation_config('nhl')
print(config)
ok, reasons = validate_prediction_row(row, config=config)
print('ok=', ok, 'reasons=', reasons)
print('Using default config (should be NBA-like)')
ok2, reasons2 = validate_prediction_row(row)
print('ok=', ok2, 'reasons=', reasons2)
