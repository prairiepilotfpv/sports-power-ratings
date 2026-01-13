def test_debug_load_ml(tmp_path, capsys):
    from pathlib import Path
    import json
    from ensemble.io import load_ml_weights

    p = Path('outputs') / 'ensembles' / 'NBA' / '2025-26'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'ensemble_ml_v1.json').write_text(json.dumps({'m1':2,'m2':1}))
    w = load_ml_weights('NBA','2025-26')
    print('loaded weights:', w)
    assert w is not None
