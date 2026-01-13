def test_debug_candidates(tmp_path):
    from ensemble.io import _market_weight_candidates
    cand = list(_market_weight_candidates('NBA','2025-26','ensemble_ml_v1','ML'))
    print('candidates:')
    for c in cand:
        print('-', c, 'exists=', c.exists())
    assert cand
