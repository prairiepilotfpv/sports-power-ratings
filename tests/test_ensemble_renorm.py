import json
from pathlib import Path

import pandas as pd

from ensemble.ml_v1 import MLWeightedAverageEnsemble


def test_renormalize_weights_when_some_probs_missing(tmp_path):
    # Prepare weights: two equal weights
    base = Path("outputs") / "ensembles" / "TEST" / "2025"
    base.mkdir(parents=True, exist_ok=True)
    (base / "ensemble_ml_v1.json").write_text(json.dumps({"a": 1, "b": 1}))

    ens = MLWeightedAverageEnsemble("TEST", "2025")
    df = pd.DataFrame(
        [{"model_name": "a", "p_home_win": 0.7}, {"model_name": "b", "p_home_win": float('nan')}]
    )
    p, comps = ens.combine(df)
    assert p == 0.7
    comps_list = json.loads(comps)
    # a should have weight 1.0 after renormalization, b weight 0.0
    d = {c["model"]: c for c in comps_list}
    assert d["a"]["weight"] == 1.0
    assert d["b"]["weight"] == 0.0
