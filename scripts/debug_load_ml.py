import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
from ensemble.ml_v1 import MLWeightedAverageEnsemble
p = Path('outputs')/'ensembles'/'NBA'/'2025-26'
p.mkdir(parents=True, exist_ok=True)
(p/'ensemble_ml_v1.json').write_text(json.dumps({'m1':2,'m2':1}))
ens = MLWeightedAverageEnsemble('NBA','2025-26')
import pandas as pd
p_val, comps = ens.combine(pd.DataFrame([{'model_name':'m1','p_home_win':0.6},{'model_name':'m2','p_home_win':0.4}]))
print('p_val=',p_val)
print('comps=',comps)
