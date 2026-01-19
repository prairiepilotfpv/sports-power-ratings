from src.pipelines import guardrails
from src.backtest import runner
import inspect
print('guardrails.apply_prediction_validation signature:')
print(inspect.signature(guardrails.apply_prediction_validation))
print('\nchecking runner source for apply_prediction_validation call site:')
import inspect as insp
src = insp.getsource(runner)
print('apply_prediction_validation(' in src)
