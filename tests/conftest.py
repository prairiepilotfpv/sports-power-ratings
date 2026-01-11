from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is on sys.path so the `src` package can be imported (tests import `src.*`).
sys.path.insert(0, str(ROOT))
# Also add the inner `src` directory so modules that import top-level names (e.g., `config`) work.
sys.path.insert(0, str(ROOT / "src"))

# Suppress a pandas FutureWarning about concatenating empty/all-NA DataFrames
# which appears in some tests that intentionally concat into an initially-empty
# sheet. This is safe to silence for now; the underlying concat behavior is
# exercised by the tests and we're not changing functional outcomes.
import warnings
warnings.filterwarnings(
	"ignore",
	message=".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
	category=FutureWarning,
)

# Provide a lightweight sklearn shim when scikit-learn isn't installed so
# calibration-related tests can run (or skip) reliably in CI without adding
# a hard dependency. The shim implements only the minimal surface area used
# by `src/calibration` (LogisticRegression.predict_proba, IsotonicRegression.transform).
try:
	import sklearn  # type: ignore
except Exception:
	import types
	import math

	sklearn = types.ModuleType("sklearn")
	linear_model = types.ModuleType("sklearn.linear_model")
	isotonic = types.ModuleType("sklearn.isotonic")

	class LogisticRegression:
		def __init__(self, C: float = 1.0, solver: str = "lbfgs"):
			self.C = C
			self.solver = solver
			self.coef_ = None
			self.intercept_ = None

		def fit(self, X, y):
			# Simple linear fit: set coef to 1.0 and intercept to 0.0 for deterministic behaviour
			self.coef_ = [1.0]
			self.intercept_ = 0.0
			return self

		def predict_proba(self, X):
			# X is expected shape (n,1)
			import numpy as _np

			probs = []
			for row in X:
				x = float(row[0])
				# logistic-like mapping for stability
				val = 1.0 / (1.0 + math.exp(- (x * self.coef_[0] + self.intercept_)))
				probs.append([1.0 - val, val])
			return _np.array(probs)

	class IsotonicRegression:
		def __init__(self, increasing: bool = True, out_of_bounds: str = "clip"):
			self.increasing = increasing
			self.out_of_bounds = out_of_bounds

		def fit(self, X, y):
			# No-op fit for shim
			return self

		def transform(self, X):
			# Identity transform clipped to [0,1]
			import numpy as _np

			arr = _np.clip(_np.array(X, dtype=float).reshape(-1), 0.0, 1.0)
			return arr

	linear_model.LogisticRegression = LogisticRegression
	isotonic.IsotonicRegression = IsotonicRegression
	sklearn.linear_model = linear_model
	sklearn.isotonic = isotonic
	import sys as _sys

	_sys.modules["sklearn"] = sklearn
	_sys.modules["sklearn.linear_model"] = linear_model
	_sys.modules["sklearn.isotonic"] = isotonic
	# Minimal joblib shim (uses pickle) when joblib isn't installed
	try:
		import joblib as _joblib  # type: ignore
	except Exception:
		import pickle as _pickle
		import types as _types

		def _dump(obj, path):
			with open(path, "wb") as f:
				_pickle.dump(obj, f)

		def _load(path):
			with open(path, "rb") as f:
				return _pickle.load(f)

		joblib = _types.ModuleType("joblib")
		joblib.dump = _dump
		joblib.load = _load
		_sys.modules["joblib"] = joblib
