from pipelines.tuning import _resolve_jobs


def test_resolve_jobs_uses_cpu_minus_one(monkeypatch) -> None:
    monkeypatch.setattr("pipelines.tuning.os.cpu_count", lambda: 8)
    assert _resolve_jobs(0) == 7


def test_resolve_jobs_falls_back_to_one(monkeypatch) -> None:
    monkeypatch.setattr("pipelines.tuning.os.cpu_count", lambda: None)
    assert _resolve_jobs(0) == 1
