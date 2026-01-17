# Refactor Summary

## Module boundaries
- **ingest** (`src/ingest/`): source adapters, parsing, normalization, and schemas only.
- **models** (`src/models/`): model interfaces, implementations, and the registry.
- **data** (`src/data/`): database persistence and centralized path helpers.
- **pipelines** (`src/pipelines/`): orchestration and reporting/export workflows.
- **output/reporting** (`src/pipelines/*report*.py`, `src/pipelines/schedule.py`): CSV/XLSX writers live in pipeline modules, models only return data structures.
- **cli** (`src/cli/`): thin wrappers that call pipelines.

## How to add a new model
1. Implement the power rating interface:
   - Add a class in `src/models/` that defines `metadata()`, `fit(...)`, and `rankings()`.
   - Use `models.base.ModelMetadata` for the metadata payload.
2. Register the model in `src/models/registry.py`:
   - Add a `ModelSpec` entry to `_MODEL_SPECS` with the import path and abbreviation.
   - If the model requires an optional dependency, set `required_module`.
3. (Optional for tests) Use `register_model(...)`/`unregister_model(...)` for dynamic registration.

## How to add a new sport ingest
1. Implement a source adapter in `src/ingest/` (for example, `src/ingest/sources.py`):
   - Create a class that follows `ingest.base.IngestSource` with `load_path(...)` and `load_text(...)`.
2. Register it in `src/ingest/registry.py`.
3. The CLI pipeline will pick it up via `--source <name>`.

## How to add a new report/output format
1. Create a pipeline writer under `src/pipelines/` (for example, `src/pipelines/new_report.py`).
2. Keep model code output-agnostic; accept data structures from pipelines and serialize here.
3. Use `data.paths.processed_path_for(...)` and `pipelines.common.resolve_output_path(...)` for default output locations.
4. Wire the new report into the CLI if needed (`src/cli/pipeline.py`).
