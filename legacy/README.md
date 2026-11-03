# legacy/

Pre-refactor code, moved out of `src/` during the rag-etl rewrite
(refactor-plan/). Reference only: **not packaged, not importable as
`rag_etl.*` anymore, and outside every gate** (ruff, mypy, pytest).

- `courses/` — the old per-course pipeline classes (41 courses + `common/`),
  superseded by declarative specs in `src/rag_etl/projects/semester_*/`.
- `transformers/` — the old transformer packages and `base_transformer.py`,
  superseded by the fixed sequence in `src/rag_etl/transformers/`.
- `resources/` — the old `BaseResource` hierarchy, superseded by
  `src/rag_etl/core/resource.py`.
- `loaders/` — the old loaders, superseded by
  `src/rag_etl/loaders/{manifest,elasticsearch}.py`.

Nothing imports this tree; git history has the full story. Delete it once
the consumer cutover is done and a season has passed.
