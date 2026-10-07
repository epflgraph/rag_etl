# IMPLEMENTATION

Execution plan for the refactor described in CONSENSUS.md (design reference) and refactor_overview.svg (one-picture summary). This document turns the consensus into a phased, verifiable implementation order for promoting `rag-prepipeline` to `rag-etl`, the single end-to-end pipeline.

## 0. Plan-level decisions

Resolved after auditing both codebases (`rag-prepipeline` and `chatbotpipelines`) and settling the items the consensus left open. These bind the plan below; nothing else in the consensus was re-opened.

- **No translation.** The current pipeline translates every chunk to EN+FR via GraphAI (`content.en`/`content.fr` in the ES mapping, embedding computed on the English text). This was a legacy oddity, never mentioned in the consensus. LLMs handle the original language; translation adds processing time and potential bugs for no real improvement. The new ES mapping carries a single-language `content` field. Consumers that rely on `content.en`/`content.fr` are updated during the cutover (see Phase 5).
- **No MCP generation on the pipeline side.** The SVG's steps 7–11 (question generation, tool config generation/eval/refinement/promotion loop) belong to the MCP server, which autodiscovers and autogenerates the tools it serves from the pipeline's two artifacts (ES indices + the metadata file/record). `rag-etl` implements steps 1–6 only: extract → normalize → materialize → judge → cut → embed + load.
- **Ed Discussion ports as a source.** The extractor is fully implemented (though disabled in every course today); it is adapted to the `Resource` model and the rule mechanism and stays inactive until a project spec lists it as a source.
- **COM202 pilots the rollout.** Moodle + MOOC (heaviest realistic input), currently indexed and consumed by bots. Parallel-run it against the old pipeline, then cut over project by project into the weekly cron.
- **Scheduling: timers, no Airflow.** Airflow was never actually used in production (Ramtin's effort was never switched on). We keep what has been working: timers that trigger services. Instead of running `main` from each course, `rag-etl` gets a unified entry point (CLI with a `--project` parameter); whether the timer then loops over projects or one timer per project is an implementation detail we will settle when wiring the cron.
- **Embeddings: Qwen3-Embedding-8B on RCP, bypassing GraphAI.** Embedding moves to the RCP client (model + parameters from the project/run spec, per §4 of the consensus). This impacts the current GraphAI `/rag/retrieve` endpoint (which embeds queries on the consumer side) — addressed separately from this repo, not a blocker for the ETL.
- **Slide-change detection: keep GraphAI `detect_slides` for now.** It is the only working implementation. We may develop a better algorithm later and replace it both in the pipeline and on GraphAI; the pipeline will call it behind a small interface so the swap is local.
- **ES mapping details (dense/sparse, analyzer set, embedding dims): decided in Phase 5**, from the resource model and the field-name reconciliation with the consumer repos.
- **Moodle assignments are known-unfetchable** (no API access) — accepted limitation, documented.

## 1. Grounding (audit summary)

What the audit of the two repos found, for reference while implementing.

**`rag-prepipeline` (~16.6k lines, 154 files, `tests/` empty, no entry points, no CI):**

- One flat `BaseResource` dataclass + five thin subclasses; policy flags (`one_chunk_per_page`, `processing_method`, `model`, `is_gemini_processed_video`, `tikz`) sit in the data — exactly what CONSENSUS §2 kills.
- 41 course classes (`courses/<id>/<id>_course.py`) with `course_info` + `tag_metadata` dicts, re-derived into `*_type_subtypes` properties handed to transformers — the three steering channels CONSENSUS §4 replaces with one `where`-rule mechanism. Several are not courses (`plasma`, `cmi`, `swissunidemo`, `mgt645`, `bio695`).
- Five extractors: Moodle (token auth, tags from `[TAG]` filename markers, availability dates parsed from Moodle's restrict object), MOOC (edX export parser hierarchy, ~1.7k lines), Mediaspace/Kaltura (SRT captions, never downloads video), LocalFolder, Ed Discussion (implemented, disabled everywhere). No git extractor.
- Eight transformers; two parallel video paths (MOOC → Gemini video JSON; Mediaspace/Moodle → GraphAI `detect_slides` + RCP frame OCR + ffmpeg) and `<!-- page N -->` markers — both dissolved by the nested-resource design.
- Disk cache: content-hash keyed, scoped per transformer *and per course* (so identical pages across courses don't share), no eviction, no refresh flag; directory values copy top-level files only.
- RCP plumbing (`utils/llms.py`: OpenAI-compatible client + Langfuse tracing, thinking-tag stripping, structured output), GraphAI plumbing (`utils/graphai.py`: video retrieval + slide detection).

**`chatbotpipelines` (the repo being absorbed and discontinued):**

- ES: common settings (analyzers: `raw`, `base_en`, `base_fr`, `trigram`) + common mappings (`content.en`/`content.fr` with `raw`/`sayt`/`trigram` subfields; `embedding` dense_vector, 384 dims, dot_product, int8_hnsw); course pipe adds `description.en/fr` and `from`/`until` date fields.
- Index flow: fresh index `rag_{type}_{name}_index_{YYYY_MM_DD}` → bulk load (chunk 100–500) → alias `rag_{type}_{name}_index` shifted atomically (via `elasticsearch_interface`). This flow is the template for the new loader, minus the MySQL hop.
- Chunking (GraphAI `chunk_text`, 400/100) and bilingual translation (GraphAI) are absorbed differently: local header-aware chunking (`text` cut), translation dropped.
- Cache: MySQL `graphai_cache_rag.<type>_<name>_es_index_table` (`id_token`, `origin_token`, `json_result`, `processing_date`, `content_hash`) — dropped; `rag-etl` owns caching on disk.
- `not_core_file` skip mechanism and the `translate` flag die with the repo.
- What survives conceptually: slide OCR (already uniform in the new design: every PDF page is OCR'd), quiz parsing (inside the MOOC extractor), and the ES-writing flow.

## 2. Architecture delta

| Today | rag-etl |
|---|---|
| `BaseResource` flat dataclass + 5 subclasses, policy flags in data | One recursive `Resource` dataclass; container/child/chunk is *state*; policy lives in project rules |
| 41 course classes with `tag_metadata` dicts + `*_type_subtypes` plumbing | The judge infers type/subtype/is_solution/cut per resource; per-project **dataclass spec** carries exceptions (`where`-predicates → label corrections), extractor configs, model configs |
| Transformer list per course | One **fixed sequence**: extract → normalize → materialize → judge → cut → embed + load |
| `<!-- page N -->` markers, 2 parallel video paths, Gemini video JSON | PDF → container + page children (OCR each, RCP); video → segments + captions (GraphAI `detect_slides` + Kaltura SRT); cuts: `whole_document`, `per_child`, `per_exercise`, `text` |
| Disk cache keyed per-course-scope, no eviction, no refresh | Content-only keys, scope `(transform, content-hash)` shared across projects, per-run `--refresh` flag |
| GraphAI embed + chunk + bilingual translate | RCP embeddings (Qwen3-Embedding-8B, model+params from spec), local header-aware chunking, **no translation** |
| Record loader → `chatbot-pipelines` contract | Record/manifest loader (content-hash pointers, `dry` flag, schema snapshot) + **new ES loader** (fresh index `rag_{project}`, atomic alias shift) |
| Ed Discussion disabled, no git source, no entry points, no tests | Ed Discussion ports as source, **new git extractor** (repo/branch URL → container), CLI entry point `rag-etl run --project X [--dry] [--refresh T]`, pytest + CI |

## 3. Target layout (sketch)

```
src/rag_etl/
├── core/                     # Resource dataclass, pipeline-state helpers, where-matcher, Rule + resolution
├── projects/                 # ProjectSpec dataclass, registry, one thin spec file per project
├── pipeline.py               # the fixed sequence orchestrator
├── cache/                    # content-hash disk cache facade (refresh flag, (transform, hash) scoping)
├── llm/                      # RCP client (vision/text/structured/embeddings), judge, per-exercise annotator
├── extractors/               # moodle, mooc/, mediaspace, local_folder, ed_discussion, git (new)
├── transforms/               # normalize/, materialize/, cuts/ (cut registry)
├── embed/                    # RCP embedder, per chunk, cache-aware
├── loaders/                  # record (manifest), elasticsearch
└── cli.py                    # rag-etl run --project X [--dry] [--refresh TRANSFORM]
```

Module names are indicative; Phase 0 finalizes the shapes.

## 4. Phases

Each phase ends with a verification gate; no phase starts before the previous gate passes.

### Phase 0 — Foundations
- Add entry points + CLI skeleton (`rag-etl run --project X [--dry] [--refresh T]`).
- Set up pytest, ruff + mypy gates, GitHub Actions CI.
- Rename `courses/` → `projects/`; call labs/projects what they are.
- Short design note fixing the `Resource` and `ProjectSpec` dataclass shapes.

**Gate:** CI green; `pip install -e .` works; `rag-etl --help` responds.

### Phase 1 — Core model + exceptions
- Recursive `Resource` (derived `is_leaf` and `content_hash`, lineage child→parent, tree helpers).
- Exception mechanism: flat ANDed glob `where`-matcher; most-specific wins; equal specificity → first-declared wins, loudly (warn); applied on top of the judge's labels.
- The 41 `tag_metadata` tables are dropped, not converted: type/subtype/is_solution/cut are inferred by the judge (a stub for Javier's algorithm, which lands in Phase 4).

**Gate:** unit tests for the resource tree and the exception matcher; ambiguity warns visibly; unmatched resources stay untouched.

### Phase 2 — Extractors
- Adapt Moodle (auto-capture `from`/`until` from availability dates), MOOC, Mediaspace, LocalFolder, Ed Discussion to emit `Resource` trees.
- New git extractor: GitLab/GitHub repo or branch URL → container resource + file children, reasonable defaults, everything indexed by default.

**Gate:** unit tests; one real extraction per source (MICRO-303 Moodle first; MOOC, Mediaspace, Ed and git wait for their inputs); a temporarily unreachable source fails the run loud (SourceUnavailable) rather than indexing half of it.

### Phase 3 — Pipeline + cache + normalize + materialize
- Fixed-sequence orchestrator.
- Cache rewrite: content-only keys, `(transform, content-hash)` scope shared across projects, per-run `--refresh` flag; no GraphAI cache-DB.
- Normalize: zip/tar.gz → files, notebook → md, per-mime size limit (oversized → dropped).
- Materialize: PDF → per-page deterministic render + RCP vision OCR per page; video → slide-change detection (GraphAI `detect_slides`, behind a swappable interface) + captions (Kaltura SRT) upstream, then frame OCR; image → text via RCP.
- Unified RCP client: vision, text, structured output, embeddings — every model invocation (OCR, judge, annotator, embeddings) takes model + parameters from the project/run spec.

**Gate:** unit tests; COM202 dry run with cache reuse vs `--refresh` (per-resource cost flags correct; one-comma-in-200-pages re-OCRs one page).

### Phase 4 — Judge + cuts + embeddings
- Judge transformer (after materialize, before cut): a stub for Javier's inference, filling `type`/`subtype`/`is_solution` and the `cut` per resource (or dropping the resource); exceptions correct on top of judge output.
- `per_exercise` annotator: exercise number + is-solution (chunk-level), separate concern from the judge, shared model infrastructure; nothing annotated unless rules ask for it.
- Cuts registry: `whole_document` (size guard), `per_child`, `per_exercise` (annotate + stitch ordered sibling pages + cut at markers), `text` (header-aware chunking with size guard). New cuts join via PR, never course-local code.
- RCP embeddings per chunk (Qwen3-Embedding-8B), cache-aware (see §5 open items for cache interaction).

**Gate:** unit tests on cuts with synthetic docs; judge + annotator spot-checks on real COM202 pages; exceptions change labels on rerun.

### Phase 5 — Loaders: record + Elasticsearch
- Record/manifest loader: header (run timestamp, `dry` flag, project id, index name + alias, pipeline version, schema snapshot) + resource-tree body; metadata-only with content-hash pointers (no text, no embeddings); dry records land separately from real-run records.
- ES loader: common mapping template → per-project mapping; fresh index + atomic alias shift (mirroring the `create_and_write_to_index` flow, minus MySQL). **Decide here, with the resource model settled:** concrete mapping (single-language `content` field + analyzer set, `type`/`subtype`, `number`/`sub_number`, `week`, `is_solution`, `from`/`until`, lineage + deep links, text, embedding — nothing project-level per chunk; project id travels via the index/alias).
- **Field-name reconciliation:** inventory what the AI-tutor repos query on the old mapping (`content.en/fr`, `from/until`, filters) and reconcile with the new field names before freezing the mapping; consumer updates are part of the Phase 6 cutover.

**Gate:** ES loader against a local/test ES instance; record inspected by Aitor/Javier; Anna's dashboard needs confirmed against the new format.

### Phase 6 — COM202 pilot + cutover
- Dry run → review record → record fixes as overrides → first real run.
- Parallel-run against the old pipeline output: content spot-checks + retrieval spot-checks through the bots.
- Point consumers at the new alias (`rag_com202`); migrate remaining projects one by one into the weekly cron (timers → unified CLI entry point).

**Gate:** bots answer from the new index; old vs new retrieval comparable; cron enrolled for COM202.

### Phase 7 — Discontinue + handoff
- Archive `chatbotpipelines`.
- Document the metadata-file contract (record format + schema snapshot) as the interface the MCP server consumes for tool autodiscovery (steps 7–11, MCP repo, out of scope here).
- Kill remaining remnants: Gemini video path, `not_core_file`, `translate`, GraphAI embed usage, Google Drive utils.
- Packaging follow-ups: pip-installable parts colleagues can run individually (OCR, notebook conversion), docs.

**Gate:** nothing references chatbot-pipelines; docs updated; consumers unaffected.

## 5. Open items carried into implementation

- **Embeddings × cache:** which embedding model/parameters interplay with content-only cache keys + the `--refresh` flag (fresh embeddings land after an explicit cache refresh, per CONSENSUS §4/§6). Settle in Phase 3/4; dims flow into the Phase 5 mapping.
- **GraphAI `/rag/retrieve` impact:** consumers embed queries via GraphAI today; switching embeddings to RCP (Qwen3-Embedding-8B) breaks that pairing. Consumer-side work, tracked separately; the ETL's part is only model/parameter configurability.
- **Cron wiring:** one timer looping over all projects vs one timer per project calling the CLI. Settle when enrolling the first project (Phase 6).
- **MCP server handoff:** metadata-file contract documented by the end of Phase 5; MCP server work (tool autodiscovery, steps 7–11) tracked in its own repo.
- **MOOC extractor usage:** confirm during Phase 2 whether it is still actively used (COM202/ENV342/CS112g/CS119d/ME326/MGT645/MICRO303 reference it); do not drop it until confirmed.
