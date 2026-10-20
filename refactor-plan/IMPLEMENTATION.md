# IMPLEMENTATION

Execution plan for the refactor described in CONSENSUS.md (design reference) and refactor_overview.svg (one-picture summary). This document turns the consensus into a phased, verifiable implementation order for promoting `rag-prepipeline` to `rag-etl`, the single end-to-end pipeline.

## 0. Plan-level decisions

Resolved after auditing both codebases (`rag-prepipeline` and `chatbotpipelines`) and settling the items the consensus left open. These bind the plan below; nothing else in the consensus was re-opened.

- **No translation.** The current pipeline translates every chunk to EN+FR via GraphAI (`content.en`/`content.fr` in the ES mapping, embedding computed on the English text). This was a legacy oddity, never mentioned in the consensus. LLMs handle the original language; translation adds processing time and potential bugs for no real improvement. The new ES mapping carries a single-language `content` field. Consumers that rely on `content.en`/`content.fr` are updated during the cutover (see Phase 5).
- **No MCP generation on the pipeline side.** The SVG's steps 7–11 (question generation, tool config generation/eval/refinement/promotion loop) belong to the MCP server, which autodiscovers and autogenerates the tools it serves from the pipeline's two artifacts (ES indices + the metadata file/record). `rag-etl` implements steps 1–6 only: extract → normalize → materialize → judge → cut → embed + load.
- **Ed Discussion ports as a source.** The extractor is fully implemented (though disabled in every course today); it is adapted to the `Resource` model and the rule mechanism and stays inactive until a project spec lists it as a source. MICRO-303 activates it (Javier's merge added the LLM-as-jury cascade and the catalogue built from the previous run's output metadata).
- **MICRO-303 pilots the rollout** (chosen over COM202 as the more recent course with a MOOC): Moodle 19283 + MOOC export + local folder + Ed Discussion, all reachable. Parallel-run it against the old pipeline, then cut over project by project into the weekly cron.
- **Scheduling: timers, no Airflow.** Airflow was never actually used in production (Ramtin's effort was never switched on). We keep what has been working: timers that trigger services. Instead of running `main` from each course, `rag-etl` gets a unified entry point (CLI with a `--project` parameter); whether the timer then loops over projects or one timer per project is an implementation detail we will settle when wiring the cron.
- **Models: a settled catalog, not per-project config.** Embeddings are Qwen3-Embedding-8B on RCP, bypassing GraphAI. Every model invocation goes through a `RCPClient` whose model + parameters are baked in — `ragtools/llm/defaults.py` holds a catalog (`MODELS`, keyed `(model, mode)`, each entry a client built from the `.env` endpoint), and each capability pins its concern's model internally via `get_model(task)` (ocr, judge, split_exercises, stitch, embedding). Per-project model configs and per-call overrides were dropped: the pairings are settled by us, not data; a model change is a one-line edit in the catalog (or a `--refresh TRANSFORM` for the cache). This still impacts the current GraphAI `/rag/retrieve` endpoint (which embeds queries on the consumer side) — addressed separately from this repo, not a blocker for the ETL.
- **Slide-change detection: keep GraphAI `detect_slides` for now.** It is the only working implementation. We may develop a better algorithm later and replace it both in the pipeline and on GraphAI; the pipeline calls it behind a small interface so the swap is local.
- **ES mapping: decided (Phase 5).** One frozen 13-field document schema, the create-index body shipped verbatim as package data (`index_settings.json`); details in Phase 5.
- **Moodle assignments are known-unfetchable** (no API access) — accepted limitation, documented.
- **Business logic and plumbing split: `ragtools`.** The file-prep capabilities (OCR, conversions, archives, video handling, document splitting/stitching) and the unified RCP client + model catalog live in their own top-level package `src/ragtools/` that never imports `rag_etl` (enforced by `tests/test_layering.py`); the pipeline packages (`rag_etl`) depend on it, never the other way. The GraphAI and Kaltura plumbing joined it too (they serve document processing, not the pipeline). Colleagues pip install the distribution and use `ragtools` for OCR, splitting or their own file prep; a true standalone distribution splits off in Phase 7 if demand appears. Prompts are business logic and are kept verbatim.
- **Model calls: standard retry, then the run stops loud.** Every LLM call goes through the unified `RCPClient`, retried a few times by the underlying client and then raised — a page that cannot be read stops the run rather than leaving a silent gap.
- **Cache semantics: content-only keys, no eviction.** The cache is keyed by content hash only (the model is not part of the key; a model change is handled manually with `--refresh TRANSFORM`), scoped `(transform, content-hash)` and shared across projects, with full output directories as values.
- **Videos: one uniform Kaltura path.** Mediaspace and MOOC videos both go through the Kaltura slides flow (entry id from the video descriptor or the playManifest url); YouTube videos, with no Kaltura entry, keep only their captions. The Gemini video path is gone.
- **Stitching is kept, only where files are actually merged** (see Phase 4): not for slides — each slide is its own chunk — but for polycopiés and exercises; deterministic-first, LLM only as fallback.
- **The real run waits until all the code is ready.** The Phase 2/3 real-extraction gates are deferred: the MICRO303 directory structure and the MOOC export are not on disk yet. Unit tests with synthetic fixtures and stubbed models carry the verification until then.

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
| 41 course classes with `tag_metadata` dicts + `*_type_subtypes` plumbing | The judge infers type/subtype/is_solution/cut per resource; per-project **dataclass spec** carries exceptions (`where`-predicates → label corrections) and extractor configs |
| Transformer list per course | One **fixed sequence**: extract → normalize → materialize → judge → cut → embed + load |
| `<!-- page N -->` markers, 2 parallel video paths, Gemini video JSON | PDF → container + page children (OCR each, RCP); video → segments + captions (GraphAI `detect_slides` + Kaltura SRT); cuts: `whole_document`, `per_child`, `per_exercise`, `text` |
| Disk cache keyed per-course-scope, no eviction, no refresh | Content-only keys, scope `(transform, content-hash)` shared across projects, per-run `--refresh` flag |
| GraphAI embed + chunk + bilingual translate | RCP embeddings (Qwen3-Embedding-8B, pinned in the catalog), local header-aware chunking, **no translation** |
| Record loader → `chatbot-pipelines` contract | `ManifestLoader` (one JSON manifest per run: header + nested summary + metadata-only tree with content-hash pointers; `{timestamp}_manifest[_dry].json` in `{dir}/output/`) + **new ES loader** (fresh date-named index, bulk, one atomic alias shift) |
| Ed Discussion disabled, no git source, no entry points, no tests | Ed Discussion ports as source, **new git extractor** (repo/branch URL → container), CLI entry point `rag-etl run --project X [--dry] [--refresh T]`, pytest + CI |

## 3. Target layout (sketch)

```
src/
├── ragtools/                    # pure file-prep library; never imports rag_etl (tests/test_layering.py)
│   ├── llm/                     # RCP client (rcp.py) + model catalog (defaults.py: MODELS keyed (model, mode),
│   │                            #   get_model(task) for the concerns: ocr, judge, split_exercises, stitch, embedding)
│   ├── config.py                # .env + required() — the one place secrets are read in ragtools
│   ├── archives.py              # zip/tar.gz → files (NFC, UTF-8 fixes)
│   ├── documents/               # heading-aware splitting, page stitching, exercise splitting
│   │                            #   (prompts/ shipped as package data)
│   ├── encoding.py              # UTF-8 fixing, archive name decoding, filename sanitisation
│   ├── graphai.py               # GraphAI plumbing: video retrieval + slide detection
│   ├── images.py                # image → text (slide prompts) / alt text, by a vision model
│   ├── kaltura.py               # Kaltura client: entry metadata + slide flow
│   ├── mime_types.py            # mime constants and guessing
│   ├── notebooks.py             # ipynb → md (alt texts via a vision model)
│   ├── pdf.py                   # deterministic PyMuPDF page render (prompts kept here too)
│   └── video.py                 # ffmpeg frames, SRT parsing/merging, SlideDetector interface
└── rag_etl/
    ├── core/                    # resource.py (Resource + hashing), rules.py (Rule + matcher)
    ├── extractors/              # base_extractor.py (Extractor, SourceUnavailable) + moodle/, mooc/,
    │                            #   mediaspace/, local_folder/, ed_discussion/, git/
    ├── transformers/            # base.py (Transformer ABC); normalize/, materialize/ (thin adapters
    │                            #   over ragtools, no clients of their own); cut/, embed/, judge/;
    │                            #   legacy subpackages kept, mypy-suppressed, die at cutover
    ├── cache/                   # content-hash disk cache (refresh flag, (transform, hash) scoping)
    ├── pipeline.py              # the fixed sequence orchestrator
    ├── projects/                # spec.py (ProjectSpec: id, language, exceptions, sources, dir — no models) +
    │                            #   from_code discovery; semester_2627_autumn/ with one thin spec file
    │                            #   per project; legacy course class folders kept, mypy-suppressed, die at cutover
    ├── config.py                # .env CONFIG + shared config_value (required values, fail loud)
    ├── loaders/                 # manifest.py (ManifestLoader) + elasticsearch.py (ElasticsearchLoader,
    │                            #   index_settings.json shipped verbatim); legacy loaders kept,
    │                            #   mypy-suppressed, die at cutover
    ├── utils/                   # legacy but alive: llms.py (Ed's cascade), tags.py; dead: cache.py
    ├── resources/               # legacy BaseResource subclasses; mypy-suppressed, die at cutover
    └── cli.py                   # rag-etl run --project X [--dry] [--refresh TRANSFORM]
```

Specs are discovered by convention (`from_code` in `projects/__init__.py` scans the `semester_*` folders — e.g. `projects/semester_2627_autumn/micro303test.py`, MICRO-303 running under the test id until cutover), not by a registry.

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

### Phase 3 — Pipeline + cache + normalize + materialize *(done, real run deferred)*
- Fixed-sequence orchestrator (`pipeline.py`), fail-loud.
- Cache rewrite (`src/rag_etl/cache/`): content-only keys, `(transform, content-hash)` scope shared across projects, per-run `--refresh` flag; full output directories as values; no GraphAI cache-DB.
- Normalize (`transformers/normalize/`): zip/tar.gz → files, notebook → md, per-mime size limits (PDF 50 MB, images 10 MB, text 5 MB; oversized dropped).
- Materialize (`transformers/materialize/`): PDF → per-page deterministic render (PyMuPDF, ~72 DPI) + RCP vision OCR per page (page children); video → one uniform Kaltura path (entry id from descriptor or playManifest url; slide changes via GraphAI `detect_slides` behind the swappable `SlideDetector`; ffmpeg frames; captions quoted into slides); image → text via RCP.
- Unified RCP client (`ragtools/llm/`): vision, text, structured output, embeddings — every capability pins its concern's model internally via `get_model` (model + parameters baked in the catalog, endpoint from the `.env`); standard retry, then the run stops loud. No `llm` fields on the transformers, no clients built by the pipeline.
- **Business logic split into `ragtools`** (see §0): the colleague-facing library; transformers are thin adapters.
- Project spec final shape (`ProjectSpec`: exceptions + extractor configs, no models) + specs in semester folders, discovered by convention (`from_code`); CLI switched to spec + orchestrator; MICRO-303 the first real spec.

**Gate:** unit tests (68, incl. one-comma-in-200-pages re-OCRs one page — with stubbed models); the real dry run (MICRO-303) **waits until all the code is ready** (§0): the MICRO303 directory structure and the MOOC export are not on disk yet.

### Phase 4 — Judge + cuts + embeddings *(done, real spot-checks deferred with the real run)*
- Judge transformer (after materialize, before cut): a deterministic stub for Javier's inference — it applies the spec's exceptions and fills the default cuts (readable leaves cut whole, containers of text cut per child, roots never cut), leaving `type`/`subtype`/`is_solution` unlabelled until his inference replaces it.
- Cuts registry (`transformers/cut/cut_transformer.py`, one `CUTS` dict): `whole_document` (stitch multi-page children, size guard drops over-limit chunks loudly), `per_child` (readable children chunk in place; children with their own different cut keep it), `per_exercise` (stitch + cached annotation + cut at markers, deep links from page markers), `text` (header-aware chunking with the guard). Unknown labels fail loud. A cut consumes the node it applies to, top-down.
- `per_exercise` annotator (`ragtools/documents/exercises.py`, `split_exercises`): exercise number + is-solution (chunk-level), separate concern from the judge; nothing split unless the cut asks for it. The old prompt and call name kept verbatim, migrated to documents (`ragtools/documents/prompts/*.md`, shipped as package data).
- **Stitching landed here, at cut time** (see DESIGN.md): deterministic-first (page markers, split tables/equations joined, hyphenation heuristic — zero LLM calls; heading normalization left to the fallback); the old 10-page-batch LLM stitching is replaced, LLM only as a per-document fallback behind the same `stitch` interface, off by default.
- RCP embeddings per chunk (Qwen/Qwen3-Embedding-8B), batched, cache-aware: one JSON file per chunk under `{CACHE_DIR}/embed/{text hash}/` via new `Cache.read`/`write`; vectors annotate the chunk nodes (`Resource.embedding`).
- `ModelConfig(s)` dropped entirely (they existed to make models configurable per project): the model pairings are settled in the `ragtools.llm.defaults` catalog, the cuts call `ragtools.documents` capabilities that pin their own model — no circular import to fix, because no cut needs a model.
- `run_pipeline` returns nothing content-wise: embed is the fifth transform (it annotates the chunks); Phase 5 loaders are called inside `run_pipeline` and publish the index.

**Gate:** unit tests on the cuts with synthetic docs (98 total, annotator stubbed); judge + annotator spot-checks on real pages **wait for the real run** (§0); exceptions change labels on rerun (judge test).

### Phase 5 — Loaders: manifest + Elasticsearch *(code done; real ES gate pending)*
- **Loaders take trees.** The pipeline contract is uniform: extractors produce trees, transformers take and return trees, loaders take trees (`load(roots)`) and are configured by constructor args — no intermediate run object.
- **Embed joined the sequence** as the fifth transform (`EmbedTransformer`): it annotates every chunk with its vector in place (`Resource.embedding`), same batching and content-hash cache as settled in Phase 4.
- **`ManifestLoader`** — one JSON manifest per run, written for human inspection; nothing reads it. Header (project, dry, timestamp, index, alias, pipeline version); summary (sources, chunks, without_url + a nested drill-down counting chunks/without_url/is_solution per source → type → subtype → number, `unknown` buckets, None last); tree (metadata-only, no text, no embeddings: content-hash pointers, `chunk` flag, the `cut` decision). All manifests land in `{dir}/output/` as `{timestamp}_manifest[_dry].json` — name-sorted, dry distinguished by suffix.
- **`ElasticsearchLoader`** — doc ids `{epoch_ms}-{content_hash}` with within-run content dedupe (first occurrence wins, warned); lineage breadcrumb per node (title → url → `"unknown"`, root→leaf); url always present (own → nearest ancestor's → `"<redacted>"`, redaction is pipeline policy, e.g. Moodle hidden resources); flow: recreate the date-named index (delete + create on same-day collision, Ramtin's `recreate_index` behavior) → bulk → refresh → **one atomic** `update_aliases` (remove old + add new; add-only on first run). A failed run never touches the alias.
- **Frozen document schema (13 fields)** — `index_settings.json` shipped verbatim as package data, `content.analyzer` patched per project language at creation: `id`, `content` (text, `base_en`/`base_fr`), `embedding` (dense_vector 4096, dot_product, int8_hnsw), `lineage` (keyword array), `url`, `type`, `subtype`, `number`, `subnumber` (keyword), `week` (integer), `is_solution` (boolean), `from`/`until` (date, ES default ISO). Dropped vs the old mapping: `content.en/fr` (no translation), `description.en/fr`, `raw`/`sayt`/`trigram` subfields + their analyzers and filters, synonyms, `dd/MM/yyyy` formats, 384 dims.
- **Connection**: flat env keys `ES_HOST`/`ES_PORT`/`ES_USERNAME`/`ES_PASSWORD`/`ES_CAFILE`, proper TLS verification via `ca_certs` (the `verify_certs=False` shortcut is gone); `elasticsearch>=8,<9` dependency.
- **`ProjectSpec.dir`**: the project's material folder; manifests land in `dir/output` (fallback `BASE_PATH/{id}`). No `RECORDS_DIR` env var.
- **Field-name reconciliation** (with Aitor): search stays **hybrid** (embeddings + Elasticsearch text); consumers filter on `from`/`until` (automatic), `type`/`subtype`/`is_solution`; returned fields and index routing are the MCP's concern. Carried into Phase 6: GraphAI's `/rag/retrieve` must embed queries with Qwen3-Embedding-8B (4096 dims) and query `content`.
- **Gate shakedown fixes**: RCP endpoint/key resolve from the environment at call time (the model catalog is built at import, which could precede the `.env` load); the Moodle extractor logs per file with an end summary and takes a `sections` limiter for tests; Langfuse media-upload errors (non-fatal observability noise) are filtered from the logs.

**Gate:** dry manifest inspected (Aitor) on the `micro303test` project (Moodle-only, first section); real ES write against the shared instance under the `rag_micro303test` alias **pending**; Anna's dashboard needs confirmed against the new format.

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

- **Embeddings × cache:** settled (§0): content-only keys, no eviction, model changes handled manually with `--refresh TRANSFORM`. Dims decided with the Phase 5 mapping: 4096 (Qwen3-Embedding-8B).
- **GraphAI `/rag/retrieve` impact:** consumers embed queries via GraphAI today; switching embeddings to RCP (Qwen3-Embedding-8B, 4096 dims) breaks that pairing, and the text half must query `content` (the old `content.en/fr` are gone). Consumer-side work, tracked separately; the ETL side is done.
- **Cron wiring:** one timer looping over all projects vs one timer per project calling the CLI. Settle when enrolling the first project (Phase 6).
- **MCP server handoff:** metadata-file contract documented by the end of Phase 5; MCP server work (tool autodiscovery, steps 7–11) tracked in its own repo.
- **MOOC extractor usage:** confirmed (Phase 2) — actively used, kept and adapted; MOOC videos join the uniform Kaltura path.
- **Stitching details:** which documents stitch and where deterministic fixes are enough — settle during Phase 4 on real polycopiés; the design direction is in DESIGN.md.
