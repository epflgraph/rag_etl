# CONSENSUS

Reference document consolidating what we agreed (and what we didn't yet) while redesigning the RAG pipeline. See CONTEXT.md for background and THOUGHTS.md for the raw brainstorm. Nothing here binds implementation details; it is the shared reference for the design and the future implementation.

Status markers:
- **Decided** — agreed consensus.
- **Leaning** — direction we agreed in spirit but did not explicitly ratify.
- **Open** — discussed, no consensus yet.

## 1. Goal

- **Decided.** Promote `rag-prepipeline` to the single end-to-end pipeline (`rag-etl`): it extracts, transforms, and loads straight into the Elasticsearch RAG database, absorbing what survives of `chatbot-pipelines`, which is discontinued afterwards. The intermediate metadata-file contract between the two disappears — the file itself survives, reborn in an adapted format as the pipeline's metadata output (§7).
- **Decided.** `rag-etl` serves RAG for courses (main use case) and any other project (`lex`, `servicedesk`, `plasma`, `cmi`, debate partner, explique...); it is agnostic of course/admin/other RAGs.
- **Decided.** Everything remains in-house: RCP for models, files on disk for communication, no external services for data.
- **Decided.** Not feature parity: functionalities may change, improve, or be dropped as long as the services consuming the RAG database keep working.
- **Decided** (early in the discussion, before the detailed rounds): project definitions are Python dataclass specs.

## 2. Resource model

- **Decided.** One recursive `Resource` class that can reference other resources: containers (PDF, zip, video, notebook) expand into children (pages, slides, files). Uniform treatment across all levels; no pre-decided nesting depth; no per-level classes.
- **Decided.** No `kind` discriminator field. "Container vs child vs chunk" is pipeline *state*, auto-discoverable (still has raw bytes? has text yet? has children?). Identity of what a thing *is* is carried by the existing fields (`type`/`subtype`, `mime_type`).
- **Decided.** Resources carry data about themselves (`mime_type`, `type`/`subtype`, locators, names, numbers, dates); **policy about what to do with them lives in project config**. The boolean processing flags (`one_chunk_per_page`, `processing_method`, `is_gemini_processed_video`, `model`, `tikz`, ...) die — they were policy leaking into data.
- **Decided.** The extractor keeps all names as data (resource title, file name, folder/lineage); no canonical-name collapse.
- **Leaning.** Derived nodes inherit `type`/`subtype` from their container at expansion time; an expanding step may refine them.
- **Decided.** Lineage is kept for every child (child → parent): needed for deep links and document-level filtering.
- **Decided.** Nesting is the only structural relation between resources: no arbitrary links between separate resource trees (Javier confirmed — not a priority, likely unnecessary complexity). Link-like behavior (video ↔ its slide deck, publication ↔ lab session) is not modeled; if ever needed, it would ride on matchable metadata joined at query time by consumers.

## 3. Labels, tags, and release windows

- **Decided.** `type`/`subtype` stay free-form strings anchored on a small documented "recommended set" — no closed enum, no validation that bites a new course. Deviations are sometimes necessary.
- **Decided.** The scenario fork is resolved: **(B) automated metadata** is the way forward — Javier's tag-automation experiment works. `type`/`subtype`/`is_solution` come from the LLM-as-a-judge; the professor role shrinks to an **exclude-only** signal: the professor marks what to exclude in Moodle, everything else is judged. Scenario A (tags remain) was the alternative until the experiment landed and is dropped.
- **Decided.** One manual override channel (the project spec) sits on top of the judge's default labels. No per-label provenance and no jury-consensus flags are tracked — the reviewer trusts the values and reacts via override + rerun.
- **Decided.** HITL lives between runs: a new project's first run is manual (dry run → fixes recorded as overrides → enroll in the weekly cron); weekly runs are fully automatic — the reviewer checks the record and, when something is off, adds an override and reruns that project.
- **Decided.** The index loads exactly the selected material, and nothing else: everything the judge labels minus the professor's exclusions. Changes to the selection take effect on the weekly run after they happen.
- **Decided.** Release windows (`from`/`until`) are captured automatically from Moodle availability dates by the extractor — reliable source-side data, not a professor labeling burden. Every index document carries them as plain fields; they are never a load filter — consumers (e.g. the MCP server) filter in-window at query time, so all selected material is pre-indexed and becomes available the second its window opens.

## 4. Rules, overrides, and project definitions

- **Decided.** Projects never list transformers. The pipeline is one fixed sequence; projects declare *data*: exceptions that correct the judge's labels for specific resources, matched against resource attributes via `where`-predicates.
- **Decided.** There are no tags anywhere: extractors return plain includable documents (Moodle: everything not marked NO_BOT), and the judge infers the rest. The only rule table is the exception table (staff corrections).
- **Decided.** `where`-matching discipline: flat ANDed globs over fields the resource actually carries (title, filename, path, mime, ...); no nested logic. (E.g. two PDFs inside one zip each match their own rules — different cuts for exercises and theory.) A nested condition is the signal that a case wants custom code, not a bigger config language.
- **Decided.** Priority: **the human override wins**; everything not overridden comes from the judge (the default channel). "Most specific wins" applies only *within* the rule table; if two staff rules match the same resource sufficiently ambiguously, it warns rather than resolving silently.
- **Decided.** Overrides are deliberate fixes written when a particularity arises (cost of a rare quirk ≈ two lines); they live in the project spec and are reviewed in the repo. They are not maintained forever: they are pruned whenever a review shows one has become redundant.
- **Decided.** The judge fills type/subtype/is_solution/cut for every resource — or drops the resource. There is **no type/subtype → cut mapping in the codebase**: how the judge decides (a mapping, an LLM pass per resource, ...) is Javier's business.
- **Decided.** Every model invocation is configurable — not only embeddings: the OCR vision LLM, the metadata judge, the `per_exercise` annotator each take their model *and* inference parameters (temperature, top_p, thinking enabled/disabled, ...) from the project/run spec, never baked into code. Since cache keys ignore the model and its parameters (section 6), new settings take effect after an explicit cache refresh (section 6).

## 5. Pipeline shape and cuts

- **Decided.** One fixed sequence shared by every project:

  extract → normalize (zip, tar.gz → files; notebook→md) → materialize (PDF/video/image → clean text) → cut (chunk) → embed + load

  Ordering is inherent to the structure (no analyzer-like ordering DSL). No step-level escape hatch.
- **Decided.** The normalize step also filters large files: the size limit is per file type (mime type), and oversized files are dropped rather than passed on to materialize (the same reasonable default the git extractor applies).
- **Decided.** The metadata judge is a transformer in the fixed sequence: it runs after normalize and materialize — it labels the produced text rather than a hand-prepared file — and before cut.
- **Decided.** A git extractor (gitlab+github) joins as a new source: a repo or branch URL becomes a container resource whose files are child resources. Reasonable defaults apply (e.g. oversized files ignored), and everything in the repo/branch is indexed by default, at least for now. No existing project uses it yet.
- **Decided.** New kinds of cuts join the shared cut catalogue — a PR to `rag-etl` — never course-local code. The v0 set: `whole_document` (with a size guard; covers a code file), `per_child` (one chunk per child — pages and video segments alike), `per_exercise`, `text` (header-aware chunking for typeset notes).
- **Decided.** Materialization is uniform: every PDF → container + one child per page, each page OCR'd via the RCP vision LLM (page-by-page is the only way to OCR via a vision model, and it happens with all PDFs). The `<!-- page N -->` marker mechanism is dropped: it was an artifact of the file-based pipeline; page geometry is native in the resource model.
- **Decided.** OCR stays pure and reusable: transcription only, nothing else folded into it. No local text-layer fast path. Rendering must be deterministic (stable rasterization settings), because a page's rendered input is what gets hashed.
- **Decided.** `per_exercise` is annotation + cut: an LLM pass first annotates each exercise's number and whether it holds a statement or a solution; then the ordered sibling pages are concatenated; then the text is cut at the annotated markers. Nothing is annotated unless the cut asks for it.
- **Decided.** `is_solution` is chunk-level: a document split by a cut is itself neither solution nor statement — its children are, labeled by the `per_exercise` annotator. (The `per_exercise` annotator and the metadata judge share model infrastructure but are separate concerns.)
- **Decided.** The combinatorial cases (statement-only / solution-only / both documents / statements-and-solutions siblings / stitch or not) decompose into: detect + pick one of the cuts + derive labels. Sibling PDFs pair via the shared exercise number; whether inline solutions become one chunk or two is a cut *parameter*, not a new path.
- **Decided.** Videos: slide-change detection and captions are resolved once upstream in materialization; the judge picks the cut. (Which upstream service does detection is an open topic.)

## 6. Identity and caching

- **Decided.** Elasticsearch is disposable state: every run builds a fresh index and moves the alias. Identity's only customer is the cache.
- **Decided.** The cache works at every level of the tree, keyed by content hash (as today): a container with unchanged bytes reuses its whole cached subtree; a changed container inspects its children — unchanged pages reused, new ones processed. One comma in a 200-page PDF re-OCRs one page.
- **Decided.** Cache keys are content-only: no prompt/model/config hashed in. Two runs with the same prompt and model can differ anyway, so keying by them buys no reproducibility; eviction will most likely be by date; manual cache edits (out-of-band fixes) are acceptable.
- **Decided.** Explicit invalidation replaces key-based invalidation: a per-run refresh flag ("re-run a given transform ignoring its cache") so prompt/model/config changes invalidate deliberately rather than serving stale output silently.
- **Decided.** Locators/URLs are properties, never identity: refreshed each run, so Moodle URL churn and re-uploads change properties, not what a thing is. Deep links stay fresh (each page child carries a `#page=N` link built from the URL of the day).
- **Leaning.** Cache scoping keyed by (transform, node content hash); course-level scoping is not strictly required (identical pages across projects legitimately share cached transcriptions). Details open at implementation.
- **Leaning.** Caching is owned by `rag-etl` on disk; the GraphAI cache-DB connector dependency is dropped.
- **Decided.** Chunk/document ids *in Elasticsearch* are an implementation detail since the whole index is rebuilt: no stable-id machinery, no chain hashes, no diff-based deletion needed.

## 7. Loaders and run outputs

- **Decided.** Two loaders per run: (1) the revised content+metadata record loader (always on) and (2) the new Elasticsearch loader.
- **Decided.** The record documents the run for manual inspection and for the analyses the team already performs; the bots never see it. Its format is free (no longer the `chatbot-pipelines` contract) and should be a faithful dump of the new resource model: lineage, locators, labels, content-hash pointers, chunk counts, and cost flags.
- **Decided.** The record is metadata-only with **content-hash pointers** instead of duplicated content: analyses that need actual text resolve pointer → cache folder (one small lookup); Kibana covers indexed text. Embeddings are skipped in the record.
- **Decided.** The dry run *is* the plan: it skips Elasticsearch loading, uses cached results as-is, mocks the uncached costly operations (LLM calls, OCR of new material), and produces the same record flagged `dry`. A flag per resource (reused-from-cache / freshly-computed / would-compute) makes it a cold-start cost estimator for free. Dry records land separately from the record folder, so consumers of the record folder need no filtering; the dry path doubles as the pending-review inbox for first-run enrollment.
- **Decided.** The record doubles as the **manifest** onto which derived things are built (section 9). Header: run timestamp, `dry` flag, project id, index name + alias, pipeline version, and a snapshot of the schema (which filter dimensions exist for this project — release windows, which `type`/`subtype` words it uses). Body: the resource tree.

## 8. Index strategy and project metadata

- **Decided.** One index per project (`rag_{project}`), instantiated from a common mapping template, alias atomically shifted per run. A weekly run for one course never touches other projects' indices; stale-document handling is trivial (whole index replaced). The course chatbot and the forum bot point at the same project alias.
- **Decided.** Project-level metadata lives in the project definition; nothing is denormalized into chunk documents — no course title, year, or language per chunk. The record loader outputs the full project section into the metadata file (the single shared artifact, section 9), where consumers (the MCP server for its config and overview, Anna's dashboard, the team) read it. The project id travels via the index/alias (`rag_{project}`), not per-chunk.
- **Decided.** `week` stays in the per-chunk field set: Javier uses it for the MOOCs (it had been dropped as unused in retrieval).
- **Open.** The concrete Elasticsearch mapping (text fields, filter fields, embeddings config — dense/sparse, model choice). The per-chunk field set falls out of the resource model: `type`/`subtype`, `number`/`subnumber`, `week`, `is_solution`, `from`/`until`, lineage + deep links, text, embeddings — nothing project-level; project id and course facts travel via the index/alias and the metadata file.

## 9. MCP server and overview tool

- **Decided.** The pipeline produces exactly two artifacts per run — the Elasticsearch index and the new-format metadata file (the record/manifest, section 7) — and generates no MCP-anything.
- **Decided.** The metadata file serves every non-bot consumer: Aitor + Javier for manual inspection and debugging; Anna's dashboard (her analyses) via direct file access on the server; and the MCP server. The record folder holds only real runs; a project not yet run is simply absent from the config until its first real run.
- **Decided.** The MCP server consumes the metadata file itself: it generates whatever config it needs from it, and its overview tool derives from it a high-level structured summary of what's in the index (e.g. 10 theory slide PDFs numbered 1-10, 5 unnumbered extra theory material, 6 exercise series 1-6 with their 6 solutions, release dates throughout). No overview summary is generated on the pipeline side.
- **Decided.** No HTTP is needed for file access — everybody reads files on the same server.
- **Open.** Deployment/hosting of the MCP server; how it notices new metadata files (poll/watch vs. a scheduling-layer nudge); reload-on-config-change. Tool surface: retrieval over the index + the overview summary.

## 10. Rejected directions (so we don't re-litigate)

- **Locator/URL-based identity** — sources churn constantly (Moodle URL edits on metadata change, re-uploaded videos); identity is content, locators are properties.
- **Page-marked text as pipeline geometry** — replaced by nested per-page child resources; markers were the artifact that kept page geometry fake.
- **Folding exercise/section detection into the OCR call** — OCR stays clean and reusable; detection is a later, rule-gated pass.
- **Per-course transformer lists and a step-level escape hatch** — shared cuts; growth happens in the catalogue via PR, not course-local code.
- **Single-LLM auto-labeling (the ~80% experiment) as the labeling mechanism** — superseded by the two-scenario model; the LLM-as-a-judge exists only in scenario B, wrapped by overrides and HITL.
- **A dedicated inspection CLI command** — Kibana plus the always-on record cover inspection; no new tooling.
- **prompt/model/config in the cache key** — content-only keys plus explicit invalidation instead.
- **Asynchronous/webhook refresh or preemptive Moodle parsing** — the pipeline runs weekly, nothing more.
- **Per-label provenance and consensus flags in the record (dry runs included)** — inferable from the scenario in force; the reviewer trusts the values and reacts via the override list + rerun.
- **Arbitrary links between resource trees** — not a priority and likely unnecessary complexity (Javier); nesting is the only structural relation (§2), anything link-like rides on metadata joins.

## 11. Open topics

- **GraphAI.** What survives its responsibilities now that OCR goes to RCP: subtitle extraction, slide-change detection, captions; its cache-DB schema and connector are being dropped. Candidate for its remaining role: things RCP cannot do. *Direction settled: OCR goes to RCP through the unified client; `detect_slides` stays behind a swappable interface.*
- **Source side in detail.** What extractors owe the new resource model; a temporarily unreachable source fails the run loud (no half-indexing); the MOOC extractor (the largest) is confirmed actively used and was adapted, not dropped. Moodle assignments are known-unfetchable (no API access) — accepted limitation. *Settled in Phases 2–3.*
- **Embeddings.** Which model, dense/sparse mapping details, and how embedding caching interacts with the content-only cache key plus the refresh flag. The retrieval-quality/latency experiments with newer RCP embedding models (e.g. Qwen3-Embedding-8B) are consumer-side work that will especially drive the MCP server; the ETL's part is only model/parameter configurability (section 4). *Cache interaction settled: content-only keys, model changes handled with `--refresh`.*
- **Reranking (consumer-side).** Query-time rerankers (e.g. Qwen3-Reranker-8B, bge-reranker-v2-m3) are outside `rag-etl` scope; whether the pipeline owes anything for them is undecided — probably nothing beyond what the metadata file already exposes.
- **Packaging and best practices.** Pip-installable parts colleagues can run individually (OCR, notebook conversion), tests (directory currently empty), CI, docs; module/directory renames (`courses/` currently holds non-courses such as `plasma`, and course files are named `*_course.py`). *Partially settled: the file-prep logic is its own package (`ragtools`); tests exist.*
- **ES schema.** The concrete mapping, to be written from the resource model once the dataclass shapes settle.

## 12. Settled during implementation

Decisions taken while implementing, binding the phases (details in DESIGN.md and IMPLEMENTATION.md §0):

- **Business logic vs plumbing: two top-level packages.** Pure file preparation lives in `ragtools` (archives, notebooks, PDFs, images, videos, the unified RCP client and its parameter defaults) and never imports `rag_etl`; the pipeline (`rag_etl`) depends on it. Colleagues use `ragtools` for OCR or their own file prep; a standalone distribution splits off in Phase 7 if demand appears. Prompts are business logic, kept verbatim.
- **Transformers follow an explicit-inheritance contract** (`Transformer`, frozen dataclass, like `Extractor`), reshaping resource trees in place: normalize (archives, notebooks, size limits) and materialize (per-page PDF OCR, image OCR, one uniform Kaltura video path with captions quoted into slides).
- **Model calls: standard retry, then the run stops loud.** One unified `RCPClient`; the model travels with each call from the spec; recommended parameters per model in one defaults file.
- **Cache: content-only keys, no eviction.** Scoped `(transform, content-hash)`, shared across projects; `--refresh TRANSFORM` handles model changes manually.
- **Size limits (normalize):** PDF 50 MB, images 10 MB, text 5 MB, as module constants.
- **Videos: one uniform Kaltura path** for Mediaspace and MOOC videos (entry id from the video descriptor or the playManifest url); YouTube videos keep only their captions. The Gemini video path is gone.
- **Stitching is kept, only where files are actually merged** (never for slides): polycopiés and exercises stitch ordered sibling pages before being cut. It lands at cut time (Phase 4), deterministic-first, LLM only as a per-document fallback; the old 10-page-batch LLM stitching is replaced.
- **Specs live in semester folders** (e.g. `projects/semester_2627_autumn/micro303.py`), one thin file per project, discovered by convention (`from_code`) with no registry.
- **MICRO-303 pilots the rollout** (over COM202, more recent); the real run waits until all the code is ready.

Phase 4 (judge, cuts, embeddings):

- **The judge stub is deterministic**: it applies the spec's exceptions and fills the default cuts (readable leaves cut whole, containers of text cut per child, roots never cut); `type`/`subtype`/`is_solution` stay unlabelled until Javier's inference lands.
- **A cut consumes the node it applies to**, top-down: the labelled node is replaced by chunk leaves and its subtree never needs a second cut. `per_child` only consumes readable children whose cut is `None` or `per_child` — a child carrying its own different cut (an exception inside a folder) keeps it. Unknown labels fail loud.
- **`whole_document` size guard: drop + warn** (2000 lines, the old batching bound); fixable with a rule. Prevents one huge doc from killing the embedding step.
- **Prompts are documents**: the annotator and stitch prompts migrated verbatim (via `ast`) to md files in `cut/prompts/`, shipped as package data — hand-editable, never transcribed by hand.
- **Chunk vectors: one JSON file per chunk** under `{CACHE_DIR}/embed/{text hash}/embedding.json`, via new `Cache.read`/`write` (text values read in memory); same content-only-key + no-eviction policy as every other transform.
- **`ModelConfig(s)` moved to `rag_etl/core/models.py`** (next to `Resource` and `Rule`): the cut reads the annotator model without dragging in `rag_etl.projects`, whose init eagerly loads every legacy project — a circular import, fixed at the root instead of a `TYPE_CHECKING` workaround.
- **`run_pipeline` returns nothing content-wise**: loaders run inside the pipeline (Phase 5) and publish the index; the embed step computes `(chunk, embedding)` pairs internally.
