# DESIGN

Short note fixing the core shapes that Phases 1–5 implement, updated as the phases settle them.

## Pipeline order

extract (plain documents) → normalize → materialize (OCR etc.) → judge → cut → embed (annotates chunks) → load

- Extractors return plain documents — whatever is includable (Moodle: everything not marked NO_BOT). Whatever per-source labels a source itself infers (MOOC export tags, Ed Discussion's jury) stays in its extractor; the judge fills everything else.
- The **judge** fills `type`/`subtype`/`is_solution` and the `cut` per resource — or drops the resource. It sits after materialize and before cut, because the cut depends on it. There is **no type/subtype → cut mapping in this codebase**: how the judge decides is Javier's business. The built stub (`transformers/judge/`) applies the exceptions and fills the default cuts — readable leaves cut whole, containers of text cut per child, roots never cut — leaving `type`/`subtype`/`is_solution` unlabelled until his inference replaces it.
- Exceptions (below) correct specific resources the judge would mislabel.

## Two packages: `ragtools` and `rag_etl` (Phase 3)

**`src/ragtools/`** — pure file preparation, the colleague-facing library: archives (zip/tar.gz → files, NFC + UTF-8 fixes), notebooks (ipynb → md, ALT texts via a vision model), PDFs (deterministic PyMuPDF page render), images (→ text by a vision model), documents (heading-aware splitting, page stitching, exercise splitting, `documents/`), videos (ffmpeg frame extraction, SRT parsing/merging, the `SlideDetector` interface), the GraphAI and Kaltura plumbing, and the RCP client (`llm/`) with the settled model catalog (`defaults.py`: `MODELS` keyed `(model, mode)`, `get_model(task)` for the concerns — ocr, judge, split_exercises, stitch, embedding). It takes plain paths and hands files and text back; it **never imports `rag_etl`** (enforced by `tests/test_layering.py`) — capabilities build their own clients from the catalog and the environment (`RCP_BASE_URL`/`RCP_API_KEY`, read at call time so import order can never freeze empty endpoints; a `None` field means from-env, an explicit value wins), so the pipeline constructs none. A colleague pip installs the distribution and uses `ragtools` for OCR, splitting or their own file prep; the true standalone distribution (own pyproject, light deps only) splits off in Phase 7.

**`src/rag_etl/`** — the pipeline plumbing: the `Resource` tree, extractors, transformers (thin adapters: walk the tree, call `ragtools` per file, handle caching and node bookkeeping), the cache, the orchestrator, project specs, loaders and the CLI. Transformers follow the explicit-inheritance contract of `Transformer` (`src/rag_etl/transformers/base.py`): frozen dataclasses with `transform(roots) -> roots`, reshaping trees in place.

## `Resource` (Phase 1, `src/rag_etl/core/`)

One recursive dataclass. Nodes are created from extracted files and reshaped in place as the pipeline runs: operations add, update and remove children (or whole nodes), recursively. A chunk is just a leaf with text. Policy never sits in the data.

```python
@dataclass
class Resource:
    # identity
    path: Path | None             # where the content lives on disk; None for pure containers
    url: str | None = None        # public url the resource is available at
    mime_type: str | None = None

    # lineage
    parent: Resource | None = None
    children: list["Resource"] = field(default_factory=list)

    # descriptive metadata (extractors fill some; the judge fills type/subtype/is_solution/cut; exceptions correct)
    title: str | None = None
    type: str | None = None       # theory / practice / ... (inferred by the judge)
    subtype: str | None = None    # lecture_slides / homework / ...
    cut: str | None = None        # whole_document | per_child | per_exercise | text
    number: str | None = None     # exercise number (per_exercise annotator); MOOC dotted "1.3.3"
    subnumber: str | None = None  # exercise sub-number
    week: int | None = None       # lecture week (slides, recordings)
    is_solution: bool = False
    from_: str | None = None      # release date and time, ISO format
    until: str | None = None      # close date and time

    # content (chunks only)
    text: str | None = None
    embedding: list[float] | None = None  # the chunk's vector, set by embed
```

Invariants:

- `is_leaf` is derived (`not children`), never stored. A freshly extracted PDF is a leaf; materialize adds page children; a cut produces chunk leaves. Always current, never outdated. (A video carrying only its caption leaves is a container from the start.)
- `content_hash` is a derived property computed fresh on every access, never stored, so it can never go stale when a step changes content:
  - `text` set → sha256 of the text (most-derived: chunks, OCR'd pages, extracted docs)
  - else children → sha256 over the children's hashes in document order (a container's content IS its children; order matters)
  - else path → sha256 of the file bytes
  - else → sha256 of empty
  Equal content anywhere in any project → equal hash → shared cache entries, `(transform, content_hash)` keys.
- Deep links (Moodle URL, Mediaspace timestamp, Ed post, Git commit) travel as `url`; `lineage()` reconstructs the chain from any node back to its source. Video leaves never carry an `entry_id` field — the Kaltura entry id stays inside the video descriptor/XML and is read by materialize.
- Embeddings land on the node (`embedding`, chunks only): the embed transform annotates the tree in place, and the ES loader reads vectors straight from the tree.

State helpers on the class: `is_leaf`, `walk()` (depth-first), `lineage()` (root→self), `add_child()`. The hashing helpers `hash_bytes`/`hash_text` are the single sha256 implementation behind `content_hash`. Transformers add and remove children directly; a node reshaped into a container keeps its identity and gains children.

## Exceptions (Phase 1, `src/rag_etl/core/rules.py`)

The only rules in the system: manual corrections for specific resources that would trip up the judge.

```python
@dataclass(frozen=True)
class Rule:
    where: str                    # flat ANDed globs over resource attributes, e.g. 'title="*[[]HOMEWORK*]*"'
    type: str | None = None
    subtype: str | None = None
    cut: str | None = None        # whole_document | per_child | per_exercise | text
    number: str | None = None
    week: int | None = None
    is_solution: bool | None = None
```

- `apply(resource, exceptions)` corrects matching resources in place, on top of whatever the judge filled.
- Most-specific wins (most predicates); equal specificity → first-declared wins, loudly (warn, never silently resolve); no match → resource untouched.
- Globs are fnmatch patterns; a literal `[` is written `[[]`. A `where` naming a non-existent attribute never matches, with a warning.

## `ProjectSpec` (Phase 3, `src/rag_etl/projects/spec.py`)

One thin, declarative spec per project. Sources are **extractor instances**, each configured for its source (frozen dataclasses with typed fields); the judge labels everything else. Models are not in the spec: they are settled pairings in the `ragtools.llm.defaults` catalog, pinned per concern by the capabilities themselves.

```python
@dataclass(frozen=True)
class ProjectSpec:
    id: str                                  # lowercase, e.g. "com202"; names the index/alias rag_{id}
    language: str | None = None              # the project's language (materialize slide detection)
    exceptions: tuple[Rule, ...] = ()        # manual corrections for resources the judge would mislabel
    sources: tuple[Extractor, ...] = ()      # extractor instances, one per source
    dir: Path | None = None                  # the project's material folder; manifests land in dir/output
```

Specs live in **semester folders** (e.g. `projects/semester_2627_autumn/micro303test.py` — MICRO-303 under the test id until cutover), one thin file per project, named after the project in lowercase and defining the spec under the project code in uppercase. There is no registry: `from_code` discovers a spec by convention (scans the semester folders; a file that exists is a project) and errors loudly when there is none.

## Normalize + materialize (Phase 3, `src/rag_etl/transformers/`)

- **Normalize**: zip/tar.gz archives are extracted (the archive node keeps its identity and gains one child per extracted file, mime-filtered, UTF-8-fixed), Jupyter notebooks become Markdown (ALT texts via a vision model), and files bigger than their mime family's limit are dropped (PDF 50 MB, images 10 MB, text 5 MB — module constants).
- **Materialize**: every PDF page is rendered deterministically (PyMuPDF at ~72 DPI, clamped to 2048×3072) and read by a vision model — the pdf node gains one child per page, carrying the parent's descriptive metadata; images are read in place (unreadable ones dropped); videos join one uniform Kaltura path — the entry id comes from the video descriptor (Mediaspace) or the playManifest url (MOOC), the slides entry is found, slide changes detected (GraphAI behind the swappable `SlideDetector`), one frame per segment extracted with ffmpeg straight from the remote url, every frame read, and the words spoken over each segment appended to it as a quote. The caption leaf is consumed (the subtitles are part of the slides); YouTube videos, with no Kaltura entry, pass through untouched.
- Model calls (OCR, ALT texts) go through the unified `RCPClient`: standard retry, then the run stops loud — a page that cannot be read is not silently skipped.
- GraphAI failing on one video drops that video rather than the run (old behavior, kept).

## Cuts + embed (Phase 4, `src/rag_etl/transformers/cut/`, `embed/`)

- The **cuts registry** (`CUTS`): one entry per kind of cut, mapping a resource's `cut` label to its cut function; new cuts join via PR, never course-local code. An unknown label fails loud.
- A cut consumes the node it applies to, top-down: the node is replaced by chunk leaves (a chunk is a leaf with text) and its subtree never needs a second cut. Roots are never cut whole.
  - `whole_document`: the node becomes one chunk — its own file, or its ordered page children stitched (children consumed). Over the size guard (2000 lines) the chunk is dropped with a loud warning; fixable with a rule.
  - `per_child`: each readable child becomes a chunk in place, text read from its file. Children carrying their own different cut keep it (an exception inside a folder cut per child still applies).
  - `per_exercise`: the node's pages are stitched, then split — one LLM pass returning exercises as line ranges with number + is-solution (prompts kept as documents in `ragtools/documents/prompts/`, cached by the stitched text's hash) — then cut at the annotated markers: one chunk per exercise, deep-linking into the pdf page its text starts on. Nothing is split unless the cut asks for it.
  - `text`: header-aware chunking (split at the most common heading level, adjacent parts merged up to the guard).
- **Embed** (`embed/embed.py`, the fifth transform): RCP embeddings per chunk (`Qwen/Qwen3-Embedding-8B`, pinned in the catalog), batched, cache-aware — one JSON file per chunk under `{CACHE_DIR}/embed/{text hash}/`, via the cache's `read`/`write` text API — annotating each chunk in place (`Resource.embedding`).
- `run_pipeline` returns nothing content-wise: loaders run inside the pipeline and publish the index; failures propagate loud.

## Loaders (Phase 5, `src/rag_etl/loaders/`)

Loaders take trees — the pipeline contract is uniform (extractors produce trees, transformers take and return trees, loaders take trees) — and are configured by constructor args; no intermediate run object.

- **`ManifestLoader`** writes one JSON manifest per run for human inspection; nothing reads it. All manifests land in `{dir}/output/` as `{timestamp}_manifest[_dry].json` (name-sorted, dry by suffix). Header: project, dry, timestamp, index, alias, pipeline version. Summary: sources, chunks, without_url, and a nested drill-down (source → type → subtype → number) counting chunks, without_url and is_solution per group, with `unknown` buckets and None sorted last. Tree: metadata-only — no text, no embeddings — with content-hash pointers, the `chunk` flag and the `cut` decision.
- **`ElasticsearchLoader`** publishes one run: recreate the date-named index (`rag_{project}_{YYYY_MM_DD}`, Ramtin's convention; delete + create on same-day collision) → bulk (doc id `{epoch_ms}-{content_hash}`, within-run content dedupe, first occurrence wins with a warning) → refresh → **one atomic** `update_aliases` (remove old + add new; add-only on first run). A failed run never touches the alias. Documents carry `content`, `embedding` (from the node), a `lineage` breadcrumb (title → url → `"unknown"` per node, root→leaf), and a `url` that is always present: own → nearest ancestor's → `"<redacted>"` (redaction is pipeline policy).
- **The frozen schema** ships verbatim as package data (`index_settings.json`, settings + mappings in one create-index body); the loader patches `content.analyzer` per project language (`base_en`/`base_fr`) at creation. 13 fields: `id`, `content`, `embedding` (dense_vector 4096, dot_product, int8_hnsw), `lineage`, `url`, `type`, `subtype`, `number`, `subnumber`, `week`, `is_solution`, `from`, `until` (ES default ISO dates). Dropped vs the old mapping: `content.en/fr`, `description.en/fr`, `raw`/`sayt`/`trigram` subfields + their analyzers and filters, synonyms, legacy date formats, 384 dims.
- **Connection**: flat `ES_HOST`/`ES_PORT`/`ES_USERNAME`/`ES_PASSWORD`/`ES_CAFILE` env keys, proper TLS verification (`ca_certs`); `elasticsearch>=8,<9`. Search stays hybrid (embeddings + Elasticsearch text); consumers filter on `from`/`until` (automatic), `type`/`subtype`/`is_solution` — returned fields and routing are the MCP's concern.

## Stitching (Phase 4, at cut time)

Page-by-page OCR leaves pages that sometimes need merging: headings at different levels, formulas or tables cut at page boundaries. Stitching is **kept, but only where files are actually merged**: not for slides (each slide is its own chunk), but for course notes (polycopié) and exercises, where ordered sibling pages are merged before being cut into chunks. It therefore belongs to the cuts (Phase 4), not to materialize — materialize always keeps per-page children.

Direction (replacing the old 10-page-batch LLM stitching, ~20 LLM calls per long PDF):

- **Deterministic first** (implemented, `ragtools/documents/stitch.py`): page boundary markers, tables and display equations cut across consecutive pages joined (no marker between the joined sides — it would break the block), soft-hyphenated words put back together (heuristic: the continuation starts in lowercase). Heading-level normalization is deliberately not deterministic — too risky without document context; the LLM fallback covers it.
- **LLM stitch as a fallback, per document in a single call**, behind the same stitch interface, off by default — the old prompt, kept as a document in `ragtools/documents/prompts/`.

## What this kills

- The 41 `tag_metadata` tables and `course_info`: **not converted — dropped.** Type/subtype/is_solution/cut are inferred by the judge; only manual exceptions survive as rules (MOOC-export tags and Ed Discussion's jury stay in their extractors).
- Policy flags in data (`one_chunk_per_page`, `processing_method`, `model`, `is_gemini_processed_video`, `tikz`).
- The two parallel video paths and `<!-- page N -->` markers (Phase 3 materialize: PDF → page children OCR'd by RCP; video → GraphAI `detect_slides` + Kaltura SRT upstream, then frame OCR; MOOC videos join the same Kaltura path, the Gemini path is gone).
- The old LLM stitching of PDF pages into one document (10-page batches): replaced by per-page children and, in Phase 4, deterministic-first stitching where files are actually merged.
- Moodle exclusions: the extractor returns only what is includable (everything not NO_BOT); nothing downstream re-decides exclusion.
