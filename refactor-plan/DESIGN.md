# DESIGN

Short note fixing the core shapes that Phases 1–5 implement, updated as the phases settle them.

## Pipeline order

extract (plain documents) → normalize → materialize (OCR etc.) → judge → cut → embed + load

- Extractors return plain documents — whatever is includable (Moodle: everything not marked NO_BOT). Whatever per-source labels a source itself infers (MOOC export tags, Ed Discussion's jury) stays in its extractor; the judge fills everything else.
- The **judge** (a stub we build, replaced by Javier's inference) fills `type`/`subtype`/`is_solution` and the `cut` per resource — or drops the resource. It sits after materialize and before cut, because the cut depends on it. There is **no type/subtype → cut mapping in this codebase**: how the judge decides is Javier's business.
- Exceptions (below) correct specific resources the judge would mislabel.

## Two packages: `ragtools` and `rag_etl` (Phase 3)

**`src/ragtools/`** — pure file preparation, the colleague-facing library: archives (zip/tar.gz → files, NFC + UTF-8 fixes), notebooks (ipynb → md, ALT texts via a vision model), PDFs (deterministic PyMuPDF page render), images (→ text by a vision model), videos (ffmpeg frame extraction, SRT parsing/merging, the `SlideDetector` interface) and the unified RCP client (`llm/`) with the recommended parameters per model. It takes plain paths and hands files and text back; it **never imports `rag_etl`** — model clients are constructed by the caller and handed over. A colleague pip installs the distribution and uses `ragtools` for OCR or their own file prep; the true standalone distribution (own pyproject, light deps only) splits off in Phase 7.

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
- Embeddings are **not** on the node: the embed step produces them per chunk and hands the ES loader `(chunk, embedding)` pairs.

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

One thin, declarative spec per project. Sources are **extractor instances**, each configured for its source (frozen dataclasses with typed fields); the judge labels everything else.

```python
@dataclass(frozen=True)
class ModelConfig:
    model: str                              # the model; recommended params come from ragtools.llm.defaults
    params: Mapping[str, Any] = {}          # per-call overrides of those defaults

@dataclass(frozen=True)
class ModelConfigs:
    ocr: ModelConfig | None = None          # vision reads: PDF pages, frames, images (materialize)
    judge: ModelConfig | None = None        # type/subtype/is_solution/cut inference (Phase 4)
    annotator: ModelConfig | None = None    # exercise number + is-solution per chunk (Phase 4)
    embedding: ModelConfig | None = None    # chunk embeddings (Phase 4)

@dataclass(frozen=True)
class ProjectSpec:
    id: str                                  # lowercase, e.g. "com202"; names the index/alias rag_{id}
    language: str | None = None              # the project's language (materialize slide detection)
    exceptions: tuple[Rule, ...] = ()        # manual corrections for resources the judge would mislabel
    sources: tuple[Extractor, ...] = ()      # extractor instances, one per source
    models: ModelConfigs = ModelConfigs()    # OCR, judge, annotator, embedding
```

Specs live in **semester folders** (e.g. `projects/semester_2627_autumn/micro303.py`), one thin file per project, named after the project in lowercase and defining the spec under the project code in uppercase. There is no registry: `from_code` discovers a spec by convention (scans the semester folders; a file that exists is a project) and errors loudly when there is none.

## Normalize + materialize (Phase 3, `src/rag_etl/transformers/`)

- **Normalize**: zip/tar.gz archives are extracted (the archive node keeps its identity and gains one child per extracted file, mime-filtered, UTF-8-fixed), Jupyter notebooks become Markdown (ALT texts via a vision model), and files bigger than their mime family's limit are dropped (PDF 50 MB, images 10 MB, text 5 MB — module constants).
- **Materialize**: every PDF page is rendered deterministically (PyMuPDF at ~72 DPI, clamped to 2048×3072) and read by a vision model — the pdf node gains one child per page, carrying the parent's descriptive metadata; images are read in place (unreadable ones dropped); videos join one uniform Kaltura path — the entry id comes from the video descriptor (Mediaspace) or the playManifest url (MOOC), the slides entry is found, slide changes detected (GraphAI behind the swappable `SlideDetector`), one frame per segment extracted with ffmpeg straight from the remote url, every frame read, and the words spoken over each segment appended to it as a quote. The caption leaf is consumed (the subtitles are part of the slides); YouTube videos, with no Kaltura entry, pass through untouched.
- Model calls (OCR, ALT texts) go through the unified `RCPClient`: standard retry, then the run stops loud — a page that cannot be read is not silently skipped.
- GraphAI failing on one video drops that video rather than the run (old behavior, kept).

## Stitching (Phase 4, at cut time)

Page-by-page OCR leaves pages that sometimes need merging: headings at different levels, formulas or tables cut at page boundaries. Stitching is **kept, but only where files are actually merged**: not for slides (each slide is its own chunk), but for course notes (polycopié) and exercises, where ordered sibling pages are merged before being cut into chunks. It therefore belongs to the cuts (Phase 4), not to materialize — materialize always keeps per-page children.

Direction (replacing the old 10-page-batch LLM stitching, ~20 LLM calls per long PDF):

- **Deterministic first**: heading-level normalization (one document title, consistent levels), joining tables and display equations split across pages, fixing soft hyphenation — all deterministic post-processing over the ordered page children, zero LLM calls.
- **LLM stitch as a fallback, per document in a single call**, behind the same stitch interface, off by default — only where deterministic fixes leave problems.

## What this kills

- The 41 `tag_metadata` tables and `course_info`: **not converted — dropped.** Type/subtype/is_solution/cut are inferred by the judge; only manual exceptions survive as rules (MOOC-export tags and Ed Discussion's jury stay in their extractors).
- Policy flags in data (`one_chunk_per_page`, `processing_method`, `model`, `is_gemini_processed_video`, `tikz`).
- The two parallel video paths and `<!-- page N -->` markers (Phase 3 materialize: PDF → page children OCR'd by RCP; video → GraphAI `detect_slides` + Kaltura SRT upstream, then frame OCR; MOOC videos join the same Kaltura path, the Gemini path is gone).
- The old LLM stitching of PDF pages into one document (10-page batches): replaced by per-page children and, in Phase 4, deterministic-first stitching where files are actually merged.
- Moodle exclusions: the extractor returns only what is includable (everything not NO_BOT); nothing downstream re-decides exclusion.
