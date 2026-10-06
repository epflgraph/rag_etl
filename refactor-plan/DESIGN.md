# DESIGN

Short note fixing the core shapes that Phases 1–5 implement, updated as the phases settle them.

## Pipeline order

extract (plain documents) → normalize → materialize (OCR etc.) → judge → cut → embed + load

- Extractors return plain documents — whatever is includable (Moodle: everything not marked NO_BOT). No tags anywhere.
- The **judge** (a stub we build, replaced by Javier's inference) fills `type`/`subtype`/`is_solution` and the `cut` per resource — or drops the resource. It sits after materialize and before cut, because the cut depends on it. There is **no type/subtype → cut mapping in this codebase**: how the judge decides is Javier's business.
- Exceptions (below) correct specific resources the judge would mislabel.

## `Resource` (Phase 1, `src/rag_etl/core/`)

One recursive dataclass. Nodes are created from extracted files and reshaped in place as the pipeline runs: operations add, update and remove children (or whole nodes), recursively. A chunk is just a leaf with text. Policy never sits in the data.

```python
@dataclass
class Resource:
    # identity
    path: Path | None             # where the content lives on disk; None for pure containers
    mime_type: str | None = None

    # lineage
    parent: Resource | None = None
    children: list["Resource"] = field(default_factory=list)

    # descriptive metadata (extractors fill some; the judge fills type/subtype/is_solution/cut; exceptions correct)
    title: str | None = None
    type: str | None = None       # theory / practice / ... (inferred by the judge)
    subtype: str | None = None    # lecture_slides / homework / ...
    cut: str | None = None        # whole_document | per_child | per_exercise | text
    number: int | None = None     # exercise number (per_exercise annotator)
    sub_number: int | None = None # exercise sub-number
    week: int | None = None       # lecture week (slides, recordings)
    is_solution: bool = False
    from_: date | None = None     # release date (Moodle availability)
    until: date | None = None     # close date

    # content (chunks only)
    text: str | None = None
```

Invariants:

- `is_leaf` is derived (`not children`), never stored. A freshly extracted PDF is a leaf; materialize adds page children; a cut produces chunk leaves. Always current, never outdated.
- `content_hash` is a derived property computed fresh on every access, never stored, so it can never go stale when a step changes content:
  - `text` set → sha256 of the text (most-derived: chunks, OCR'd pages, extracted docs)
  - else children → sha256 over the children's hashes in document order (a container's content IS its children; order matters)
  - else path → sha256 of the file bytes
  - else → sha256 of empty
  Equal content anywhere in any project → equal hash → shared cache entries, `(transform, content_hash)` keys.
- Deep links (Moodle URL, Mediaspace timestamp, Ed post, Git commit) travel as `path`/URL fragments; `lineage()` reconstructs the chain from any node back to its source.
- Embeddings are **not** on the node: the embed step produces them per chunk and hands the ES loader `(chunk, embedding)` pairs.

State helpers on the class: `is_leaf`, `walk()` (depth-first), `lineage()` (root→self), `add_child()`. The hashing helpers `hash_bytes`/`hash_text` are the single sha256 implementation behind `content_hash`.

## Exceptions (Phase 1, `src/rag_etl/core/rules.py`)

The only rules in the system: manual corrections for specific resources that would trip up the judge.

```python
@dataclass(frozen=True)
class Rule:
    where: str                    # flat ANDed globs over resource attributes, e.g. 'title="*[[]HOMEWORK*]*"'
    type: str | None = None
    subtype: str | None = None
    cut: str | None = None        # whole_document | per_child | per_exercise | text
    number: int | None = None
    week: int | None = None
    is_solution: bool | None = None
```

- `apply(resource, exceptions)` corrects matching resources in place, on top of whatever the judge filled.
- Most-specific wins (most predicates); equal specificity → first-declared wins, loudly (warn, never silently resolve); no match → resource untouched.
- Globs are fnmatch patterns; a literal `[` is written `[[]`. A `where` naming a non-existent attribute never matches, with a warning.

## `ProjectSpec` (Phase 1, `src/rag_etl/projects/`)

One thin, declarative spec per project.

```python
@dataclass(frozen=True)
class ProjectSpec:
    id: str                                  # lowercase, e.g. "com202"; names the index/alias rag_{id}
    exceptions: tuple[Rule, ...] = ()
    sources: tuple[SourceConfig, ...] = ()   # extractor configs (Phase 2)
    models: ModelConfigs = ModelConfigs()    # OCR, judge, annotator, embedding (Phase 3)
```

## What this kills

- The 41 `tag_metadata` tables and `course_info`: **not converted — dropped.** Type/subtype/is_solution/cut are inferred by the judge; only manual exceptions survive as rules.
- Policy flags in data (`one_chunk_per_page`, `processing_method`, `model`, `is_gemini_processed_video`, `tikz`).
- The two parallel video paths and `<!-- page N -->` markers (Phase 3 materialize: PDF → page children OCR'd by RCP; video → GraphAI `detect_slides` + Kaltura SRT upstream, then frame OCR).
- Moodle exclusions: the extractor returns only what is includable (everything not NO_BOT); nothing downstream re-decides exclusion.
