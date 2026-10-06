# DESIGN

Short note fixing the two core dataclass shapes that Phases 1–5 implement, so the phases build against one agreement rather than re-deciding per phase. Module names follow IMPLEMENTATION.md §3; nothing here is implemented yet.

## `Resource` (Phase 1, `src/rag_etl/core/`)

One recursive dataclass. Container / child / chunk is *state*, not subclassing; policy never sits in the data.

```python
@dataclass
class Resource:
    # identity
    path: Path | None             # where the content lives on disk; None for pure containers
    content_hash: str             # sha256 of the content this node represents
    mime_type: str | None = None

    # lineage
    parent: Resource | None = None
    children: list["Resource"] = field(default_factory=list)

    # descriptive metadata (extractors fill some; judge fills type/subtype/is_solution; rules resolve conflicts)
    title: str | None = None
    type: str | None = None       # theory / practice / ... (resource-level, resolved by rules or judge)
    subtype: str | None = None    # lecture_slides / homework / notebook / ...
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

- A container has non-empty `children`; a file has none until cut; a chunk has `text`. `kind` is derived (`container` iff children, `chunk` iff text), not stored.
- `content_hash` is computed from file bytes for files/pages and from text for chunks. Equal content anywhere in any project → equal hash → shared cache entries, `(transform, content_hash)` keys.
- Deep links (Moodle URL, Mediaspace timestamp, Ed post, Git commit) travel as `path`/URL fragments on the node so every chunk keeps a link back to its origin.
- Embeddings are **not** on the node: the embed step produces them per chunk and hands the ES loader `(chunk, embedding)` pairs, cache-aligned with `content_hash` (settled in Phase 3/4).

State helpers on the class: `is_container`, `walk()` (depth-first), `lineage()` (root→self list), `add_child()`. The pipeline passes trees; transformers reshape them in place.

## `ProjectSpec` (Phase 1, `src/rag_etl/projects/`)

One thin, declarative spec per project replaces the 41 course classes with their `course_info` + `tag_metadata` dicts and `*_type_subtypes` plumbing.

```python
@dataclass(frozen=True)
class Rule:
    where: str                    # flat ANDed globs over resource attributes, e.g. "filename=*_exercises.pdf AND path=week*"
    type: str | None = None
    subtype: str | None = None
    recipe: str | None = None     # whole_document | one_per_page | per_exercise | per_slide | text
    number: int | None = None
    week: int | None = None
    is_solution: bool | None = None
    exclude: bool = False         # exclude-only rules (professor's Moodle exclusions)

@dataclass(frozen=True)
class ProjectSpec:
    id: str                       # lowercase, e.g. "com202"; names the index/alias rag_{id}
    rules: tuple[Rule, ...] = ()
    overrides: tuple[Rule, ...] = ()   # staff corrections; win over rules and judge defaults
    default_recipe: str = "text"       # fallback for anything no rule matches
    sources: tuple[SourceConfig, ...] = ()   # extractor configs (moodle course id, mooc path, mediaspace channel, ...)
    models: ModelConfigs = ModelConfigs()    # OCR, judge, annotator, embedding model + parameters
```

Resolution order (Phase 1): staff override > judge default > rules; most-specific within a table; ambiguity → warn, never silent. The `where`-matcher is flat ANDed globs — no nesting, no logic beyond AND.

## What this kills

- `BaseCourse` + 41 subclasses, `tag_metadata`, `course_info`, and every `*_type_subtypes` derived property (Phase 1 converts the tables mechanically, with a round-trip check).
- Policy flags in data (`one_chunk_per_page`, `processing_method`, `model`, `is_gemini_processed_video`, `tikz`).
- The two parallel video paths and `<!-- page N -->` markers (Phase 3 materialize: PDF → page children OCR'd by RCP; video → GraphAI `detect_slides` + Kaltura SRT upstream, then frame OCR).
