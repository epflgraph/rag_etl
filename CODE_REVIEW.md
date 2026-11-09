# Code review: refactor proposal for `rag-etl`

Review of `src/rag_etl` and `src/ragtools` after the refactor. The MOOC and Ed
Discussion extractors were reviewed by a helper agent; its main claims were
spot-checked (the `tag_metadata` flags, Ed's catalogue still reading the old
files) but not every line reference. The test suite passes (153 tests) as a
baseline. Nothing has been changed yet.

## The core ideas

1. **Declare the label fields once.** The metadata fields (`type`, `subtype`,
   `number`, …, `from_`, `until`) are currently typed out by hand in five
   places: `Rule`, the rule matcher, judge inheritance, the manifest and the
   Elasticsearch document. Grouping them into one small `Labels` dataclass
   would let all five follow from a single declaration. This is the same kind
   of simplification as the singleton → block change.
2. **One interface for every step.** Extractors, transformers and loaders
   would share one shape: a `Step` with a `name` and `run(roots) -> None`.
   Trees are changed in place, so returning them adds nothing. The pipeline
   becomes a plain list, and `name` drives the cache folder, the log lines and
   the allowed `--refresh` values.
3. **A shared Merkle helper in the cache.** About 80 lines of node-record
   reading and writing are copied between `materialize/pdf.py` and
   `materialize/video.py`. Moving them into the cache removes the copy. If the
   recipe version also becomes part of the cache folder name (for example
   `materialize/pdf-ocr-1`), the separate stale/`bypass` handling can be
   deleted entirely.
4. **One config module.** Both packages would read real environment
   variables through `ragtools.config`.
5. **Move remaining business logic into `ragtools`.** That covers HTML →
   markdown, edX quiz → markdown, slide-deck classification, the GraphAI slide
   detector and Kaltura session setup. Extractors and transformers then stay
   thin adapters.
6. **One name per concept.** See the terminology table in §5.

## 1. Legacy references to fix

| Where | What |
|---|---|
| `projects/semester_2627_autumn/micro303.py:17-40`, `extractors/mooc/utils.py:77` (`cut_from_flags`), MOOC parsers, `tests/test_extractors.py:247` | Still uses `tag_metadata` with the `one_chunk_per_page`/`one_chunk_per_doc` flags |
| `extractors/mooc/html_parser.py:485` | Reads a `split_exercises` flag from data |
| `extractors/ed_discussion/catalogue.py`, `ed_discussion_extractor.py:55` | Builds its catalogue from `output/metadata/*.json`, which nothing writes any more, so it always returns empty. It also uses the old key names `sub_number`, `doc_number` and `"from"` |
| `extractors/ed_discussion/ed_discussion_extractor.py:59-71`, `.example.env` | Picks its models from `RCP_EDSTEM_EXTRACTOR_MODEL*` instead of the model catalogue. The `RCP_BASE/THINKING/VISION/EMBEDDING_MODEL` variables are read by nothing |
| `extractors/ed_discussion/image_to_md.py:3-6` | Docstring still talks about "pre-refactor transformers" |
| `extractors/mediaspace/mediaspace_extractor.py:42, 229-255` | Mentions `VideoToFramesTransformer`; uses an `srt_path` variable |
| `README.md` | Describes the three old video transformers end to end |
| `pyproject.toml:66, 78` | Comment about `course_info`/`tag_metadata`; mypy override for `rag_etl.utils.llms`, which no longer exists |
| `cli.py:24` | Help text "Project code, e.g. COM202 (as in the class name)": projects are no longer classes |
| `rag_etl/utils/` | Its own docstring calls it "legacy but alive"; `tags.py` only parses legacy `[TAG]` markers |
| MOOC titles (`html_parser.py:171,406`, `quiz_parser.py:93`, `video_parser.py:118`) | Still prefix the parent's name ("vertical - item"), which commit 23fc962 removed elsewhere |
| `refactor-plan/EXAMPLES.md` | Shows `pdf_to_markdown`, `video_to_slides` and `jupyter_to_markdown`, none of which exist |
| `ragtools/mime_types.py:41` | Stale "ToDo: Support (.tar.gz)" (tar is supported now) |

## 2. Bugs found

- **Real environment variables are ignored.** `rag_etl/config.py` uses
  `dotenv_values(<repo>/.env)`, which reads only the file. Variables set by
  systemd or cron, or by a pip-installed (non-editable) copy, are missed.
  `ragtools.config` reads `os.environ`.
- **Some video slides never get their subtitles.** In
  `materialize/video.py:294-297`, any slide whose transcription contains a
  `\n> ` line is treated as already merged. The guard isn't needed anyway,
  because the work folder is wiped before each video.
- **`--refresh` accepts any string.** A typo refreshes nothing, silently.
- **Two size limits disagree.** `whole_document` allows up to 2000 lines, but
  embed raises above 96k characters. A long-lined 1,500-line document passes
  the cut and then crashes the run.
- **Content hashes are recomputed far too often.** `content_hash` is
  recalculated on every access, and the manifest calls it for every node.
  Containers rehash their whole subtree, and PDFs are re-read from disk at
  each level.
- **MOOC bugs reported by the review agent (not re-checked):**
  - `get_video_platform_id` raises `UnboundLocalError` when both ids are
    empty.
  - `quiz_parser.py:281` chains `sol.find(...) or …`, and an lxml element with
    no children is falsy, so the wrong container can be picked.
  - The quiz solution leaf should be `"statement + solution"`.
- **Ed Discussion leaves always have `url=None`**, even though the thread id
  is available for a deep link.
- **The Elasticsearch mapping declares an `id` field that is never written.**

## 3. Abstractions

### a) `Labels`

```python
@dataclass
class Labels:
    type: str | None = None
    subtype: str | None = None
    number: str | None = None
    subnumber: str | None = None
    week: int | None = None
    exercise_parts: ExerciseParts | None = None
    has_solution: bool = False
    available_from: str | None = None
    available_until: str | None = None

    def inherit(self, parent: Labels) -> None: ...   # fill only the fields that are None
    def update(self, other: Labels) -> None: ...     # apply an override's non-None fields
```

- `Resource` then has four groups: identity (`title`, `path`, `url`,
  `mime_type`), tree (`parent`, `children`), `labels`, and content (`text`,
  `embedding`). `cut` stays on `Resource`, because it is a processing decision
  rather than metadata that goes into the index.
- `Override(where, cut=None, labels=Labels(...))` replaces `Rule` and its
  copied field list.
- The judge's `_inherit` becomes `node.labels.inherit(node.parent.labels)`.
- The manifest and Elasticsearch document are built from
  `asdict(node.labels)`, with renames only where Elasticsearch needs another
  name (`from`/`until`).
- `add_child` stops copying `cut`, `text` and `embedding`. The `cut=None`
  overrides in `cut.py` exist only to undo that copying.

### b) Helpers on `Resource`

- `detach()` replaces about 12 copies of
  `if node.parent: node.parent.children.remove(node)`.
- `is_chunk` and an `iter_chunks(roots)` function replace the `chunks_of` /
  `_chunks` / Elasticsearch walks.
- `is_text` moves to `mime_types.is_text(mime)`. The current check
  (`MARKDOWN or text/*`) is redundant, because markdown is already `text/*`.

### c) `Step` and cache naming

```python
class Step(ABC):
    name: ClassVar[str]              # "normalize", "materialize", …, "manifest", "elasticsearch"
    @abstractmethod
    def run(self, roots: list[Resource]) -> None: ...
```

- The pipeline becomes `for step in steps: step.run(roots)`.
- The cache is built per step as `Cache(root, project, step.name)`.
- `--refresh` gets `choices=[s.name for s in steps if s.cached]`.
- The loaders and `EmbedTransformer` become frozen dataclasses like the rest.
  Today `EmbedTransformer` is a plain class and the loaders have no base
  class.

### d) Cache

- One API with two levels:
  - `get`/`set` for files.
  - `read_text`/`write_text` for small values.
  - `node(key) -> list[(name, hash)] | None`, `adopt(children, dir)` and
    `write_node(key, children)` for the Merkle records.
- The recipe version goes into the cache folder name. That deletes the
  `fingerprint` field in node records, the `stale` flag and
  `ocr_cached(..., bypass=)`.
- The cache root is passed in (from the pipeline) instead of `Cache` reading
  the config itself.

### e) Materialize as a registry like `CUTS`

`MATERIALIZERS = {mt.PDF: pdf, mt.PNG: image, …}`. The "already
materialized?" checks become one rule: a node is materialized once, while it
is still a leaf. Video subtitles would be attached as an attribute rather than
a child for this to hold.

### f) Videos without descriptor files

- Mediaspace writes a JSON file holding only the entry id, and materialize
  parses it back. The entry id is already in the url (`/media/t/<id>`), so:
  - extend `extract_entry_id_from_url` to read that url shape;
  - key the work folder and the node record by entry id;
  - delete `write_descriptor` and `build_descriptor_path` and the JSON branch
    of `entry_id_of`.
- MOOC currently points `path` at the XML with an MP4 mime type, which is the
  same workaround.
- Build each slide's text in one pass (transcription + `> transcript`) instead
  of rewriting files in place. That removes `_merge_captions`' re-sorting and
  the guard bug from §2.

### g) Labels on extractors become overrides

- `MediaspaceExtractor.type/subtype/cut/exercise_parts` and MOOC's
  `tag_metadata` are policy living on extractors.
- Add a `source` attribute to override matching (the root's title, or an
  extractor `name`). The spec can then say
  `Override('source="mediaspace*"', cut=...)`.
- MOOC keeps only what the export itself states, as a typed
  `labels: Mapping[str, MoocLabel]` with real cut names; `cut_from_flags` is
  deleted.

## 4. Proposed layout

```
src/ragtools/
├── config.py            # required()/optional() over os.environ — the ONLY config module (rag_etl uses it too)
├── files.py             # iter_files (one hidden/macOS-skipping walk), sanitize_filename, ensure_utf8, zip names
│                        #   ← encoding.py + archives.iter_files + mediaspace/utils.safe_filename + local/git walks
├── archives.py
├── mime_types.py        # constants as tuples/frozensets + is_text/is_image/is_archive/size_family
├── prompts/*.md         # ALL prompts: pdf_page, slide, alt_text, stitch, split_exercises, classify_slides, ed_jury
├── llm/client.py        # was rcp.py; _send + _send_streaming merged into one path
├── llm/catalog.py       # was defaults.py; + "ed_jury" task, drop unused entries/tasks
├── transcribe.py        # was images.py: transcribe_image, alt_text
├── pdf.py               # render pages (+ public pdf_to_markdown for colleagues); drop unused to_data_uri, dpi
├── notebooks.py         # notebook_to_markdown
├── video.py             # frames, subtitles (Cue, parse_srt, text_between), SlideDetector
├── documents/           # headings, stitch, exercises, html.py (← mooc html_parser), edx_quiz.py (← quiz_parser)
└── services/
    ├── graphai.py       # + GraphAISlideDetector (moved from rag_etl/transformers/materialize/videos.py)
    └── kaltura.py       # + session_from_env(), video_url(entry_id)  (dedupes mediaspace + materialize)

src/rag_etl/
├── resource.py          # Resource, Labels, hashing   (core/ flattened — 2 files don't need a package)
├── overrides.py         # was core/rules.py
├── step.py              # Step ABC
├── cache.py             # + Merkle node helpers
├── pipeline.py, cli.py
├── extractors/
│   ├── base.py, markers.py   # NO_BOT + marker stripping (← utils/tags.py; dedupes moodle/local_folder)
│   ├── moodle.py, local_folder.py, git.py, mediaspace.py   # single-file packages flattened
│   ├── mooc/            # one tree.py instead of course/chapter/sequential/vertical parsers
│   └── ed_discussion/
├── transformers/
│   ├── normalize.py, judge.py, cut.py, embed.py            # cut/cut_transformer.py etc. flattened
│   └── materialize/__init__.py (registry), pdf.py, image.py, video.py   # ocr.py & videos.py dissolved
├── loaders/manifest.py, elasticsearch.py, index_settings.json
└── projects/spec.py, semester_2627_autumn/
```

Deleted entirely: `rag_etl/config.py`, `rag_etl/utils/`, `rag_etl/core/`,
`transformers/materialize/{ocr,videos}.py` and
`extractors/ed_discussion/image_to_md.py`. The single-file `cut/`, `judge/`,
`embed/` and `normalize/` subpackages also go; today they have no
`__init__.py` and only work as implicit namespace packages.

According to the review agent, MOOC and Ed together should shrink by roughly
1,100 lines, about 400 of which move into `ragtools`.

## 5. Terminology

| Concept | Names in use today | Proposal |
|---|---|---|
| A unit indexed for one bot | project, course, code, `spec.id`, `from_code(code)`, `course_path` | **project** and `project.id`; `from_code` → `load_project(id)` |
| Manual corrections | exceptions, rules, overrides, `Rule`, `apply` | **override** (`Override`, `spec.overrides`). "Exceptions" collides with Python exceptions |
| type/subtype/number/… | metadata, labels, tags | **labels**. "Tag" only for source markers (MOOC `[TAG]`, `[NO_BOT]`) |
| Release window | `from_`/`until`, availability, release window | `available_from`/`available_until` in Python; Elasticsearch keeps `from`/`until` |
| Image → text by a vision model | ocr, read, materialize, image-to-markdown | **transcribe** (`transcribe_image`, "transcript"); **materialize** is only the step name |
| Spoken text | caption, subtitles, srt, transcript, cue | **subtitles** for the file and parsing; **speech** for what is appended to a slide |
| Video time span | segment, interval, slide, frame | **segment** for the time span, **frame** for the image, **slide** for the resulting node |
| Retrieval unit | chunk, leaf with text, document, snippet, part, section | **chunk** in the pipeline, **document** only in Elasticsearch, **section** for heading splits. `ExerciseSnippet` → `Exercise`; the pydantic `Exercise` → `ExerciseSpan` |
| Cut names | `whole_document`, `per_child`, `per_exercise`, `text` | `whole`, `per_child`, `per_exercise`, `per_section`, as a `StrEnum Cut`. `text` also clashes with `Resource.text` and the `text()` function |
| Run output | manifest, record | **manifest**; "record" only for cache node records |
| Model selection | `get_model(task)` returns a client; task/concern/model | `client_for(task)`; "task" everywhere |
| Folders | `BASE_PATH`, `dir`, `material_dir`, `course_path`, `work_root`, `work_dir`, `CACHE_DIR` | `PROJECTS_DIR` → `project.dir` → `project.dir/work`, `project.dir/output`; per extractor `project.dir/<source name>`, so specs no longer pass a `dir` to each extractor |
| Parts of an exercise | `exercise_parts` as a free string with `"solution" in parts` checks | `ExerciseParts` Literal/StrEnum plus an `includes_solution` property |
| Ed Discussion | `categories` (really thread types), `subtype` for message roles | `thread_types`, `role` |

## 6. Smaller simplifications by module

- **`cut.py`:**
  - Cut functions take the whole transformer only to reach `cache` and
    `stitch_fallback`. Make them methods, or pass a small context object.
  - `has_solution` is derived in two places (in `per_exercise` and in a pass
    afterwards). One `_mark_solutions(node)` is enough.
  - One chunk-size limit, defined once, shared by cut and embed (fixes the bug
    in §2).
- **`judge.py`:** after `Labels`, the three passes become three short loops.
  The docstrings should match what the stub actually does.
- **`normalize.py`:** uses the root logger (`logging.info`). Size limits come
  from `mime_types.size_family`.
- **Specs:**
  - `micro303.py` reads `BASE_PATH` when imported, so `from_code` fails for
    every project whenever that variable is missing. With `project.dir`
    resolved at run time, specs never touch config.
  - MOOC's `include_untagged_documents` + `theory_tag` + `theory_slides_tag`
    → one `untagged: UntaggedDocuments | None`.
- **Moodle:** no request timeouts, root logger, `CONFIG[...]` (gives a
  `KeyError` rather than a clear error) and the `modplural[:-1]` hack.
- **Ed Discussion:**
  - Make it a frozen dataclass with `dir`, raising `SourceUnavailable` like
    the other extractors.
  - Use the step cache instead of its own `processed/*.json` and
    `force_regeneration`.
  - Get its models from the catalogue.
  - Build the catalogue from sibling trees in the same run, or from the
    latest manifest.
- **`llm/client.py`:** `_send` and `_send_streaming` repeat the system-prompt
  handling, error wrapping and thinking-tag stripping. Merge them into one
  completion path, or always stream.
- **Loaders:**
  - Elasticsearch: `bulk()` builds the whole list, including 4096-float
    vectors, in memory; pass a generator instead.
  - Index/alias naming belongs to `ProjectSpec`, so the manifest no longer
    imports from `elasticsearch.py`.
- **Comments:** many modules repeat the design rationale in long docstrings
  and comments. Keep the *why* that is local to the code and leave the
  architecture to one document.

## 7. Packaging and good practice

- **Dependencies:** `bs4` is a dummy package; use `beautifulsoup4`. Add
  `pydantic`, which is imported directly. Drop `black` and use `ruff format`.
- **Tests:** add the missing `[tool.setuptools.packages.find]` rules, or
  explicit `__init__.py` files, together with a test that imports every
  module.
- **CI:** none yet (Phase 0 of the plan); a 15-line GitHub Actions workflow
  running ruff, mypy and pytest.
- **Logging:** module loggers everywhere
  (`logger = logging.getLogger(__name__)`); five modules use the root logger.
- **Constants:** tuples or frozensets instead of mutable lists
  (`DEFAULT_MIME_TYPES`, `C_SOURCES`).
- **Small removals:** the `if __name__ == "__main__"` demo in `encoding.py`,
  the dead `extract_week`, `find_document_links_local_regex`, and the unused
  `mime_types` fields on MOOC and Ed.

## 8. Docs

The six refactor-plan files now partly contradict each other and the code.
Examples: the document id scheme, whether the cache is shared across projects,
what the judge stub does, the minimum slide length, 13 vs 14 Elasticsearch
fields. Fold what is still true into one `ARCHITECTURE.md` (the current
DESIGN.md, corrected), rewrite the README as a short user guide, and move
CONSENSUS, IMPLEMENTATION and THOUGHTS to `refactor-plan/archive/`.

## Suggested order

Each step keeps the tests green.

1. Remove legacy references and fix the bugs in §2 (small, safe).
2. Single config module (`ragtools.config`), specs no longer reading config at
   import, `project.dir` derived at run time.
3. `Resource` helpers (`detach`, `iter_chunks`), `Labels`, `Override`.
4. `Step` interface and the flatter layout (§4, mostly file moves).
5. Cache Merkle helper, recipe version in the folder name, simpler videos (no
   descriptors, slides built in one pass).
6. `ragtools` reorganisation: `prompts/`, `services/`, `files.py`, public
   `pdf_to_markdown` etc. matching EXAMPLES.
7. MOOC: typed labels, collapsed parsers, HTML and quiz code moved into
   `ragtools`.
8. Ed Discussion rework (it is disabled today, so this can wait).
9. Docs.

## Open decisions

1. **Names:** do the terminology choices work, especially `Override`,
   `available_from`/`available_until` and the cut names? The Elasticsearch
   field names would stay as they are unless consumers should see the change
   too.
2. **`Labels` as a nested dataclass** (`node.labels.type`) or flat fields
   declared once through `field(metadata=...)`? Nested is recommended: it is
   more explicit and lets overrides be typed.
3. **Recipe version in the cache folder name:** this departs slightly from
   "content-only keys" (CONSENSUS §6), though it replaces the fingerprint
   mechanism that already departs from it. OK?
4. **Where to start:** steps 1–2 first, then pause for review, or several
   steps per round?
