# AGENTS.md

Guidance for AI agents working on this repository. Read README.md first for
what the tool does and how it is run; this file covers why it is shaped the
way it is, the rules the code follows, and what is still open.

## Context

CEDE (EPFL) runs two kinds of educational tools on course material through
RAG: course chatbots and forum bots (Tutor Bot answers on Ed Discussion).
They retrieve from one Elasticsearch index per project through an MCP server.
Projects are mostly courses, but also labs, admin chatbots (`lex`,
`servicedesk`) and other tools: nothing here may assume a course.

`rag-etl` replaces two systems: `rag-prepipeline` (which this repository used
to be: it prepared files and a metadata file) and `chatbot-pipelines` (which
OCR'd, chunked, translated and loaded them through GraphAI and a MySQL cache).
It now runs end to end. Goals that still steer every change:

- **Everything in-house**: models on RCP, files on the same server's disk, no external data services.
- **Not feature parity**: a behaviour may change or go as long as the consumers keep working. Question old behaviour before porting it.
- **Simple, modern, modular**: one way to do each thing, no duplication, existing libraries where reasonable.
- **`ragtools` is reusable on its own**: colleagues pip install the package for OCR, notebook conversion or splitting.

People: Aitor (maintainer, pipeline), Javier (the judge's inference, Tutor
Bot). Anna's dashboard reads the manifests directly.

## Working agreements

- **Propose before changing.** For anything beyond a local fix, lay out what is wrong, why, and what you plan; wait for agreement. One item at a time.
- **Do not run git** (commit, branch, stash, checkout). The maintainer handles git; suggest a one-line commit message in the repository's style (`:recycle: …`, `:sparkles: …`, `:bug: …`).
- **Edit files with the editor tools**, not with sed/perl one-liners. Move files one at a time.
- **Fail loud, where the value is used.** Read settings with `os.environ["X"]` (required) or `.get` (optional) at the point of use; no config layer decides what is required. A source that cannot be reached raises `SourceUnavailable`; a model call is retried by the client, then raises. Nothing is silently skipped, except the documented cases below.
- **Keep the filesystem's order.** Do not sort files or children to look tidy; the order a source gives is the order of the material.
- **Group by meaning**: prompts as `.md` files next to the module that uses them (`ragtools/images/prompts/`, `ragtools/documents/prompts/`), no global folders of one kind of thing.
- **Extractors state facts; the judge and the overrides decide.** No per-course labelling logic in extractors, no tag tables.
- **Per-project quirks are overrides, not code.** A recurring kind of processing joins the shared code (a new cut goes in the `CUTS` registry) via review.
- **Comments say the local why**, briefly. Architecture belongs here, not repeated in docstrings.
- **Tests**: synthetic fixtures, stubbed models, no network. Keep them green; do not over-invest in them.

Commands: `venv/bin/python -m pytest -q`, `venv/bin/ruff check src tests`.

## Layout

```
src/
├── ragtools/                 file preparation; never imports rag_etl (tests/test_layering.py)
│   ├── llm/                  rcp.py (RCPClient: completions, structured output, embeddings, Langfuse tracing)
│   │                         catalog.py (MODELS keyed (model, mode); get_model(task))
│   ├── images/               transcribe_page, transcribe_slide, alt_text (+ prompts/)
│   ├── documents/            headings (split/merge sections), stitch, exercises (split_exercises) (+ prompts/)
│   ├── pdf.py                deterministic page rendering (PyMuPDF)
│   ├── video.py              ffmpeg frames, SRT parsing, SlideDetector interface
│   ├── graphai.py            GraphAI client + GraphAISlideDetector
│   ├── kaltura.py            Kaltura client, session_from_env, KalturaVideoResolver
│   ├── html.py               html_to_markdown + the shared text clean-ups
│   ├── edx_quiz.py           parse_quiz, quiz_to_markdown (Open edX single/multiple choice)
│   ├── archives.py, notebooks.py, files.py (iter_files), encoding.py, mime_types.py
└── rag_etl/
    ├── core/                 resource.py (Resource, METADATA_FIELDS, iter_chunks, assign_ids), overrides.py
    ├── extractors/           base.py (Extractor, SourceUnavailable), markers.py (NO_BOT),
    │                         moodle.py, local_folder.py, git.py, mediaspace.py, mooc/, ed_discussion/
    ├── transformers/         base.py, normalize.py, materialize/ (facade + pdf, image, video), judge.py, cut.py, embed.py
    ├── loaders/              base.py, manifest.py, elasticsearch.py, index_settings.json
    ├── projects/             Project, load_project; semester_2627_autumn/<project>.py
    ├── cache.py, pipeline.py, cli.py
legacy/                       pre-rewrite code, reference only, not packaged; delete after the consumer cutover
```

## The pipeline

`pipeline.run_pipeline(project, dry, refresh)`: extract → normalize →
materialize → judge → cut → embed → load. One fixed sequence; projects never
list steps. The contracts:

- `Extractor.extract() -> Resource` — one root per source. Extractors are frozen dataclasses with a `dir` (relative to the project folder, resolved by the pipeline with `dataclasses.replace`).
- `Transformer.transform(roots) -> roots` — frozen dataclasses with a `name` (ClassVar) that names their cache and their `--refresh` value. Trees are reshaped in place.
- `Loader.load(roots) -> None` — frozen dataclasses configured by their fields.

The work area (`<project>/work/`) is scratch, wiped at every run start:
everything in it can be rebuilt from the sources and the cache without a
model call. A dry run does everything except publishing (model calls
included) and writes a `_dry` manifest.

## The resource tree

One recursive dataclass, `Resource` (`core/resource.py`), for every level:
a course, a folder, a PDF, a page, a video, a slide, a chunk. No per-level
classes and no kind field: container, leaf and chunk are states (`is_leaf`
is "no children", `is_chunk` is "has text").

- **Fields**: identity (`title`, `path`, `url`, `mime_type`, `id`); lineage (`parent`, `children`); processing (`cut`, never indexed); descriptive metadata, declared once in `METADATA_FIELDS` (`type`, `subtype`, `number`, `subnumber`, `week`, `exercise_parts`, `has_solution`, `from_`, `until`); content (`text`, `embedding`).
- **Data, not policy.** A resource says what it is; what to do with it is the judge's, the overrides' and the cuts'. Processing flags in the data (`one_chunk_per_page`, `processing_method`, …) are what the rewrite removed.
- **Identity is the node** (`eq=False`). `content_hash` is computed on every access (text, else children's hashes in order, else file bytes), so it never goes stale.
- **`add_child(**fields)`** builds a child as a copy of its parent, naming only what differs: a page keeps its PDF's url and labels. `add_child(child)` attaches a subtree as it is. `detach()` removes a node and its subtree.
- **Deep links travel as `url`**, copied down and refined (a page's `#page=N`, a slide's timestamp). Loaders read `node.url`; nothing re-resolves lineage.
- **Nesting is the only relation** between resources. Links between trees (a video and its deck) are not modelled.
- **Chunk ids** are assigned once the trees are final (`assign_ids`, at load): `<position in the run>-<content hash>`, unique within a run, not stable across runs. Equal content under different metadata stays two chunks: courses own their duplicates.

## Projects and overrides

A project is a frozen `Project(id, language, overrides, sources, dir)` in
`projects/semester_<season>/<id>.py`, defined under its id in capitals;
`load_project(id)` finds it by convention, no registry. Nothing in a project
file reads the environment: `dir` is resolved against `$BASE_PATH` at run
time. Models are never part of a project.

`Override(where, cut=None, exclude=False, **metadata)` (`core/overrides.py`)
is the only manual channel:

- `where`: flat `field="glob"` predicates joined by `AND` over `path`, `filename`, `mime_type`, `title`, `url`, `cut`, `is_leaf`, `text` and the metadata fields. fnmatch globs; a literal `[` is `[[]`. No nesting, no OR: needing them means the case wants code.
- Applied by the judge after inheritance, on top of its labels: the human wins. Most predicates wins; at equal specificity the first declared wins, with a warning. An unknown field name fails when the project loads.
- `exclude=True` drops the matching resources and their subtrees (and containers left empty); it wins over everything and sets nothing else.
- Overrides are pruned when a review shows one is no longer needed.

## Steps in detail

**Extractors.** Return everything includable; professors exclude with a
`NO_BOT` marker (Moodle, file and folder names, MOOC pages).

- *Moodle*: sections → modules → files; availability dates become `from_`/`until`. Assignments cannot be fetched (no API access) — accepted.
- *MOOC* (edX export): course → chapters (weeks, from `first_week_chapter`/`week_count`) → sequentials → verticals → pages, linked documents, quizzes (statement and statement + solution files), videos with their subtitles. Documents linked on a video page are skipped as duplicates of the video's slides. A video naming no source stops the run.
- *Mediaspace*: a channel or playlist; videos with their Kaltura captions. Videos have no `path`; overrides reach them by `url` (`url="https://mediaspace.epfl.ch/*"`).
- *Local folder*, *Git* (repo or branch url; everything indexed by default).
- *Ed Discussion*: threads filed under Ed's categories and subcategories, one markdown Q&A per answered thread (staff answers; endorsed student answers if asked). Which categories to keep and how to label them is done by overrides.

**Normalize.** zip and tar.gz unpack into children (UTF-8 names fixed),
notebooks become markdown (alt texts by a vision model), files over their
family's limit are dropped (PDF 50 MB, images 10 MB, text 5 MB).

**Materialize** (facade over pdf, image and video materializers).

- *PDF*: each page rendered deterministically (stable settings: the render is what gets hashed) and transcribed by the vision model; one child per page. No text-layer shortcut.
- *Image*: transcribed in place.
- *Video*: one path for Mediaspace and MOOC videos. Kaltura entry id from the url → slides entry → GraphAI `detect_slides` (behind `SlideDetector`, swappable) → changes closer than `min_slide_seconds` dropped → one ffmpeg frame per segment, read from the url (never downloaded) → each slide transcribed, with the subtitles spoken over it appended as a quote; a slide links to its moment. GraphAI failing on one video drops that video, not the run. YouTube videos have no Kaltura entry and pass through.

**Judge.** Fills `type`, `subtype` and `cut` for every resource, or drops it.
How it decides is Javier's work; until it lands, `judge.py` is a deterministic
stub: default cuts (readable leaves whole, containers of text per child,
roots never), labels from title cues (practice vs theory, a subtype, a
series number), metadata inherited down the tree, overrides, then fallbacks
(`theory`/`unknown`). There is deliberately no type/subtype → cut mapping in
the code. Chunk-level facts (`exercise_parts`, `has_solution`) are never
inherited. The planned inference has three stages: infer each document's type
(theory, practice, exam, organization); an LLM jury (three models voting)
infers subtype, number and `has_solution` for every practice/exam document of
a course; then the course's subtype labels are normalized. Like every other
step it should use the cache, so unchanged files are not judged again.

**Cut** (`CUTS` registry in `cut.py`; an unknown cut fails). A cut consumes
its node, top-down, replacing it by chunk leaves.

- `whole_document`: one chunk, pages stitched.
- `per_child`: each readable child becomes a chunk (pages, slides); a child with its own different cut keeps it.
- `per_exercise`: pages stitched, split by `split_exercises` (one LLM pass, cached by the text's hash: number and statement/solution per exercise), one chunk per exercise linking to its page; `has_solution` set across the exercise.
- `text`: split at the most common heading level, sections merged up to the size limit.

Stitching joins pages only where a file is merged (never slides),
deterministically first (page markers, tables and equations split across
pages, hyphenation); an LLM stitch exists as a per-document fallback, off by
default. One size limit, `MAX_CHUNK_CHARS` (`embed.py`), applies to every
chunk: the cut drops a bigger one with a warning.

**Embed.** Qwen3-Embedding-8B on RCP (4096 dims), batched, cached per text
hash, written to `Resource.embedding`.

**Load.**

- *Manifest* (always): `output/<timestamp>_manifest[_dry].json`, for people (inspection, Anna's dashboard, the MCP server's overview); nothing in the pipeline reads it. Header (project, dry, timestamp, index, alias, version), a summary drilling down source → type → subtype → number → subnumber → parts with per-group coverage of url/week/from/until, and the trees without text or vectors (content hashes and chunk ids instead).
- *Elasticsearch* (not on dry runs): create `rag_<id>_<YYYYMMDD>_<HHMMSS>` from `index_settings.json` (content analyzer set per language), bulk-load, refresh, then one atomic alias update moves `rag_<id>`. The alias's index is never deleted; older indices stay for rollback. TLS verified with `ES_CAFILE`.

The document schema (14 fields), the contract with the consumers: `id`,
`content`, `embedding` (dense_vector 4096, dot_product, int8_hnsw), `lineage`
(titles root → chunk), `url` (own or nearest ancestor's, absent if none),
`type`, `subtype`, `number`, `subnumber`, `week`, `exercise_parts`,
`has_solution`, `from`, `until`. Single language, no translation. Nothing
project-level per chunk: the project travels in the index name. Release
windows are filtered by consumers at query time, never at load. Search is
hybrid (vectors + text); query vectors must be unit-normalized.

## Cache and models

**Cache** (`cache.py`): on disk under `$CACHE_DIR/<project>/<step>/`, keyed
by content hash only, values are files or directories, no eviction.

- No model, prompt or setting in the key. A change is applied deliberately: `--refresh <step>` reruns one step, or a materializer bumps its recipe version (`CACHE_VERSION`, e.g. `pdf-ocr-1`), which starts a fresh scope.
- PDFs and videos cache as Merkle trees: transcripts are leaves under their own hash, and the parent keeps a small node record (`children.json`) of its ordered leaves. A node hit adopts the whole subtree without rendering; a miss falls back to the leaves, so one changed page re-reads one page.

**Models**: every call goes through `RCPClient`. `catalog.py` holds the
settled model + parameter pairings and one entry per task (`ocr`, `judge`,
`split_exercises`, `stitch`, `embedding`); each capability asks
`get_model(task)` itself. Changing a model is a one-line edit there, followed
by a `--refresh`.

## Decided, do not reopen without the maintainer

- Locators and urls are properties, never identity: they churn (Moodle edits, re-uploads).
- No page markers as geometry: pages are child resources.
- OCR stays transcription only: no exercise detection or labelling folded into it.
- No per-project step lists, no escape hatch per project.
- No prompt/model in cache keys; explicit refresh instead.
- Elasticsearch is disposable: every run builds a fresh index; no stable ids, no diff-based updates.
- No translation, no GraphAI embeddings, no MySQL cache, no Airflow (timers call the CLI).
- No dedicated inspection command: the manifest and Kibana cover it.

## Open work

- **The judge's inference** (Javier). It should also absorb what Ed Discussion's former LLM jury did: matching a thread to the document it is about, and its dates and week.
- **YouTube captions** are never indexed (SRT is not text and YouTube videos skip the video path).
- **Ed Discussion**: images in posts are transcribed during extraction; that belongs in materialize.
- **Consumers**: GraphAI's `/rag/retrieve` must embed queries with Qwen3-Embedding-8B and query `content`.
- **Operations**: CI (ruff, pytest) is missing; the weekly timers are not wired; MICRO-303 is the pilot, other projects follow one by one; delete `legacy/` after the cutover.
