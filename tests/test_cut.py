"""Unit tests for the cuts: synthetic docs, the splitter stubbed."""

from pathlib import Path

import pytest

import ragtools.mime_types as mt
from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.cut.cut_transformer import CutTransformer
from ragtools.documents import MAX_SECTION_LINES, Exercise, ExerciseList, ExerciseSnippet, split_exercises, stitch


@pytest.fixture
def transformer(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "cache")}, raising=False)
    (tmp_path / "cache").mkdir()

    return CutTransformer(cache=Cache("micro303", "cut"))


def write_page(tmp_path: Path, name: str, content: str) -> Path:
    page = tmp_path / name
    page.write_text(content, encoding="utf-8")
    return page


def test_whole_document_leaf_reads_its_file(tmp_path, transformer):
    md = write_page(tmp_path, "notes.md", "# Notes\nsome text")
    root = Resource(title="root")
    leaf = root.add_child(title="notes", path=md, mime_type=mt.MARKDOWN, cut="whole_document")

    transformer.transform([root])

    assert leaf.text == "# Notes\nsome text"
    assert leaf.is_leaf


def test_whole_document_container_stitches_its_pages(tmp_path, transformer):
    root = Resource(title="root")
    doc = root.add_child(title="doc", cut="whole_document")
    for i in range(1, 3):
        doc.add_child(title=f"page {i}", path=write_page(tmp_path, f"page_{i}.md", f"page {i} text"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert doc.text is not None
    assert "<!-- page 1 -->" in doc.text and "<!-- page 2 -->" in doc.text
    assert doc.is_leaf


def test_whole_document_size_guard_drops(tmp_path, transformer):
    root = Resource(title="root")
    doc = root.add_child(title="big", cut="whole_document")
    lines = "\n".join(f"line {i}" for i in range(MAX_SECTION_LINES + 1))
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", lines), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert doc not in root.children


def test_whole_document_with_nothing_readable_drops(tmp_path, transformer):
    root = Resource(title="root")
    doc = root.add_child(title="empty", cut="whole_document")

    transformer.transform([root])

    assert doc not in root.children


def test_per_child_reads_each_readable_child(tmp_path, transformer):
    root = Resource(title="root")
    doc = root.add_child(title="doc", cut="per_child")
    for i in range(1, 3):
        doc.add_child(title=f"page {i}", path=write_page(tmp_path, f"page_{i}.md", f"page {i}"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert [child.text for child in doc.children] == ["page 1", "page 2"]
    assert all(child.cut is None for child in doc.children)


def test_per_child_dropping_one_child_keeps_the_next(tmp_path, transformer):
    root = Resource(title="root")
    doc = root.add_child(title="doc", cut="per_child")
    doc.add_child(title="empty", path=write_page(tmp_path, "empty.md", "   "), mime_type=mt.MARKDOWN)
    doc.add_child(title="good", path=write_page(tmp_path, "good.md", "good text"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert [child.text for child in doc.children] == ["good text"]


def test_per_child_keeps_children_with_their_own_cut(tmp_path, transformer):
    md = write_page(tmp_path, "notes.md", "# A\nalpha\n# B\nbeta")
    root = Resource(title="root")
    folder = root.add_child(title="folder", cut="per_child")
    notes = folder.add_child(title="notes", path=md, mime_type=mt.MARKDOWN, cut="text")

    transformer.transform([root])

    assert notes.cut == "text"
    # The small sections merged into one part
    assert len(notes.children) == 1


def test_per_exercise_cuts_at_markers(tmp_path, transformer, monkeypatch):
    """A sheet cut in 2n: statement and solution chunks, all has_solution."""
    snippets = [
        ExerciseSnippet(number="1", parts="statement", text="<!-- page 1 -->\nExercise 1"),
        ExerciseSnippet(number="1", parts="solution", text="<!-- page 1 -->\nSolution 1"),
    ]
    monkeypatch.setattr("rag_etl.transformers.cut.cut_transformer.split_exercises", lambda *args, **kwargs: snippets)

    root = Resource(title="root")
    doc = root.add_child(title="sheet", cut="per_exercise", url="https://moodle/x.pdf")
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", "Exercise 1"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert [child.number for child in doc.children] == ["1", "1"]
    assert [child.exercise_parts for child in doc.children] == ["statement", "solution"]
    assert [child.title for child in doc.children] == ["Exercise 1 (statement)", "Exercise 1 (solution)"]
    # Both chunks of the solved exercise carry has_solution, not just the solution one
    assert all(child.has_solution for child in doc.children)
    assert doc.children[0].url == "https://moodle/x.pdf#page=1"


def test_per_exercise_statement_only_sheet_has_no_solution(tmp_path, transformer, monkeypatch):
    """A sheet with only statements: every chunk is a statement, has_solution False."""
    snippets = [ExerciseSnippet(number="1", parts="statement", text="Exercise 1")]
    monkeypatch.setattr("rag_etl.transformers.cut.cut_transformer.split_exercises", lambda *args, **kwargs: snippets)

    root = Resource(title="root")
    doc = root.add_child(title="sheet", cut="per_exercise")
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", "Exercise 1"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    [chunk] = doc.children
    assert chunk.exercise_parts == "statement"
    assert not chunk.has_solution
    assert chunk.title == "Exercise 1 (statement)"


def test_per_exercise_solution_only_sheet(tmp_path, transformer, monkeypatch):
    """A corrigé: every chunk is a solution and has_solution."""
    snippets = [ExerciseSnippet(number="1", parts="solution", text="Solution 1")]
    monkeypatch.setattr("rag_etl.transformers.cut.cut_transformer.split_exercises", lambda *args, **kwargs: snippets)

    root = Resource(title="root")
    doc = root.add_child(title="corrigé", cut="per_exercise")
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", "Solution 1"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    [chunk] = doc.children
    assert chunk.exercise_parts == "solution"
    assert chunk.has_solution
    assert chunk.title == "Exercise 1 (solution)"


def test_per_exercise_joined_statement_and_solution(tmp_path, transformer, monkeypatch):
    """A sheet cut in n: each chunk holds statement and solution together."""
    snippets = [ExerciseSnippet(number="1", parts="statement + solution", text="Exercise 1\nSolution 1")]
    monkeypatch.setattr("rag_etl.transformers.cut.cut_transformer.split_exercises", lambda *args, **kwargs: snippets)

    root = Resource(title="root")
    doc = root.add_child(title="sheet", cut="per_exercise")
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", "Exercise 1"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    [chunk] = doc.children
    assert chunk.exercise_parts == "statement + solution"
    assert chunk.has_solution
    assert chunk.title == "Exercise 1 (statement + solution)"


def test_solution_file_chunks_catch_up_has_solution(tmp_path, transformer):
    """A per_child chunk inheriting a solution file's parts counts as has_solution."""
    root = Resource(title="root")
    folder = root.add_child(title="corrigé", exercise_parts="solution", cut="per_child")
    child = folder.add_child(title="1", path=write_page(tmp_path, "1.md", "Solution 1"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert child.exercise_parts == "solution"
    assert child.has_solution


def test_per_exercise_drops_when_no_exercises(tmp_path, transformer, monkeypatch):
    monkeypatch.setattr("rag_etl.transformers.cut.cut_transformer.split_exercises", lambda *args, **kwargs: [])

    root = Resource(title="root")
    doc = root.add_child(title="sheet", cut="per_exercise")
    doc.add_child(title="page 1", path=write_page(tmp_path, "page_1.md", "no exercises here"), mime_type=mt.MARKDOWN)

    transformer.transform([root])

    assert doc not in root.children


def test_text_cut_chunks_by_headings(tmp_path, transformer):
    section = "\n".join(f"line {i}" for i in range(1500))
    md = write_page(tmp_path, "notes.md", f"# Doc\n\n## A\n{section}\n\n## B\n{section}")
    root = Resource(title="root")
    leaf = root.add_child(title="notes", path=md, mime_type=mt.MARKDOWN, cut="text")

    transformer.transform([root])

    # Two 1500-line sections merge to 3000 > the guard, so they stay apart;
    # the intro joins the first one
    assert len(leaf.children) == 2
    assert [child.text for child in leaf.children] == [f"# Doc\n## A\n{section}", f"## B\n{section}"]


def test_text_cut_merges_small_sections(tmp_path, transformer):
    md = write_page(tmp_path, "notes.md", "# Doc\nintro\n## A\nalpha\n## B\nbeta")
    root = Resource(title="root")
    leaf = root.add_child(title="notes", path=md, mime_type=mt.MARKDOWN, cut="text")

    transformer.transform([root])

    assert len(leaf.children) == 1


def test_unknown_cut_fails_loud(transformer):
    root = Resource(title="root")
    root.add_child(title="doc", cut="per_paragraph")

    with pytest.raises(ValueError):
        transformer.transform([root])


def test_stitch_joins_split_tables():
    pages = ["| a | b |\n| 1 | 2 |", "| 3 | 4 |"]

    text = stitch(pages)

    # No marker between the joined rows: it would break the table
    assert "| 2 |\n| 3 |" in text


def test_stitch_joins_split_equations():
    pages = ["$$\nx = 1", "= y\n$$"]

    text = stitch(pages)

    assert "$$\nx = 1\n= y\n$$" in text


def test_stitch_joins_hyphenated_words():
    pages = ["the contin-", "uation of the text"]

    text = stitch(pages)

    assert "continuation of the text" in text


def test_stitch_keeps_markers_between_pages():
    pages = ["first page", "second page"]

    text = stitch(pages)

    assert "<!-- page 1 -->" in text and "<!-- page 2 -->" in text


def test_split_exercises_merges_duplicate_exercises_on_separate_lines(monkeypatch):
    """An exercise cut across sections reassembles in order, parts on separate lines."""

    class FakeLLM:
        def structured(self, prompt, schema, **kwargs):
            return ExerciseList(exercises=[Exercise(start_line=1, end_line=2, number="1", parts="statement")])

    monkeypatch.setattr("ragtools.documents.exercises.get_model", lambda task: FakeLLM())

    # A first section too big to merge into the second: both get annotated
    filler = "\n".join(f"line {i}" for i in range(MAX_SECTION_LINES))
    text = f"# A\n{filler}\n\n# B\nbeta"

    snippets = split_exercises(text)

    assert len(snippets) == 1
    assert snippets[0].text == "# A\nline 0\n# B\nbeta"
