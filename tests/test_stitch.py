"""Unit tests for the LLM stitch and its step: the model stubbed, no network."""

import re
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.transformers import StitchTransformer
from ragtools.documents import llm_stitch, split_pages
from ragtools.documents.stitch import BATCH_OVERLAP, BATCH_PAGES, mark_page


def stub_model(monkeypatch, answer) -> list[str]:
    """Replace the stitch model by `answer(marked pages of one call) -> text`; return the calls' prompts."""
    prompts: list[str] = []

    def text(prompt, **kwargs):
        prompts.append(prompt)
        return answer(re.findall(r"```\n(.*?)\n```", prompt, flags=re.DOTALL))

    # The module by its object: the package also exports a function named stitch
    module = sys.modules["ragtools.documents.stitch"]
    monkeypatch.setattr(module, "get_model", lambda task: SimpleNamespace(text=text))
    return prompts


def echo(pages: list[str]) -> str:
    """A model that hands the pages back joined, each heading demoted: visibly stitched, markers kept."""
    return "\n\n".join(page.replace("# ", "## ") for page in pages)


def test_split_pages_is_the_inverse_of_marking():
    pages = [(1, "# Title\n\nintro"), (3, "more")]
    marked = "\n\n".join(mark_page(number, text) for number, text in pages)
    assert split_pages(marked) == pages
    # Text before the first marker belongs to the first page
    assert split_pages(f"preamble\n\n{marked}") == [(1, "preamble\n\n\n\n# Title\n\nintro"), (3, "more")]


def test_llm_stitch_batches_long_documents_and_joins_them_at_a_marker(monkeypatch):
    pages = [(number, f"# Heading {number}") for number in range(1, 2 * BATCH_PAGES + 1)]
    prompts = stub_model(monkeypatch, echo)

    stitched = split_pages(llm_stitch(pages))

    # Every page comes back once, in order, stitched
    assert stitched == [(number, f"## Heading {number}") for number, _ in pages]
    # Consecutive calls share their boundary pages, and none sees more than a batch
    seen = [[int(n) for n in re.findall(r"<!-- page (\d+) -->", prompt)] for prompt in prompts]
    assert all(len(batch) <= BATCH_PAGES for batch in seen)
    assert seen[1][:BATCH_OVERLAP] == seen[0][-BATCH_OVERLAP:]
    assert sorted({number for batch in seen for number in batch}) == [number for number, _ in pages]


def test_llm_stitch_keeps_the_pages_of_a_batch_the_model_gave_nothing_for(monkeypatch):
    stub_model(monkeypatch, lambda pages: "")
    assert split_pages(llm_stitch([(1, "a"), (2, "b")])) == [(1, "a"), (2, "b")]


def pdf_with_pages(tmp_path: Path, texts: dict[int, str]) -> Resource:
    root = Resource(title="root")
    pdf = root.add_child(title="lecture", path=tmp_path / "lecture.pdf", mime_type=mt.PDF)
    for number, text in texts.items():
        path = tmp_path / f"page_{number}.md"
        path.write_text(text, encoding="utf-8")
        pdf.add_child(title=f"page {number}", path=path, mime_type=mt.MARKDOWN)
    return root


@pytest.fixture
def stitcher(tmp_path):
    return StitchTransformer(project_dir=tmp_path / "project")


def test_stitch_harmonises_the_pages_and_keeps_them_children(tmp_path, stitcher, monkeypatch):
    stub_model(monkeypatch, echo)
    root = pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})

    stitcher.transform([root])

    pages = root.children[0].children
    assert [page.title for page in pages] == ["page 1", "page 2"]
    assert [page.path.read_text() for page in pages] == ["## Intro", "## Method"]
    # Written in the step's work folder, the pages as read left alone
    assert all(page.path.is_relative_to(stitcher.work_root) for page in pages)
    assert (tmp_path / "page_1.md").read_text() == "# Intro"


def test_stitch_is_cached_by_the_pages_as_read(tmp_path, stitcher, monkeypatch):
    prompts = stub_model(monkeypatch, echo)
    stitcher.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})])
    stitcher.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})])
    assert len(prompts) == 1

    # A page read differently is another document
    stitcher.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Results"})])
    assert len(prompts) == 2


def test_stitch_keeps_the_pages_as_read_when_a_marker_is_lost(tmp_path, stitcher, monkeypatch, caplog):
    stub_model(monkeypatch, lambda pages: "everything in one piece, no marker left")
    root = pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})

    stitcher.transform([root])

    assert [page.path.read_text() for page in root.children[0].children] == ["# Intro", "# Method"]
    assert "did not keep its page markers" in caplog.text


def test_stitch_drops_a_page_merged_into_its_neighbour(tmp_path, stitcher, monkeypatch):
    # The table cut by the page boundary is merged into page 1: page 2 holds nothing anymore
    merged = f"{mark_page(1, '| a |\n| b |')}\n\n{mark_page(2, '')}\n\n{mark_page(3, 'after')}"
    stub_model(monkeypatch, lambda pages: merged)
    root = pdf_with_pages(tmp_path, {1: "| a |", 2: "| b |", 3: "after"})

    stitcher.transform([root])

    assert [page.title for page in root.children[0].children] == ["page 1", "page 3"]


def test_single_pages_are_not_stitched(tmp_path, stitcher, monkeypatch):
    prompts = stub_model(monkeypatch, echo)
    root = pdf_with_pages(tmp_path, {1: "# Alone"})

    stitcher.transform([root])

    assert prompts == [] and root.children[0].children[0].path == tmp_path / "page_1.md"
