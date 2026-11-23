"""Unit tests for harmonizing a document's pages, and its step: the model stubbed, no network."""

import re

from pathlib import Path
from types import SimpleNamespace

import pytest

import ragtools.mime_types as mt
from rag_etl.core import Resource
from rag_etl.transformers import HarmonizeTransformer
from ragtools.documents import harmonize_pages, split_pages
from ragtools.documents.harmonize import BATCH_OVERLAP, BATCH_PAGES
from ragtools.documents.stitch import mark_page


def stub_model(monkeypatch, answer) -> list[str]:
    """Replace the model by `answer(marked pages of one call) -> text`; return the calls' prompts."""
    prompts: list[str] = []

    def text(prompt, **kwargs):
        prompts.append(prompt)
        return answer(re.findall(r"```\n(.*?)\n```", prompt, flags=re.DOTALL))

    monkeypatch.setattr("ragtools.documents.harmonize.get_model", lambda task: SimpleNamespace(generate=text))
    return prompts


def echo(pages: list[str]) -> str:
    """A model that hands the pages back joined, each heading demoted: visibly harmonized, markers kept."""
    return "\n\n".join(page.replace("# ", "## ") for page in pages)


def test_split_pages_is_the_inverse_of_marking():
    pages = [(1, "# Title\n\nintro"), (3, "more")]
    marked = "\n\n".join(mark_page(number, text) for number, text in pages)
    assert split_pages(marked) == pages
    # Text before the first marker belongs to the first page
    assert split_pages(f"preamble\n\n{marked}") == [(1, "preamble\n\n\n\n# Title\n\nintro"), (3, "more")]


def test_long_documents_go_in_batches_joined_at_a_marker(monkeypatch):
    pages = [(number, f"# Heading {number}") for number in range(1, 2 * BATCH_PAGES + 1)]
    prompts = stub_model(monkeypatch, echo)

    harmonized = split_pages(harmonize_pages(pages))

    # Every page comes back once, in order, harmonized
    assert harmonized == [(number, f"## Heading {number}") for number, _ in pages]
    # Consecutive calls share their boundary pages, and none sees more than a batch
    seen = [[int(n) for n in re.findall(r"<!-- page (\d+) -->", prompt)] for prompt in prompts]
    assert all(len(batch) <= BATCH_PAGES for batch in seen)
    assert seen[1][:BATCH_OVERLAP] == seen[0][-BATCH_OVERLAP:]
    assert sorted({number for batch in seen for number in batch}) == [number for number, _ in pages]


def test_a_batch_the_model_gave_nothing_for_keeps_its_pages(monkeypatch):
    stub_model(monkeypatch, lambda pages: "")
    assert split_pages(harmonize_pages([(1, "a"), (2, "b")])) == [(1, "a"), (2, "b")]


def pdf_with_pages(tmp_path: Path, texts: dict[int, str]) -> Resource:
    root = Resource(title="root")
    pdf = root.add_child(title="lecture", path=tmp_path / "lecture.pdf", mime_type=mt.PDF)
    for number, text in texts.items():
        path = tmp_path / f"page_{number}.md"
        path.write_text(text, encoding="utf-8")
        pdf.add_child(title=f"page {number}", path=path, mime_type=mt.MARKDOWN)
    return root


@pytest.fixture
def harmonizer(tmp_path):
    return HarmonizeTransformer(project_dir=tmp_path / "project")


def test_pages_are_harmonized_and_stay_children(tmp_path, harmonizer, monkeypatch):
    stub_model(monkeypatch, echo)
    root = pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})

    harmonizer.transform([root])

    pages = root.children[0].children
    assert [page.title for page in pages] == ["page 1", "page 2"]
    assert [page.path.read_text() for page in pages] == ["## Intro", "## Method"]
    # Written in the step's work folder, the pages as read left alone
    assert all(page.path.is_relative_to(harmonizer.work_root) for page in pages)
    assert (tmp_path / "page_1.md").read_text() == "# Intro"


def test_a_document_is_harmonized_once_for_its_pages_as_read(tmp_path, harmonizer, monkeypatch):
    prompts = stub_model(monkeypatch, echo)
    harmonizer.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})])
    harmonizer.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})])
    assert len(prompts) == 1
    assert (tmp_path / "project" / "cache" / "harmonize-1").is_dir()

    # A page read differently is another document
    harmonizer.transform([pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Results"})])
    assert len(prompts) == 2


def test_pages_stay_as_read_when_a_marker_is_lost(tmp_path, harmonizer, monkeypatch, caplog):
    stub_model(monkeypatch, lambda pages: "everything in one piece, no marker left")
    root = pdf_with_pages(tmp_path, {1: "# Intro", 2: "# Method"})

    harmonizer.transform([root])

    assert [page.path.read_text() for page in root.children[0].children] == ["# Intro", "# Method"]
    assert "did not keep its page markers" in caplog.text


def test_a_page_merged_into_its_neighbour_goes(tmp_path, harmonizer, monkeypatch):
    # The table cut by the page boundary is merged into page 1: page 2 holds nothing anymore
    merged = f"{mark_page(1, '| a |\n| b |')}\n\n{mark_page(2, '')}\n\n{mark_page(3, 'after')}"
    stub_model(monkeypatch, lambda pages: merged)
    root = pdf_with_pages(tmp_path, {1: "| a |", 2: "| b |", 3: "after"})

    harmonizer.transform([root])

    assert [page.title for page in root.children[0].children] == ["page 1", "page 3"]


def test_single_pages_are_left_alone(tmp_path, harmonizer, monkeypatch):
    prompts = stub_model(monkeypatch, echo)
    root = pdf_with_pages(tmp_path, {1: "# Alone"})

    harmonizer.transform([root])

    assert prompts == [] and root.children[0].children[0].path == tmp_path / "page_1.md"
