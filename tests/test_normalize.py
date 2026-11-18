"""Unit tests for normalize: archives, notebooks and size limits. No network."""

import json
import tarfile
import zipfile

from pathlib import Path

import pytest

import ragtools.mime_types as mt
from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.normalize import NormalizeTransformer


@pytest.fixture
def transformer(tmp_path, monkeypatch):
    return NormalizeTransformer(cache=Cache(tmp_path / "cache" / "normalize"), work_root=tmp_path / "work" / "normalize")


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zip_ref:
        for name, content in files.items():
            zip_ref.writestr(name, content)
    return path


def test_zip_expands_into_container(tmp_path, transformer):
    zip_path = make_zip(
        tmp_path / "bundle.zip",
        {"lecture.pdf": b"%PDF fake", "notes.txt": b"notes", "junk.bin": b"unsupported"},
    )
    root = Resource(title="root", children=[Resource(title="bundle", path=zip_path, mime_type=mt.ZIP)])

    transformer.transform([root])

    bundle = root.children[0]
    assert [child.title for child in bundle.children] == ["lecture.pdf", "notes.txt"]
    assert [child.mime_type for child in bundle.children] == [mt.PDF, mt.TXT]


def test_normalize_writes_nothing_beside_the_sources(tmp_path, transformer):
    # A source folder that is not wiped between runs would extract what was
    # written into it as material of its own on the next run
    source = tmp_path / "local"
    source.mkdir()
    zip_path = make_zip(source / "bundle.zip", {"notes.txt": b"notes"})
    notebook_path = source / "tp.ipynb"
    notebook_path.write_text(
        json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8"
    )
    root = Resource(
        title="root",
        children=[
            Resource(title="bundle", path=zip_path, mime_type=mt.ZIP),
            Resource(title="tp", path=notebook_path, mime_type=mt.IPYNB),
        ],
    )

    transformer.transform([root])

    assert sorted(path.name for path in source.iterdir()) == ["bundle.zip", "tp.ipynb"]
    assert all(node.path.is_relative_to(transformer.work_root) for node in root.walk() if node.path and node.is_leaf)


def test_targz_expands_into_container(tmp_path, transformer):
    tar_path = tmp_path / "bundle.tar.gz"
    inner = tmp_path / "notes.txt"
    inner.write_text("notes", encoding="utf-8")
    with tarfile.open(tar_path, "w:gz") as tar_ref:
        tar_ref.add(inner, arcname="notes.txt")

    root = Resource(title="root", children=[Resource(title="bundle", path=tar_path, mime_type=mt.TAR)])

    transformer.transform([root])

    bundle = root.children[0]
    assert len(bundle.children) == 1
    assert bundle.children[0].mime_type == mt.TXT


def test_unsupported_archive_dropped(tmp_path, transformer):
    zip_path = make_zip(tmp_path / "bundle.zip", {"junk.bin": b"unsupported"})
    root = Resource(title="root", children=[Resource(title="bundle", path=zip_path, mime_type=mt.ZIP)])

    transformer.transform([root])

    assert root.children == []


def test_notebook_converted(tmp_path, transformer):
    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Hello"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    root = Resource(
        title="root",
        children=[Resource(title="notebook", path=notebook_path, mime_type=mt.IPYNB)],
    )

    transformer.transform([root])

    notebook = root.children[0]
    assert notebook.mime_type == mt.MARKDOWN
    assert notebook.path.name == "notebook.md"
    assert notebook.path.is_relative_to(transformer.work_root)


def test_notebook_images_still_resolve_from_the_work_area(tmp_path, transformer, monkeypatch):
    monkeypatch.setattr("ragtools.notebooks.alt_text", lambda path: "a plot")
    source = tmp_path / "local"
    (source / "figures").mkdir(parents=True)
    (source / "figures" / "plot.png").write_bytes(b"png fake")
    notebook_path = source / "tp.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["![](figures/plot.png)"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    root = Resource(title="root", children=[Resource(title="tp", path=notebook_path, mime_type=mt.IPYNB)])

    transformer.transform([root])

    md_path = root.children[0].path
    target = md_path.read_text(encoding="utf-8").split("](<", 1)[1].split(">)", 1)[0]
    assert (md_path.parent / target).resolve() == (source / "figures" / "plot.png").resolve()


def test_html_converted(tmp_path, transformer):
    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body><h2>Title</h2><p>Some <b>text</b>.</p></body></html>", encoding="utf-8")
    root = Resource(title="root", children=[Resource(title="page", path=html_path, mime_type=mt.HTML)])

    transformer.transform([root])

    page = root.children[0]
    assert page.mime_type == mt.MARKDOWN
    assert page.path.is_relative_to(transformer.work_root)
    assert page.path.read_text(encoding="utf-8") == "## Title\n\nSome **text** ."


def test_oversized_files_dropped(tmp_path, transformer):
    big_txt = tmp_path / "big.txt"
    big_txt.write_bytes(b"a" * (6 * 1024**2))
    big_pdf = tmp_path / "big.pdf"
    big_pdf.write_bytes(b"a" * (51 * 1024**2))
    small = tmp_path / "small.txt"
    small.write_text("fine", encoding="utf-8")

    root = Resource(
        title="root",
        children=[
            Resource(title="big.txt", path=big_txt, mime_type=mt.TXT),
            Resource(title="big.pdf", path=big_pdf, mime_type=mt.PDF),
            Resource(title="small.txt", path=small, mime_type=mt.TXT),
        ],
    )

    transformer.transform([root])

    assert [child.title for child in root.children] == ["small.txt"]
