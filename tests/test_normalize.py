"""Unit tests for normalize: archives, notebooks and size limits. No network."""

import json
import tarfile
import zipfile

from pathlib import Path
from types import SimpleNamespace

import pytest

import ragtools.mime_types as mt
from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.normalize.normalize_transformer import NormalizeTransformer


@pytest.fixture
def transformer(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "cache")}, raising=False)
    (tmp_path / "cache").mkdir()
    return NormalizeTransformer(cache=Cache("normalize"), llm=SimpleNamespace(), vision_model="fake")


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
    assert [child.title for child in bundle.children] == ["bundle > lecture.pdf", "bundle > notes.txt"]
    assert [child.mime_type for child in bundle.children] == [mt.PDF, mt.TXT]


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
    assert notebook.path == notebook_path.with_suffix(".md")


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
