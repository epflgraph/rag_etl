"""Unit tests for materialize: PDF pages, images, videos. Models are stubbed."""

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

import ragtools.mime_types as mt
from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.materialize.materialize_transformer import MaterializeTransformer


class FakeResolver:
    def __init__(self, url: str | None = "https://streaming.epfl.ch/video.mp4"):
        self.url_value = url
        self.requested: list[str] = []

    def url(self, entry_id: str) -> str | None:
        self.requested.append(entry_id)
        return self.url_value


@pytest.fixture
def transformer(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_etl.config.CONFIG", {"CACHE_DIR": str(tmp_path / "cache")}, raising=False)
    (tmp_path / "cache").mkdir()

    llm = SimpleNamespace(vision=lambda *args, **kwargs: "slide text")
    detector = SimpleNamespace(detect=lambda *args, **kwargs: [0, 40, 80])

    fake_resolver = FakeResolver()
    monkeypatch.setattr(
        "rag_etl.transformers.materialize.materialize_transformer.KalturaVideoResolver", lambda: fake_resolver
    )
    monkeypatch.setattr(
        "rag_etl.transformers.materialize.materialize_transformer.extract_frame", fake_extract_frame
    )

    return MaterializeTransformer(cache=Cache("materialize"), llm=llm, vision_model="fake", detector=detector)


def fake_extract_frame(video_url: str, seconds: int, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"jpeg fake")
    return True


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_pdf_gains_one_child_per_page(tmp_path, transformer, monkeypatch):
    pdf_path = make_pdf(tmp_path / "lecture.pdf", ["alpha", "beta"])
    root = Resource(
        title="root",
        children=[Resource(title="lecture", path=pdf_path, mime_type=mt.PDF, week=3, type="theory", cut="whole_document")],
    )

    transformer.transform([root])

    lecture = root.children[0]
    assert [child.title for child in lecture.children] == ["lecture > page 1", "lecture > page 2"]
    assert all(child.mime_type == mt.MARKDOWN for child in lecture.children)
    # The page children carry the parent's descriptive metadata
    assert all(child.week == 3 and child.type == "theory" for child in lecture.children)
    # And the rendered pages are on disk next to the pdf
    assert (tmp_path / "lecture" / "page_1.png").exists()
    assert (tmp_path / "lecture" / "page_1.md").exists()


def test_pdf_one_changed_page_reOCRs_one_page(tmp_path, transformer, monkeypatch):
    """One comma in 200 pages re-OCRs one page: per-page cache keys on the rendered page."""
    pdf_path = make_pdf(tmp_path / "lecture.pdf", ["alpha", "beta"])
    root = Resource(title="root", children=[Resource(title="lecture", path=pdf_path, mime_type=mt.PDF)])
    transformer.transform([root])

    calls = []
    counting_llm = SimpleNamespace(vision=lambda *args, **kwargs: calls.append(kwargs.get("name")) or "slide text")
    counting = MaterializeTransformer(cache=Cache("materialize"), llm=counting_llm, vision_model="fake", detector=transformer.detector)

    make_pdf(pdf_path, ["alpha", "beta, slightly changed"])
    root = Resource(title="root", children=[Resource(title="lecture", path=pdf_path, mime_type=mt.PDF)])
    counting.transform([root])

    assert len(calls) == 1


def test_image_read_in_place(tmp_path, transformer):
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"png fake")
    root = Resource(title="root", children=[Resource(title="slide", path=image_path, mime_type=mt.PNG)])

    transformer.transform([root])

    slide = root.children[0]
    assert slide.mime_type == mt.MARKDOWN
    assert slide.path == image_path.with_suffix(".md")


def test_unreadable_image_dropped(tmp_path, transformer):
    empty = MaterializeTransformer(
        cache=transformer.cache,
        llm=SimpleNamespace(vision=lambda *args, **kwargs: ""),
        vision_model="fake",
        detector=transformer.detector,
    )
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"png fake")
    root = Resource(title="root", children=[Resource(title="slide", path=image_path, mime_type=mt.PNG)])

    empty.transform([root])

    assert root.children == []


def test_video_gains_one_slide_child_per_segment(tmp_path, transformer, monkeypatch):
    descriptor_path = tmp_path / "0_abc.json"
    descriptor_path.write_text('{"entry_id": "0_abc"}', encoding="utf-8")
    srt_path = tmp_path / "0_abc_fr.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:10,000\nhello\n\n2\n00:00:45,000 --> 00:00:60,000\nworld\n",
        encoding="utf-8",
    )

    video = Resource(title="lecture 1", url="https://mediaspace.epfl.ch/watch/0_abc", path=descriptor_path, mime_type=mt.MP4)
    video.add_child(Resource(path=srt_path, mime_type=mt.SRT))
    root = Resource(title="root", children=[video])

    transformer.transform([root])

    lecture = root.children[0]
    # Two slide segments: [0, 40) and [40, end); each frame is named after
    # the second its interval starts at, which is where the title reads back
    assert [child.title for child in lecture.children] == ["lecture 1 > 0:00", "lecture 1 > 0:40"]
    # The words spoken over each slide were appended as a quote
    assert "hello" in lecture.children[0].path.read_text(encoding="utf-8")
    assert "world" in lecture.children[1].path.read_text(encoding="utf-8")
    # The caption leaf is consumed: the subtitles are part of the slides now
    assert all(child.mime_type == mt.MARKDOWN for child in lecture.children)
    # Slides carry the deep link into their own moment and a per-slide cut
    assert lecture.children[0].url.startswith("https://")
    assert lecture.children[0].cut == "whole_document"


def test_youtube_video_left_untouched(tmp_path, transformer):
    video_xml = tmp_path / "1_2_1.xml"
    video_xml.write_text("<video/>", encoding="utf-8")
    video = Resource(title="mooc video", url="https://www.youtube.com/watch?v=abc", path=video_xml, mime_type=mt.MP4)
    root = Resource(title="root", children=[video])

    transformer.transform([root])

    # No Kaltura entry id: the node passes through untouched
    assert root.children == [video]


def test_video_with_no_frames_dropped(tmp_path, transformer):
    one_timestamp = MaterializeTransformer(
        cache=transformer.cache,
        llm=transformer.llm,
        vision_model="fake",
        detector=SimpleNamespace(detect=lambda *args, **kwargs: [0]),
    )
    descriptor_path = tmp_path / "0_abc.json"
    descriptor_path.write_text('{"entry_id": "0_abc"}', encoding="utf-8")
    video = Resource(title="lecture 1", url="https://mediaspace.epfl.ch/watch/0_abc", path=descriptor_path, mime_type=mt.MP4)
    root = Resource(title="root", children=[video])

    one_timestamp.transform([root])

    assert root.children == []
