"""Unit tests for PDF rendering: each page at the resolution its smallest print needs."""

from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ragtools.pdf import MAX_ZOOM, NO_TEXT_ZOOM, SMALL_PRINT_PX, page_zoom, save_pdf_pages, smallest_font_size


def make_pdf(path: Path, pages: list[list[tuple[str, float]]]) -> Path:
    """A PDF whose pages each hold (text, font size) lines; an empty list is a page without text."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()  # 595 x 842 points
        for position, (text, size) in enumerate(lines):
            page.insert_text((72, 100 + 40 * position), text, fontsize=size)
    doc.save(path)
    doc.close()
    return path


def zoom_of(tmp_path: Path, lines: list[tuple[str, float]]) -> float:
    with pymupdf.open(make_pdf(tmp_path / "page.pdf", [lines])) as doc:
        return page_zoom(doc[0])


def test_the_smallest_text_sets_the_zoom_however_little_of_it_there_is(tmp_path):
    # A body in 10 pt and one subscript in 5 pt: the subscript decides
    with pymupdf.open(make_pdf(tmp_path / "page.pdf", [[("body text " * 20, 10), ("2", 5)]])) as doc:
        assert smallest_font_size(doc[0]) == pytest.approx(5)
        assert page_zoom(doc[0]) == pytest.approx(SMALL_PRINT_PX / 5)


def test_large_print_stays_at_72_dpi(tmp_path):
    assert zoom_of(tmp_path, [("A slide title", 32), ("a bullet", 14)]) == 1.0


def test_tiny_print_stops_at_the_ceiling(tmp_path):
    assert zoom_of(tmp_path, [("body", 10), ("a stray glyph", 1)]) == MAX_ZOOM


def test_a_page_without_text_is_rendered_as_a_scan(tmp_path):
    assert zoom_of(tmp_path, []) == pytest.approx(NO_TEXT_ZOOM, abs=0.01)


def test_each_page_is_rendered_at_its_own_resolution(tmp_path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", [[("large", 24)], [("small", 6)], []])

    paths = save_pdf_pages(pdf_path, tmp_path / "pages")

    assert [path.name for path in paths] == ["page_1.png", "page_2.png", "page_3.png"]
    widths = [Image.open(path).width for path in paths]
    # 595 points wide: as it is, doubled for 6 pt print, and at 150 dpi without text
    assert widths == [595, 1190, round(595 * 150 / 72)]


def test_rendering_is_deterministic(tmp_path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", [[("body", 10), ("subscript", 7)]])

    first = save_pdf_pages(pdf_path, tmp_path / "a")[0].read_bytes()
    second = save_pdf_pages(pdf_path, tmp_path / "b")[0].read_bytes()

    assert first == second


def test_one_dpi_for_all_pages_when_asked(tmp_path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", [[("small", 6)]])
    [path] = save_pdf_pages(pdf_path, tmp_path / "pages", dpi=72)
    assert Image.open(path).width == 595
