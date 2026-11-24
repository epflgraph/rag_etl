"""Unit tests for the reads of images: the length bound on answers. Model stubbed."""

from types import SimpleNamespace

import pytest

from ragtools.images import MAX_ANSWER_CHARS, strip_image_targets, transcribe_page, transcribe_slide
from ragtools.llm import AnswerTooLong


def stub_vision(monkeypatch, answer: str | Exception) -> None:
    def vision(*args, **kwargs):
        assert kwargs["max_chars"] == MAX_ANSWER_CHARS
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("ragtools.images.get_model", lambda task: SimpleNamespace(generate=vision))


def test_looped_read_dropped(tmp_path, monkeypatch):
    stub_vision(monkeypatch, AnswerTooLong(f"the answer was cut off at {MAX_ANSWER_CHARS:,} chars"))
    assert transcribe_page(tmp_path / "page.png") == ""


def test_real_transcription_kept(tmp_path, monkeypatch):
    answer = "# Lithography\n\n- spin coat\n- expose\n- develop"
    stub_vision(monkeypatch, answer)
    assert transcribe_slide(tmp_path / "slide.png") == answer


@pytest.mark.parametrize(
    "transcript, expected",
    [
        # What the model invents where a picture's address would go
        ("![A QR code](https://example.com/qr.png)", "![A QR code]"),
        ("![A logo](image_url_placeholder)", "![A logo]"),
        ("![A diagram]()", "![A diagram]"),
        ("![An interferometer](https://commons.wikimedia.org/wiki/File:Mirau_(1).svg)", "![An interferometer]"),
        # A description holding brackets of its own, and two images on one line
        ("| ![Beam [1] hitting a mask](a.png) | ![SEM](b.png) |", "| ![Beam [1] hitting a mask] | ![SEM] |"),
        # What is not an image stays as it is
        ("![A plot of f(x)]", "![A plot of f(x)]"),
        ("See [the notes](https://x.org/notes.pdf) and $f(x) = [a](b)$", "See [the notes](https://x.org/notes.pdf) and $f(x) = [a](b)$"),
    ],
)
def test_images_are_left_as_their_description_alone(transcript, expected):
    assert strip_image_targets(transcript) == expected


def test_a_transcript_never_carries_an_invented_image_link(tmp_path, monkeypatch):
    stub_vision(monkeypatch, "# Title\n\n![A QR code](https://example.com/qr.png)\n\ntext")
    assert transcribe_page(tmp_path / "page.png") == "# Title\n\n![A QR code]\n\ntext"
