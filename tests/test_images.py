"""Unit tests for the reads of images: the length bound on answers. Model stubbed."""

from types import SimpleNamespace

from ragtools.images import MAX_ANSWER_CHARS, alt_text, ocr_image
from ragtools.llm import AnswerTooLong


def stub_vision(monkeypatch, answer: str | Exception) -> None:
    def vision(*args, **kwargs):
        assert kwargs["max_chars"] == MAX_ANSWER_CHARS
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("ragtools.images.get_model", lambda task: SimpleNamespace(vision=vision))


def test_looped_read_dropped(tmp_path, monkeypatch):
    stub_vision(monkeypatch, AnswerTooLong(f"the answer was cut off at {MAX_ANSWER_CHARS:,} chars"))
    assert ocr_image(tmp_path / "page.png") == ""


def test_real_transcription_kept(tmp_path, monkeypatch):
    answer = "# Lithography\n\n- spin coat\n- expose\n- develop"
    stub_vision(monkeypatch, answer)
    assert ocr_image(tmp_path / "page.png") == answer


def test_looped_alt_text_dropped(tmp_path, monkeypatch):
    stub_vision(monkeypatch, AnswerTooLong(f"the answer was cut off at {MAX_ANSWER_CHARS + 1:,} chars"))
    assert alt_text(tmp_path / "figure.png") == ""
