"""Unit tests for the reads of images: the length bound on answers. Model stubbed."""

from types import SimpleNamespace

from ragtools.images import MAX_ANSWER_CHARS, alt_text, ocr_image


def stub_answer(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(
        "ragtools.images.get_model",
        lambda task: SimpleNamespace(vision=lambda *args, **kwargs: answer),
    )


def test_looped_read_dropped(tmp_path, monkeypatch):
    stub_answer(monkeypatch, "**4 One-dimensional microscanner** " + ". " * 20_000)
    assert ocr_image(tmp_path / "page.png") == ""


def test_real_transcription_kept(tmp_path, monkeypatch):
    answer = "# Lithography\n\n- spin coat\n- expose\n- develop"
    stub_answer(monkeypatch, answer)
    assert ocr_image(tmp_path / "page.png") == answer


def test_looped_alt_text_dropped(tmp_path, monkeypatch):
    stub_answer(monkeypatch, "x" * (MAX_ANSWER_CHARS + 1))
    assert alt_text(tmp_path / "figure.png") == ""
