"""Unit tests for the shared folder walk."""

from ragtools.files import iter_files


def test_iter_files_skips_hidden_and_os_metadata(tmp_path):
    (tmp_path / "notes.md").write_text("notes")
    (tmp_path / "week1").mkdir()
    (tmp_path / "week1" / "slides.pdf").write_bytes(b"pdf")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "._notes.md").write_bytes(b"x")
    (tmp_path / "week1" / ".ipynb_checkpoints").mkdir()
    (tmp_path / "week1" / ".ipynb_checkpoints" / "tp-checkpoint.ipynb").write_text("{}")

    found = {path.relative_to(tmp_path).as_posix() for path in iter_files(tmp_path)}

    assert found == {"notes.md", "week1/slides.pdf"}
