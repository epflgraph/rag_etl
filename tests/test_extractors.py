"""Unit tests for the extractors: synthetic fixtures, no network."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import rag_etl.utils.mime_types as mt
from rag_etl.extractors.base_extractor import Extractor, SourceUnavailable
from rag_etl.extractors.ed_discussion.ed_discussion_extractor import EdDiscussionExtractor
from rag_etl.extractors.ed_discussion.utils import ThreadClassification
from rag_etl.extractors.git.git_extractor import GitExtractor, branch_from_url, repo_from_url
from rag_etl.extractors.local_folder.local_folder_extractor import LocalFolderExtractor
from rag_etl.extractors.mediaspace.mediaspace_extractor import MediaspaceExtractor
from rag_etl.extractors.mooc.mooc_extractor import MOOCExtractor
from rag_etl.extractors.moodle.moodle_extractor import MoodleExtractor


MOODLE_SECTIONS = [
    {
        "name": "Week 1",
        "summary": "",
        "modules": [
            {
                "id": 11,
                "name": "[SLIDES] Lecture 1",
                "modname": "resource",
                "modplural": "Files",
                "visible": True,
                "availability": None,
                "url": "https://moodle.epfl.ch/mod/resource/view.php?id=11",
                "contents": [
                    {
                        "filename": "lecture1.pdf",
                        "filepath": "/",
                        "mimetype": "application/pdf",
                        "fileurl": "https://moodle.epfl.ch/webservice/pluginfile.php/1",
                    }
                ],
            },
            {
                "id": 12,
                "name": "[NO_BOT] Secret",
                "modname": "resource",
                "modplural": "Files",
                "visible": True,
                "availability": None,
                "url": "https://moodle.epfl.ch/mod/resource/view.php?id=12",
                "contents": [
                    {
                        "filename": "secret.pdf",
                        "filepath": "/",
                        "mimetype": "application/pdf",
                        "fileurl": "https://moodle.epfl.ch/webservice/pluginfile.php/2",
                    }
                ],
            },
        ],
    }
]


def fake_moodle_get(monkeypatch, sections):
    """A requests.get stand-in answering the API call and the file downloads."""

    def fake_get(url, params=None, **kwargs):
        if params is not None:
            return SimpleNamespace(json=lambda: sections, raise_for_status=lambda: None, content=b"")
        return SimpleNamespace(json=lambda: {}, raise_for_status=lambda: None, content=b"file")

    monkeypatch.setattr("rag_etl.extractors.moodle.moodle_extractor.requests.get", fake_get)


def test_moodle_tree(monkeypatch, tmp_path):
    fake_moodle_get(monkeypatch, MOODLE_SECTIONS)

    extractor = MoodleExtractor(course_id=1, dir=tmp_path / "moodle")
    course = extractor.extract()

    assert "course/view.php?id=1" in (course.url or "")
    assert len(course.children) == 1
    section = course.children[0]
    assert section.title == "Week 1"
    assert len(section.children) == 1
    leaf = section.children[0]
    assert leaf.title == "Lecture 1 > lecture1.pdf"
    assert leaf.url == "https://moodle.epfl.ch/pluginfile.php/1"
    assert leaf.mime_type == mt.PDF
    assert leaf.from_ is None and leaf.until is None
    assert leaf.path and leaf.path.read_bytes() == b"file"


def test_moodle_dates(monkeypatch, tmp_path):
    from_epoch, until_epoch = 1770000000, 1771000000
    sections = [
        {
            "name": "Week 1",
            "summary": "",
            "modules": [
                {
                    "id": 11,
                    "name": "Lecture 1",
                    "modname": "resource",
                    "modplural": "Files",
                    "visible": True,
                    "availability": json.dumps(
                        {
                            "c": [
                                {"type": "date", "d": ">=", "t": from_epoch},
                                {"type": "date", "d": "<=", "t": until_epoch},
                            ]
                        }
                    ),
                    "contents": [
                        {
                            "filename": "lecture1.pdf",
                            "filepath": "/",
                            "mimetype": "application/pdf",
                            "fileurl": "https://moodle.epfl.ch/webservice/pluginfile.php/1",
                        }
                    ],
                }
            ],
        }
    ]
    fake_moodle_get(monkeypatch, sections)

    course = MoodleExtractor(course_id=1, dir=tmp_path / "moodle").extract()
    leaf = course.children[0].children[0]
    assert leaf.from_ == datetime.fromtimestamp(from_epoch).strftime("%Y-%m-%dT%H:%M:%S.%f")
    assert leaf.until == datetime.fromtimestamp(until_epoch).strftime("%Y-%m-%dT%H:%M:%S.%f")


def test_moodle_unreachable(monkeypatch, tmp_path):
    def fake_get(url, params=None, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr("rag_etl.extractors.moodle.moodle_extractor.requests.get", fake_get)

    with pytest.raises(SourceUnavailable):
        MoodleExtractor(course_id=1, dir=tmp_path / "moodle").extract()


def test_moodle_pdf_suffix(monkeypatch, tmp_path):
    sections = [
        {
            "name": "Week 1",
            "summary": "",
            "modules": [
                {
                    "id": 11,
                    "name": "Lecture 1",
                    "modname": "resource",
                    "modplural": "Files",
                    "visible": True,
                    "availability": None,
                    "contents": [
                        {
                            "filename": "noext",
                            "filepath": "/",
                            "mimetype": "application/pdf",
                            "fileurl": "https://moodle.epfl.ch/webservice/pluginfile.php/1",
                        }
                    ],
                }
            ],
        }
    ]
    fake_moodle_get(monkeypatch, sections)

    course = MoodleExtractor(course_id=1, dir=tmp_path / "moodle").extract()
    leaf = course.children[0].children[0]
    assert leaf.path and leaf.path.suffix == ".pdf"


def test_local_folder(tmp_path):
    base = tmp_path / "local"
    (base / "week1").mkdir(parents=True)
    (base / "week1" / "notes.txt").write_text("hello")
    (base / "week1" / "from").write_text("2026-02-16T08:00:00.000000")
    (base / "dropped[NO_BOT]").mkdir()
    (base / "dropped[NO_BOT]" / "x.txt").write_text("no")

    root = LocalFolderExtractor(dir=base).extract()

    assert len(root.children) == 1
    leaf = root.children[0]
    assert leaf.title == "week1/notes.txt"
    assert leaf.from_ == "2026-02-16T08:00:00.000000"
    assert leaf.url is None
    assert leaf.mime_type == mt.TXT


def test_local_folder_missing(tmp_path):
    with pytest.raises(SourceUnavailable):
        LocalFolderExtractor(dir=tmp_path / "nowhere").extract()


def _write_mooc_export(base: Path) -> None:
    (base / "course").mkdir(parents=True)
    (base / "course" / "course.xml").write_text('<course display_name="MEMS"><chapter url_name="ch1"/></course>')
    (base / "chapter").mkdir()
    (base / "chapter" / "ch1.xml").write_text('<chapter display_name="Week 1"><sequential url_name="seq1"/></chapter>')
    (base / "sequential").mkdir()
    (base / "sequential" / "seq1.xml").write_text('<sequential display_name="Seq"><vertical url_name="v1"/></sequential>')
    (base / "vertical").mkdir()
    (base / "vertical" / "v1.xml").write_text(
        '<vertical display_name="Vert"><problem url_name="p1"/><video url_name="vid1"/></vertical>'
    )
    (base / "html").mkdir()
    (base / "html" / "h1.xml").write_text('<html display_name="Lecture"/>')
    (base / "html" / "h1.html").write_text("<html><body><p>hello world</p></body></html>")
    (base / "problem").mkdir()
    (base / "problem" / "p1.xml").write_text(
        "<problem display_name=\"Quiz 1\"><multiplechoiceresponse><choicegroup>"
        '<choice correct="true">yes</choice><choice correct="false">no</choice>'
        "</choicegroup></multiplechoiceresponse></problem>"
    )
    (base / "video").mkdir()
    (base / "video" / "vid1.xml").write_text(
        "<video display_name=\"Lecture video\" transcripts='{\"en\": \"sub-en.srt\"}'>"
        '<source src="https://mediaspace.epfl.ch/media/0_abc"/></video>'
    )
    (base / "static").mkdir()
    (base / "static" / "sub-en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    (base / "policies").mkdir()
    (base / "policies" / "assets.json").write_text("{}")


MOOC_TAG_METADATA = {
    "MOOC_QUIZ": {"type": "practice", "subtype": "mooc_quiz", "one_chunk_per_doc": True},
    "MOOC_VIDEO": {"type": "theory", "subtype": "mooc_video", "one_chunk_per_doc": True},
}


def test_mooc_tree(tmp_path):
    base = tmp_path / "export"
    _write_mooc_export(base)

    extractor = MOOCExtractor(
        dir=base,
        tag_metadata=MOOC_TAG_METADATA,
        language="en",
        first_week_chapter=1,
        week_count=1,
    )
    root = extractor.extract()

    # course > chapter (week) > sequential > vertical > leaves
    assert root.title == "MEMS"
    assert len(root.children) == 1
    chapter = root.children[0]
    assert chapter.title == "Week 1" and chapter.week == 1
    sequential = chapter.children[0]
    vertical = sequential.children[0]

    titles = {leaf.title for leaf in vertical.children}
    assert titles == {"Vert - Quiz 1", "Vert - Lecture video"}
    quiz_leaves = [leaf for leaf in vertical.children if leaf.title == "Vert - Quiz 1"]
    quiz_leaves.sort(key=lambda leaf: leaf.is_solution)
    assert [leaf.is_solution for leaf in quiz_leaves] == [False, True]
    assert all(leaf.cut == "whole_document" for leaf in vertical.children)
    assert all(leaf.number == "1" for leaf in vertical.children if leaf.title == "Vert - Quiz 1")

    video = next(leaf for leaf in vertical.children if leaf.title == "Vert - Lecture video")
    assert video.mime_type == mt.MP4
    assert video.url == "https://mediaspace.epfl.ch/media/0_abc"
    assert len(video.children) == 1
    assert video.children[0].mime_type == mt.SRT


def test_mooc_untagged_needs_both_tags(tmp_path):
    with pytest.raises(ValueError):
        MOOCExtractor(dir=tmp_path / "export", include_untagged_documents=True).extract()


def test_mediaspace(monkeypatch, tmp_path):
    module = "rag_etl.extractors.mediaspace.mediaspace_extractor"
    client = SimpleNamespace()
    entries = [
        SimpleNamespace(id="0_abc", name="Lecture 1", duration=100),
        SimpleNamespace(id="0_empty", name="Placeholder", duration=0),
    ]

    monkeypatch.setattr(f"{module}.create_kaltura_session", lambda **kwargs: client)
    monkeypatch.setattr(f"{module}.is_playlist_url", lambda url: True)
    monkeypatch.setattr(f"{module}.extract_playlist_id_from_url", lambda url: "playlist1")
    monkeypatch.setattr(f"{module}.list_playlist_entries", lambda client, playlist_id: entries)
    monkeypatch.setattr(
        f"{module}.get_subtitle_urls",
        lambda client, entry_id: [{"caption_asset_id": "cap1", "file_ext": ".srt", "language_code": "en"}],
    )
    monkeypatch.setattr(f"{module}.caption_mime_type", lambda caption: mt.SRT)

    def fake_download(client, caption, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        return True

    monkeypatch.setattr(f"{module}.download_caption", fake_download)
    monkeypatch.setattr(f"{module}.build_entry_url", lambda entry_id, url: f"{url}/{entry_id}")

    extractor = MediaspaceExtractor(
        playlist_or_channel_url="https://mediaspace.epfl.ch/playlist/1",
        dir=tmp_path / "mediaspace",
        type="theory",
        subtype="video_lecture",
        cut="per_child",
    )
    root = extractor.extract()

    assert root.url == "https://mediaspace.epfl.ch/playlist/1"
    assert len(root.children) == 1
    video = root.children[0]
    assert video.title == "Lecture 1"
    assert video.mime_type == mt.MP4
    assert video.type == "theory" and video.subtype == "video_lecture" and video.cut == "per_child"
    assert video.url == "https://mediaspace.epfl.ch/playlist/1/0_abc"
    assert video.path and json.loads(video.path.read_text())["entry_id"] == "0_abc"

    # the placeholder entry (no duration) is skipped, captions travel beside the video
    assert len(video.children) == 1
    caption = video.children[0]
    assert caption.mime_type == mt.SRT
    assert caption.path and caption.path.read_text().startswith("1\n")


def _write_catalogue(metadata_dir: Path) -> None:
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "moodle.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "type": "exam",
                        "subtype": "previous_year_exam",
                        "number": "2025",
                        "sub_number": None,
                        "title": "Exam 2025",
                        "from": "2026-02-16T08:00:00.000000",
                        "until": "2026-06-01T20:00:00.000000",
                        "week": None,
                    }
                ]
            }
        )
    )


def test_ed_discussion(monkeypatch, tmp_path):
    base = tmp_path / "project"
    _write_catalogue(base / "output" / "metadata")

    ed_dir = base / "ed_discussion" / "2025_2026"
    ed_dir.mkdir(parents=True)
    (ed_dir / "thread1.json").write_text(
        json.dumps(
            {
                "thread": {
                    "id": 1,
                    "type": "question",
                    "title": "Question about exam",
                    "category": "exam",
                    "subcategory": "",
                    "content": "What is on the exam?",
                    "answers": [{"user_id": 2, "content": "Everything", "is_endorsed": False}],
                },
                "users": [{"id": 1, "course_role": "student"}, {"id": 2, "course_role": "staff"}],
            }
        )
    )

    vote = ThreadClassification(type="exam", catalogue_id=0, mentioned_number="2025", week=None, reason="r", confidence=9)
    monkeypatch.setattr(
        "rag_etl.extractors.ed_discussion.ed_discussion_extractor.classify_thread_with_cascade",
        lambda **kwargs: (vote, [vote]),
    )

    extractor = EdDiscussionExtractor(
        ed_discussion_base_path=str(base),
        academic_year="2025_2026",
        categories=["exam"],
        language="english",
        semester=1,
        include_student_endorsed=False,
    )
    root = extractor.extract()

    assert len(root.children) == 1
    leaf = root.children[0]
    assert leaf.title == "Question about exam"
    assert leaf.type == "exam"
    assert leaf.subtype == "previous_year_exam"
    assert leaf.number == "2025"
    assert leaf.cut == "whole_document"
    assert leaf.from_ == "2026-02-16T08:00:00.000000"
    assert leaf.until == "2026-06-01T20:00:00.000000"
    assert leaf.path and leaf.path.exists()


def test_ed_discussion_empty_catalogue(tmp_path):
    base = tmp_path / "project"
    (base / "output" / "metadata").mkdir(parents=True)

    extractor = EdDiscussionExtractor(
        ed_discussion_base_path=str(base),
        academic_year="2025_2026",
        categories=["exam"],
        language="english",
        semester=1,
        include_student_endorsed=False,
    )
    root = extractor.extract()
    assert root.children == []


def test_git_branch_parsing():
    assert branch_from_url("https://github.com/org/repo/tree/dev") == "dev"
    assert branch_from_url("https://gitlab.epfl.ch/org/repo/-/tree/main") == "main"
    assert branch_from_url("https://github.com/org/repo") is None
    assert repo_from_url("https://github.com/org/repo/tree/dev") == "https://github.com/org/repo"
    assert repo_from_url("https://github.com/org/repo") == "https://github.com/org/repo"


def _init_repo(path: Path, branch: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", branch, str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.ch"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)


def _commit(path: Path, filename: str, content: str) -> None:
    (path / filename).write_text(content)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", filename], check=True, capture_output=True)


def test_git_default_branch(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote, "main")
    _commit(remote, "README.md", "# hello")

    root = GitExtractor(repo_url=str(remote), dir=tmp_path / "clone").extract()

    assert root.url == str(remote)
    assert root.title == "remote@main"
    assert len(root.children) == 1
    leaf = root.children[0]
    assert leaf.title == "README.md"
    assert leaf.path and leaf.path.read_text() == "# hello"


def test_git_branch_url(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote, "main")
    _commit(remote, "README.md", "# hello")
    subprocess.run(["git", "-C", str(remote), "switch", "-c", "dev"], check=True, capture_output=True)
    _commit(remote, "feature.txt", "dev work")

    root = GitExtractor(repo_url=f"{remote}/tree/dev", dir=tmp_path / "clone").extract()

    # dev branches off main, so it carries main's history too
    titles = {leaf.title for leaf in root.children}
    assert titles == {"README.md", "feature.txt"}


def test_extractor_contract():
    with pytest.raises(TypeError):
        Extractor(dir=Path("."))  # type: ignore[abstract]
    assert issubclass(SourceUnavailable, RuntimeError)
