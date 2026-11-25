"""Unit tests for the extractors: synthetic fixtures, no network."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import ragtools.mime_types as mt
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.extractors.ed_discussion import EdDiscussionExtractor
from rag_etl.extractors.ed_discussion.thread import format_qa, thread_markdown
from rag_etl.extractors.git import GitExtractor, branch_from_url, repo_from_url
from rag_etl.extractors.local_folder import LocalFolderExtractor
from rag_etl.extractors.mediaspace import MediaspaceExtractor
from rag_etl.extractors.mooc import MOOCExtractor
from rag_etl.extractors.moodle import MoodleExtractor, extract_from_and_until
from rag_etl.transformers import ImagesTransformer, NormalizeTransformer


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


def moodle_settings(monkeypatch):
    """The settings the Moodle extractor reads, pointing nowhere real."""
    monkeypatch.setenv("MOODLE_URL", "https://moodle.test")
    monkeypatch.setenv("MOODLE_TOKEN", "test-token")


def fake_moodle_get(monkeypatch, sections):
    """A requests.get stand-in answering the API call and the file downloads."""
    moodle_settings(monkeypatch)

    def fake_get(url, params=None, **kwargs):
        if params is not None:
            return SimpleNamespace(json=lambda: sections, raise_for_status=lambda: None, content=b"")
        return SimpleNamespace(json=lambda: {}, raise_for_status=lambda: None, content=b"file")

    monkeypatch.setattr("rag_etl.extractors.moodle.requests.get", fake_get)


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
    # Titles stay as Moodle writes them, brackets and all
    assert leaf.title == "[SLIDES] Lecture 1 > lecture1.pdf"
    assert leaf.url == "https://moodle.epfl.ch/pluginfile.php/1"
    assert leaf.mime_type == mt.PDF
    assert leaf.from_ is None and leaf.until is None
    assert leaf.path and leaf.path.read_bytes() == b"file"


def test_moodle_unnamed_section_casts_to_an_empty_title(monkeypatch, tmp_path):
    sections = deepcopy(MOODLE_SECTIONS)
    sections[0]["name"] = None
    fake_moodle_get(monkeypatch, sections)

    extractor = MoodleExtractor(course_id=1, dir=tmp_path / "moodle")
    course = extractor.extract()

    assert course.children[0].title == ""


def test_moodle_dates(monkeypatch, tmp_path):
    # One date in winter, one in summer: Lausanne's offset changes between them
    from_epoch, until_epoch = 1770000000, 1788000000
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
    # Lausanne's time, its offset stated: 02:40 UTC in winter, 10:40 UTC in summer
    assert leaf.from_ == "2026-02-02T03:40:00.000000+01:00"
    assert leaf.until == "2026-08-29T12:40:00.000000+02:00"


def test_moodle_module_without_restrictions_has_no_dates(caplog):
    # The usual case, and a silent one
    assert extract_from_and_until({"name": "Series 3", "availability": None}) == (None, None)
    assert caplog.text == ""


def test_moodle_broken_availability_stops_the_run():
    with pytest.raises(json.JSONDecodeError):
        extract_from_and_until({"name": "Series 3", "availability": "{not json"})


def test_moodle_unreachable(monkeypatch, tmp_path):
    moodle_settings(monkeypatch)

    def fake_get(url, params=None, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr("rag_etl.extractors.moodle.requests.get", fake_get)

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


def test_moodle_own_type_names_fall_back_to_the_file_name(monkeypatch, tmp_path):
    def file(name: str, mimetype: str, number: int) -> dict:
        return {
            "filename": name,
            "filepath": "/",
            "mimetype": mimetype,
            "fileurl": f"https://moodle.epfl.ch/webservice/pluginfile.php/{number}",
        }

    sections = [
        {
            "name": "Week 1",
            "summary": "",
            "modules": [
                {
                    "id": 11,
                    "name": "Material",
                    "modname": "folder",
                    "visible": True,
                    "availability": None,
                    "contents": [
                        # Moodle's own name for gzip, a type the project keeps under its usual name
                        file("series.tar.gz", "application/g-zip", 1),
                        # Nothing the file name says makes this one wanted
                        file("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 2),
                        file("flow.tcl", "document/unknown", 3),
                    ],
                }
            ],
        }
    ]
    fake_moodle_get(monkeypatch, sections)

    course = MoodleExtractor(course_id=1, dir=tmp_path / "moodle").extract()

    kept = {leaf.path.name: leaf.mime_type for leaf in course.children[0].children}
    assert kept == {"series.tar.gz": mt.TAR, "flow.tcl": mt.TCL_SOURCE}


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
        '<vertical display_name="Vert"><html url_name="h1"/><problem url_name="p1"/><video url_name="vid1"/></vertical>'
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


def test_mooc_tree(tmp_path):
    base = tmp_path / "export"
    _write_mooc_export(base)

    extractor = MOOCExtractor(dir=base, language="en", first_week_chapter=1, week_count=1)
    root = extractor.extract()

    # course > chapter (week) > sequential > vertical > leaves
    assert root.title == "MEMS"
    assert len(root.children) == 1
    chapter = root.children[0]
    assert chapter.title == "Week 1" and chapter.week == 1
    sequential = chapter.children[0]
    vertical = sequential.children[0]

    # Every item is extracted, each under its own name: the page, the quiz and the video
    titles = {leaf.title for leaf in vertical.children}
    assert titles == {"Lecture", "Quiz 1", "Lecture video"}

    # The leaves carry facts only; what they are, the judge decides
    assert all(leaf.type is None and leaf.subtype is None and leaf.cut is None for leaf in vertical.children)
    # The week is the chapter's fact, stated there only: the judge hands it down
    assert all(leaf.week is None for leaf in vertical.children)

    quiz_leaves = sorted(
        (leaf for leaf in vertical.children if leaf.title == "Quiz 1"), key=lambda leaf: len(leaf.exercise_parts)
    )
    # The solutions file renders the whole quiz again, its answers marked
    assert [leaf.exercise_parts for leaf in quiz_leaves] == ["statement", "statement + solution"]
    assert all(leaf.number == "1" for leaf in quiz_leaves)
    # The statement too has a solution: its sibling
    assert all(leaf.has_solution for leaf in quiz_leaves)

    video = next(leaf for leaf in vertical.children if leaf.title == "Lecture video")
    assert video.mime_type == mt.MP4
    assert video.url == "https://mediaspace.epfl.ch/media/0_abc"
    assert len(video.children) == 1
    assert video.children[0].mime_type == mt.SRT


def test_mooc_extraction_is_idempotent(tmp_path):
    base = tmp_path / "export"
    _write_mooc_export(base)
    delivered = sorted(path.relative_to(base) for path in base.rglob("*"))
    # A quiz written by an earlier extraction, of a problem the export no longer holds
    (base / "markdown").mkdir()
    (base / "markdown" / "gone.quiz.md").write_text("stale")

    for _ in range(2):
        root = MOOCExtractor(dir=base, language="en", first_week_chapter=1, week_count=1).extract()

    # The export is left as delivered, but for the quizzes the extractor writes into markdown/
    written = sorted(path.relative_to(base) for path in base.rglob("*"))
    assert sorted(set(written) - set(delivered)) == [
        Path("markdown"), Path("markdown/p1.quiz.md"), Path("markdown/p1.quiz_solutions.md")
    ]
    # The page goes as the export holds it, for normalize to convert
    page = next(node for node in root.walk() if node.title == "Lecture")
    assert page.mime_type == mt.HTML and page.path == base / "html" / "h1.html"


def test_mooc_page_images_are_found_in_the_export_and_read(tmp_path, monkeypatch):
    base = tmp_path / "export"
    _write_mooc_export(base)
    # The page names its image the courseware's way; the export holds it under another spelling
    (base / "html" / "h1.html").write_text(
        '<html><body><p>Before.</p><img src="/static/Intro_picture.png"/>'
        '<img src="https://x.org/web.png"/><img src="/static/gone.png"/><p>After.</p></body></html>'
    )
    (base / "static" / "Intro picture.png").write_bytes(b"png fake")
    (base / "policies" / "assets.json").write_text(json.dumps({"Intro_picture.png": {"displayname": "Intro picture.png"}}))

    root = MOOCExtractor(dir=base, language="en", first_week_chapter=1, week_count=1).extract()

    # A copy of the page, in the one folder the extractor writes, the export's own page left as it is
    page = next(node for node in root.walk() if node.title == "Lecture")
    assert page.path == base / "markdown" / "h1.html"
    assert "/static/Intro_picture.png" in (base / "html" / "h1.html").read_text()

    # Normalize makes it markdown, and the images step reads the image where it is shown
    monkeypatch.setattr("ragtools.llm.RCPClient.generate", lambda self, prompt, **kwargs: "a clean room")
    NormalizeTransformer(project_dir=tmp_path / "project").transform([root])
    ImagesTransformer(project_dir=tmp_path / "project").transform([root])

    text = page.path.read_text()
    assert "Before.\n\na clean room" in text and "After." in text
    # A web image and one the export does not hold stay as the page wrote them
    assert "https://x.org/web.png" in text and "/static/gone.png" in text


def test_mooc_page_marked_no_bot_is_left_out(tmp_path):
    base = tmp_path / "export"
    _write_mooc_export(base)
    (base / "html" / "h1.html").write_text("<html><body><p>[NO_BOT] staff notes</p></body></html>")

    root = MOOCExtractor(dir=base, language="en", first_week_chapter=1, week_count=1).extract()

    titles = {leaf.title for leaf in root.walk() if leaf.is_leaf}
    assert "Lecture" not in titles
    assert "Quiz 1" in titles


def test_mediaspace(monkeypatch, tmp_path):
    module = "rag_etl.extractors.mediaspace"
    client = SimpleNamespace()
    entries = [
        SimpleNamespace(id="0_abc", name="Lecture 1", duration=100),
        SimpleNamespace(id="0_empty", name="Placeholder", duration=0),
    ]

    monkeypatch.setattr(f"{module}.session_from_env", lambda: client)
    monkeypatch.setattr(f"{module}.is_playlist_url", lambda url: True)
    monkeypatch.setattr(f"{module}.extract_playlist_id_from_url", lambda url: "playlist1")
    monkeypatch.setattr(f"{module}.list_playlist_entries", lambda client, playlist_id: entries)
    monkeypatch.setattr(
        f"{module}.list_caption_assets",
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
    )
    root = extractor.extract()

    assert root.url == "https://mediaspace.epfl.ch/playlist/1"
    assert len(root.children) == 1
    video = root.children[0]
    assert video.title == "Lecture 1"
    assert video.mime_type == mt.MP4
    # Facts only: labelling the recordings is the judge's and the overrides' business
    assert video.type is None and video.subtype is None and video.cut is None
    # The watch url names the entry; nothing is written to stand for the video
    assert video.url == "https://mediaspace.epfl.ch/playlist/1/0_abc"
    assert video.path is None

    # the placeholder entry (no duration) is skipped, the subtitles are the video's child
    assert len(video.children) == 1
    subtitles = video.children[0]
    assert subtitles.mime_type == mt.SRT
    assert subtitles.path and subtitles.path.read_text().startswith("1\n")


def _write_thread(ed_dir: Path, name: str, title: str, category: str, subcategory: str, answered: bool) -> None:
    answers = [{"user_id": 2, "content": "Everything", "is_endorsed": False}] if answered else []
    (ed_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "thread": {
                    "id": 1,
                    "type": "question",
                    "title": title,
                    "category": category,
                    "subcategory": subcategory,
                    "content": "What is on it?",
                    "document": "What is on it?",
                    "answers": answers,
                },
                "users": [{"id": 1, "course_role": "student"}, {"id": 2, "course_role": "staff"}],
            }
        )
    )


def test_ed_discussion(tmp_path):
    ed_dir = tmp_path / "ed_discussion" / "2025_2026"
    ed_dir.mkdir(parents=True)
    _write_thread(ed_dir, "t1", "Question about exam 2024", "Exams", "Exam 2024", answered=True)
    _write_thread(ed_dir, "t2", "Room change?", "Logistics", "", answered=True)
    _write_thread(ed_dir, "t3", "Nobody answered", "Exams", "Exam 2024", answered=False)

    root = EdDiscussionExtractor(dir=tmp_path / "ed_discussion", academic_year="2025_2026", language="en").extract()

    # Ed's own filing shapes the tree; facts only on the leaves; unanswered threads are left out
    tree = {category.title: {child.title: [leaf.title for leaf in child.children] for child in category.children}
            for category in root.children}
    assert tree["Exams"] == {"Exam 2024": ["Question about exam 2024"]}
    assert list(tree["Logistics"]) == ["Room change?"]
    leaf = next(node for node in root.walk() if node.title == "Question about exam 2024")
    assert leaf.mime_type == mt.MARKDOWN and leaf.type is None and leaf.cut is None
    # Ed states that a thread is a forum Q&A; what it is about is the judge's
    assert leaf.subtype == "forum_qa"
    # Containers carry no subtype: only the threads are Q&A
    assert all(node.subtype is None for node in root.walk() if not node.is_leaf)
    assert leaf.path and "# Answer:" in leaf.path.read_text()


def test_ed_discussion_headings_follow_the_forums_language():
    assert format_qa("Q?", ["A1", "A2"], "fr") == "# Question :\nQ?\n\n# Réponse 1 :\nA1\n\n# Réponse 2 :\nA2"
    assert format_qa("Q?", ["A"], "en") == "# Question:\nQ?\n\n# Answer:\nA"


ED_USERS = [{"id": 1, "course_role": "student"}, {"id": 2, "course_role": "staff"}]


def test_ed_discussion_keeps_staff_posts_and_endorsed_ones_if_asked(tmp_path):
    thread = {
        "type": "question",
        "title": "Etching",
        "document": "Why?",
        "answers": [
            # A student's answer, endorsed, with a staff follow-up under it
            {"user_id": 1, "is_endorsed": True, "document": "student", "comments": [{"user_id": 2, "document": "staff follow-up"}]},
            {"user_id": 1, "document": "unendorsed"},
        ],
        "comments": [{"user_id": 2, "document": "staff comment"}],
    }

    without = thread_markdown(thread, ED_USERS, tmp_path, "en", include_endorsed=False)
    endorsed = thread_markdown(thread, ED_USERS, tmp_path, "en", include_endorsed=True)

    assert without == "# Question:\nEtching\n\nWhy?\n\n# Answer 1:\nstaff follow-up\n\n# Answer 2:\nstaff comment"
    assert "student" in endorsed and "unendorsed" not in endorsed
    # Unanswered, or not a question: nothing to index
    assert thread_markdown({**thread, "answers": [], "comments": []}, ED_USERS, tmp_path, "en", False) is None
    assert thread_markdown({**thread, "type": "post"}, ED_USERS, tmp_path, "en", False) is None


def test_ed_discussion_shows_every_image_of_a_post(tmp_path):
    (tmp_path / "a.png").write_bytes(b"png")
    (tmp_path / "b").write_bytes(b"png")  # Ed drops the extension of some
    post = {"user_id": 2, "document": "See", "content": '<image src="a.png"/><image src="https://static.us.edusercontent.com/files/b"/>'}
    thread = {"type": "question", "title": "T", "document": "Q", "answers": [post]}

    markdown = thread_markdown(thread, ED_USERS, tmp_path, "en", False)

    assert f"![](<{tmp_path / 'a.png'}>)" in markdown and f"![](<{tmp_path / 'b'}>)" in markdown


def test_ed_discussion_without_export(tmp_path):
    with pytest.raises(SourceUnavailable):
        EdDiscussionExtractor(dir=tmp_path, academic_year="2025_2026", language="en").extract()


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
