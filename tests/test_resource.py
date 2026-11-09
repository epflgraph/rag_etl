from pathlib import Path

import hashlib

from rag_etl.core.resource import Resource, hash_bytes, hash_text, iter_chunks


def test_hash_helpers():
    assert hash_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert hash_text("abc") == hashlib.sha256(b"abc").hexdigest()


def test_leaf_state_is_organic():
    pdf = Resource(mime_type="application/pdf", title="week1")
    assert pdf.is_leaf

    page = pdf.add_child(mime_type="application/pdf")
    assert not pdf.is_leaf
    assert page.parent is pdf

    page.text = "ocr text"
    assert page.is_leaf

    chunk = Resource(title="chunk", text="chunk")
    assert chunk.is_leaf


def test_content_hash_from_text():
    assert Resource(title="x", text="hello").content_hash == hash_text("hello")


def test_content_hash_from_file(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"pdf bytes")
    assert Resource(title="a.pdf", path=f).content_hash == hash_bytes(b"pdf bytes")


def test_content_hash_changes_with_content():
    r = Resource(title="x", text="hello")
    before = r.content_hash
    r.text = "changed"
    assert r.content_hash != before


def test_equal_content_equal_hash():
    assert Resource(title="x", text="same").content_hash == Resource(title="x", text="same").content_hash


def test_container_hash_follows_children():
    pdf = Resource(title="pdf", mime_type="application/pdf")
    first = pdf.add_child(text="one")
    second = pdf.add_child(text="two")

    combined = hashlib.sha256()
    for child in (first, second):
        combined.update(child.content_hash.encode("utf-8"))
    assert pdf.content_hash == combined.hexdigest()

    # document order matters
    flipped = Resource(title="flipped", mime_type="application/pdf")
    flipped.add_child(text="two")
    flipped.add_child(text="one")
    assert flipped.content_hash != pdf.content_hash

    # adding a child changes the parent
    before = pdf.content_hash
    pdf.add_child(text="three")
    assert pdf.content_hash != before


def test_text_wins_over_path(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"raw")
    assert Resource(title="a.pdf", path=f, text="extracted").content_hash == hash_text("extracted")


def test_walk_depth_first():
    root = Resource(title="root")
    a = root.add_child(title="a")
    root.add_child(title="b")
    a.add_child(title="a1")
    assert [n.title for n in root.walk()] == ["root", "a", "a1", "b"]


def test_lineage_root_to_self():
    root = Resource(title="root")
    page = root.add_child(title="page")
    chunk = page.add_child(text="c")
    chain = chunk.lineage()
    assert chain[0] is root and chain[1] is page and chain[2] is chunk


def test_children_get_parent_backfill():
    child = Resource(title="child")
    root = Resource(title="root", children=[child])
    assert child.parent is root


def test_add_child_builds_a_copy_of_the_parent():
    pdf = Resource(title="pdf", url="https://moodle/x", type="practice", week=3)
    page = pdf.add_child(title="page 1", path=Path("p.md"), mime_type="text/markdown")

    # everything comes down; the overrides name what differs
    assert page.parent is pdf
    assert page.title == "page 1"
    assert page.url == "https://moodle/x"
    assert page.type == "practice"
    assert page.week == 3
    assert page.mime_type == "text/markdown"


def test_add_child_overrides_win_and_attachments_stay_as_they_are():
    pdf = Resource(title="pdf", url="https://moodle/x", cut="per_exercise")
    own = pdf.add_child(title="own", url="https://elsewhere/y")
    assert own.url == "https://elsewhere/y"

    built = Resource(title="built elsewhere", url="https://built/z")
    pdf.add_child(built)
    assert built.parent is pdf
    # a subtree built elsewhere is attached as it is, not overwritten
    assert built.url == "https://built/z"
    assert built.cut is None
    assert built in pdf.children


def test_iter_chunks_finds_text_nodes():
    root = Resource(title="root")
    chunk = root.add_child(title="chunk", text="hello")
    container = root.add_child(title="container")
    container.add_child(title="page")

    assert list(iter_chunks([root])) == [chunk]


def test_detach_removes_the_subtree_and_spares_a_root():
    root = Resource(title="root")
    child = root.add_child(title="child")
    child.add_child(title="grandchild")

    child.detach()
    root.detach()

    assert root.children == []
    assert [node.title for node in child.walk()] == ["child", "grandchild"]
