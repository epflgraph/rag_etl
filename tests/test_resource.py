import hashlib

from rag_etl.core.resource import Resource, hash_bytes, hash_text


def test_hash_helpers():
    assert hash_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert hash_text("abc") == hashlib.sha256(b"abc").hexdigest()


def test_leaf_state_is_organic():
    pdf = Resource(mime_type="application/pdf", title="week1")
    assert pdf.is_leaf

    page = pdf.add_child(Resource(mime_type="application/pdf"))
    assert not pdf.is_leaf
    assert page.parent is pdf

    page.text = "ocr text"
    assert page.is_leaf

    chunk = Resource(text="chunk")
    assert chunk.is_leaf


def test_content_hash_from_text():
    assert Resource(text="hello").content_hash == hash_text("hello")


def test_content_hash_from_file(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"pdf bytes")
    assert Resource(path=f).content_hash == hash_bytes(b"pdf bytes")


def test_content_hash_changes_with_content():
    r = Resource(text="hello")
    before = r.content_hash
    r.text = "changed"
    assert r.content_hash != before


def test_equal_content_equal_hash():
    assert Resource(text="same").content_hash == Resource(text="same").content_hash


def test_container_hash_follows_children():
    pdf = Resource(mime_type="application/pdf")
    first = pdf.add_child(Resource(text="one"))
    second = pdf.add_child(Resource(text="two"))

    combined = hashlib.sha256()
    for child in (first, second):
        combined.update(child.content_hash.encode("utf-8"))
    assert pdf.content_hash == combined.hexdigest()

    # document order matters
    flipped = Resource(mime_type="application/pdf")
    flipped.add_child(Resource(text="two"))
    flipped.add_child(Resource(text="one"))
    assert flipped.content_hash != pdf.content_hash

    # adding a child changes the parent
    before = pdf.content_hash
    pdf.add_child(Resource(text="three"))
    assert pdf.content_hash != before


def test_text_wins_over_path(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"raw")
    assert Resource(path=f, text="extracted").content_hash == hash_text("extracted")


def test_walk_depth_first():
    root = Resource(title="root")
    a = root.add_child(Resource(title="a"))
    root.add_child(Resource(title="b"))
    a.add_child(Resource(title="a1"))
    assert [n.title for n in root.walk()] == ["root", "a", "a1", "b"]


def test_lineage_root_to_self():
    root = Resource(title="root")
    page = root.add_child(Resource(title="page"))
    chunk = page.add_child(Resource(text="c"))
    chain = chunk.lineage()
    assert chain[0] is root and chain[1] is page and chain[2] is chunk


def test_children_get_parent_backfill():
    child = Resource()
    root = Resource(children=[child])
    assert child.parent is root
