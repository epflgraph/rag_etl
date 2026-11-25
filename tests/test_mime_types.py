"""Unit tests for the mime helpers: what counts as text, and what extractors keep by default."""

import ragtools.mime_types as mt


def test_code_filed_under_application_is_text():
    # Tcl is how chip-design tools are scripted: its files are code, read as they are
    assert mt.guess_mime_type("flow.tcl") == mt.TCL_SOURCE
    assert mt.is_text(mt.TCL_SOURCE)
    assert mt.is_text(mt.guess_mime_type("run.sh"))
    assert mt.is_text(mt.MARKDOWN) and mt.is_text(mt.PYTHON_SOURCE)


def test_binaries_are_not_text():
    assert not mt.is_text(mt.PDF)
    assert not mt.is_text(mt.PNG)
    assert not mt.is_text(None)


def test_defaults_keep_everything_the_pipeline_reads():
    for mime_type in (mt.TCL_SOURCE, mt.MATLAB_SOURCE, mt.TAR, mt.JPEG, mt.PNG, *mt.ASSEMBLY_SOURCES, *mt.C_SOURCES):
        assert mime_type in mt.DEFAULT_MIME_TYPES
