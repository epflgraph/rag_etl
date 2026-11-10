import mimetypes

MARKDOWN = "text/markdown"
TXT = "text/plain"
SRT = "application/x-subrip"

JPEG = "image/jpeg"
PNG = "image/png"

PDF = "application/pdf"

ZIP = "application/zip"

IPYNB = "application/x-ipynb+json"
MP4 = "video/mp4"
JSON = "application/json"

GZIP = "application/gzip"
TAR = "application/x-tar"

# What Moodle's API reports for file extensions it does not recognise
MOODLE_UNKNOWN = "document/unknown"

C_SOURCES = ("text/x-c", "text/x-chdr", "text/x-csrc", "text/x-c++src", "text/x-c++hdr")

TCL_SOURCE = "application/x-tcl"

PYTHON_SOURCE = "text/x-python"

MATLAB_SOURCE = "text/x-matlab"

ASSEMBLY_SOURCES = ("text/x-asm", "text/x-verilog")

# Code that Python's map files under application/ rather than text/: it is
# text all the same, read as it is
TEXT_APPLICATION_TYPES = (TCL_SOURCE, JSON, "application/javascript", "application/x-sh")

mimetypes.add_type(IPYNB, ".ipynb")

# programming types not registered in Python's MIME map
mimetypes.add_type("text/x-asm", ".v")
mimetypes.add_type("text/x-asm", ".ld")
mimetypes.add_type(MATLAB_SOURCE, ".m")

# What extractors that filter by type keep unless a project says otherwise:
# everything the pipeline can read, archives to unpack and images to transcribe included
DEFAULT_MIME_TYPES = (
    MARKDOWN,
    TXT,
    SRT,
    PDF,
    ZIP,
    TAR,
    GZIP,
    IPYNB,
    JPEG,
    PNG,
    PYTHON_SOURCE,
    MATLAB_SOURCE,
    TCL_SOURCE,
    *C_SOURCES,
    *ASSEMBLY_SOURCES,
)


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type


def is_text(mime_type: str | None) -> bool:
    """Whether a mime type names text, which is read as it is: Markdown, plain text and code of any kind."""
    return mime_type is not None and (mime_type.startswith("text/") or mime_type in TEXT_APPLICATION_TYPES)
