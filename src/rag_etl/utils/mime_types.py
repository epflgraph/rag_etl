import mimetypes

MARKDOWN = "text/markdown"
MD = "text/markdown"
TXT = "text/plain"
SRT = "application/x-subrip"

PDF = "application/pdf"

ZIP = "application/zip"

IPYNB = "application/x-ipynb+json"
MP4 = "video/mp4"
JSON = "application/json"

C_SOURCE = "text/x-c"
TCL_SOURCE = "application/x-tcl"

PYTHON_SOURCE = "text/x-python"

mimetypes.add_type(IPYNB, ".ipynb")

DEFAULT_MIME_TYPES = [
    MARKDOWN,
    TXT,
    SRT,
    PDF,
    IPYNB,
    ZIP,
]


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type
