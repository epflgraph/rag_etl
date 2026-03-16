import mimetypes

MARKDOWN = "text/markdown"
TXT = "text/plain"
SRT = "application/x-subrip"

PDF = "application/pdf"

ZIP = "application/zip"

IPYNB = "application/x-ipynb+json"
MP4 = "video/mp4"
JSON = "application/json"

C_SOURCES = ["text/x-c", "text/x-chdr", "text/x-csrc", "text/x-c++src"]

TCL_SOURCE = "application/x-tcl"

PYTHON_SOURCE = "text/x-python"

MATLAB_SOURCE = "text/x-matlab"

mimetypes.add_type(IPYNB, ".ipynb")

DEFAULT_MIME_TYPES = [
    MARKDOWN,
    TXT,
    SRT,
    PDF,
    ZIP,
    IPYNB,
    PYTHON_SOURCE,
] + C_SOURCES


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type
