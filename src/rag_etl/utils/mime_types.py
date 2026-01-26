import mimetypes

MARKDOWN = "text/markdown"
TXT = "text/plain"
PDF = "application/pdf"
ZIP = "application/zip"
IPYNB = "application/x-ipynb+json"
MP4 = "video/mp4"

C_SOURCE = "text/x-c"

mimetypes.add_type(IPYNB, ".ipynb")

DEFAULT_MIME_TYPES = [
    TXT,
    MARKDOWN,
    PDF,
    IPYNB,
    ZIP,
]


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type
