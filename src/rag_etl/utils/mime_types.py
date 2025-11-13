import mimetypes

MARKDOWN = "text/markdown"
PDF = "application/pdf"
ZIP = "application/zip"
IPYNB = "application/x-ipynb+json"
MP4 = "video/mp4"

mimetypes.add_type(IPYNB, ".ipynb")


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type
