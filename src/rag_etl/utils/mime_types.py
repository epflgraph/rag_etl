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

# What Moodle's API reports for file extensions it does not recognise
MOODLE_UNKNOWN = "document/unknown"

C_SOURCES = ["text/x-c", "text/x-chdr", "text/x-csrc", "text/x-c++src", "text/x-c++hdr"]

TCL_SOURCE = "application/x-tcl"

PYTHON_SOURCE = "text/x-python"

MATLAB_SOURCE = "text/x-matlab"

ASSEMBLY_SOURCES = ["text/x-asm", "text/x-verilog"]

mimetypes.add_type(IPYNB, ".ipynb")

# .s and .m not registered in Python's MIME map
mimetypes.add_type("text/x-asm", ".s")
mimetypes.add_type(MATLAB_SOURCE, ".m")


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
