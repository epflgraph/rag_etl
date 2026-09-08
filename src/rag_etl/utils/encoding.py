import pathvalidate
import unicodedata
import zipfile


def sanitize_for_filename(text: str) -> str:
    # Normalize (e.g. solve é vs. e+´)
    sanitized_text = unicodedata.normalize("NFC", text)

    # Sanitize for filesystem (e.g. replace / with safe characters)
    sanitized_text = pathvalidate.sanitize_filename(sanitized_text)

    # Strip initial dots from filename (avoid hidden files and '.' or '..'
    sanitized_text = sanitized_text.lstrip(".")

    # Complain if nothing left after sanitization
    if not sanitized_text:
        raise ValueError(f"Text `{text}` is empty after sanitization")

    return sanitized_text


# Bit 11 of an entry's flags: set when the archive states its names are UTF-8
UTF8_FLAG = 0x800


def zip_entry_filename(entry: zipfile.ZipInfo) -> str:
    """
    Return the name of an entry, readable and in one normal form.

    An archive says whether its names are UTF-8 with a flag, and the ones
    written on macOS do not set it. Reading those as the format's default
    encoding turns every accent into a pair of box drawing characters, so a
    name without the flag is decoded back to UTF-8 first.

    The name is then composed, since macOS stores accents decomposed while the
    rest of the tree writes them composed, and two spellings of one name look
    like two different files to everything downstream.
    """

    name = entry.filename

    utf8_flag = entry.flag_bits & UTF8_FLAG
    if not utf8_flag:
        try:
            name = name.encode("cp437").decode("utf-8")
        except UnicodeError:
            # The name really was in the default encoding, so it stands
            pass

    return unicodedata.normalize("NFC", name)


if __name__ == "__main__":
    examples = [
        "résumé 2026: final version?.pdf",
        "../../etc/passwd",
        "/absolute/path/to/file.txt",
        r"C:\Windows\system32\file.txt",
        ".. / . / .. file ../.. 2 .pdf",
        "Slides for session 3/4",
        ".",
        "..",
        "////",
        "é.txt",
        "e\u0301.txt",
        r'<>:"/\|?*',
        "file name. ",
        "",
        "my\nfile\tname.txt",
        "CON.txt",
        "NUL",
        "LPT1.doc",
    ]

    for example in examples:
        try:
            print(f"{example!r} -> {sanitize_for_filename(example)}")
        except Exception as e:
            print(f"{example!r} -> ERROR: {e}")
