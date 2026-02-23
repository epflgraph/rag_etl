import pathvalidate
import unicodedata


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
