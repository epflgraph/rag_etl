import re


def split_tag_text(text):
    """
    Split tag from text (e.g. for "[SERIE_3] Serie 3" returns ("SERIE_3", "Serie 3")).
    """

    match = re.search(r"\[(.*?)\]", text)

    if match:
        tag = match.group(1)
        text = re.sub(r"\s*\[.*?\]\s*", "", text, count=1).strip()
        return (tag, text)

    return (None, text)


def split_tag_number_text(text):
    """
    Extract tag and number from text.

    Examples:
        "[SERIE_3] Serie 3" -> ("SERIE", "3", "Serie 3")
        "[CASE_STUDY_L3C2] Case study L3C2" -> ("CASE_STUDY", "L3C2", "Case study L3C2")
    """

    tag, text = split_tag_text(text)

    # If no tag, return
    if tag is None:
        return (None, None, text)

    # Try to extract number from tag. Supports integer numbering ("_3") and
    # lecture/case numbering ("_L3C2") used for case studies.
    match = re.search(r"_((?:\d+)|(?:L\d+C\d+))(?:_|$)", tag)

    # If no number, return only tag
    if not match:
        return (tag, None, text)

    # If number, remove it from tag and return both separately
    number = match.group(1)
    tag = re.sub(rf"_{number}", "", tag, count=1)

    # Try to cast to int and then to string ("05" -> "5" but "5A" -> "5A")
    try:
        number = str(int(number))
    except Exception:
        pass

    return (tag, number, text)
