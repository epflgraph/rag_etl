import re


def extract_tag(text):
    """
    Extract tag from text (e.g. for "[SERIE_3] Serie 3" returns "SERIE_3").
    """

    match = re.search(r"\[(.*?)\]", text)

    if match:
        return match.group(1)

    return None


def extract_tag_and_number(text):
    """
    Extract tag and number from text (e.g. for "[SERIE_3] Serie 3" returns ("SERIE", 3)).
    """

    tag = extract_tag(text)

    # If no tag, return Nones
    if tag is None:
        return (None, None)

    # Try to extract number from tag
    match = re.search(r'_(\d+)(?:_|$)', tag)

    # If no number, return only tag
    if not match:
        return (tag, None)

    # If number, remove it from tag and return both separately
    number = match.group(1)
    rest = tag.replace(f'_{number}', '')

    # Try to cast to int and then to string ("05" -> "5" but "5A" -> "5A")
    try:
        number = str(int(number))
    except Exception:
        pass

    return (rest, number)
