import re


def extract_moodle_tag(text):
    match = re.search(r"\[(.*?)\]", text)

    if match:
        return match.group(1)

    return None
