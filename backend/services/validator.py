import re

def is_urdu(text):
    """
    Checks if the text contains Urdu characters.
    """
    if not text:
        return False
    # Urdu character range: \u0600-\u06FF
    return bool(re.search(r'[\u0600-\u06FF]', text))
