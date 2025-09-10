def clean_text(text: str) -> str:
    """Basic text cleanup"""
    return text.strip().replace("\n", " ")
