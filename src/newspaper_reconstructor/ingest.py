import json


def load_article_json(path: str) -> list[dict]:
    """Load a {region_id: text} JSON file into a fragment list.

    Args:
        path: Path to a JSON file mapping region IDs to OCR text strings.

    Returns:
        List of dicts with 'id' and 'text' keys.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": k, "text": v} for k, v in data.items()]
