"""Article reconstruction from ALTO XML text fragments using LLM prompting.

Workflow:
    1. Parse ALTO XML → fragment list (alto_to_json / load_fragments)
    2. Send fragments to LLM → reconstructed items (reconstruct_articles)
    3. Evaluate against ground truth (evaluate.py)
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

from openai import APIError, APITimeoutError

from .llm import LLMClient

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def alto_to_json(alto_path: str) -> list[dict]:
    """Parse an ALTO XML file, returning one entry per TextBlock with text.

    Skips Illustration elements and empty text blocks.
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {"alto": ALTO_NS}

    result = []
    for tb in root.findall(".//alto:TextBlock", ns):
        strings = tb.findall(".//alto:String", ns)
        text = " ".join(s.get("CONTENT", "") for s in strings)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        entry = {
            "id": tb.get("ID"),
            "text": text,
            "type": tb.get("TYPE"),
            "hpos": int(tb.get("HPOS", 0)),
            "vpos": int(tb.get("VPOS", 0)),
            "width": int(tb.get("WIDTH", 0)),
            "height": int(tb.get("HEIGHT", 0)),
        }
        result.append(entry)
    return result


def load_fragments(path: str) -> list[dict] | dict[str, list[dict]]:
    """Load fragments from a file or directory.

    For .xml files: parses ALTO XML via alto_to_json.
    For .json files: reads pre-parsed fragment list.
    For directories: returns a dict keyed by page name (without extension).

    Auto-detects file type by extension.
    """
    if os.path.isdir(path):
        result = {}
        for fname in sorted(os.listdir(path)):
            if not fname.endswith((".xml", ".json")):
                continue
            page_name = os.path.splitext(fname)[0]
            fpath = os.path.join(path, fname)
            result[page_name] = _load_single(fpath)
        return result
    return _load_single(path)


def load_fragments_cached(
    path: str,
    interim_dir: str | None = None,
    force: bool = False,
) -> list[dict] | dict[str, list[dict]]:
    """Load fragments with optional JSON caching in interim_dir.

    For .xml files: if a cached .json exists in interim_dir and force is False,
    loads the cached JSON instead of re-parsing ALTO XML.
    After parsing, saves the result to interim_dir for future runs.
    For .json files: passes through directly (no caching needed).
    For directories: applies the same logic per file.

    Args:
        path: path to ALTO/JSON file or directory
        interim_dir: directory for cached JSON files (created if needed)
        force: if True, re-parse ALTO even if cached JSON exists
    """
    if interim_dir is None:
        return load_fragments(path)

    if os.path.isdir(path):
        return _load_dir_cached(path, interim_dir, force)
    return _load_single_cached(path, interim_dir, force)


def _load_dir_cached(
    input_dir: str, interim_dir: str, force: bool
) -> dict[str, list[dict]]:
    result = {}
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith((".xml", ".json")):
            continue
        page_name = os.path.splitext(fname)[0]
        fpath = os.path.join(input_dir, fname)
        result[page_name] = _load_single_cached(fpath, interim_dir, force)
    return result


def _load_single_cached(fpath: str, interim_dir: str, force: bool) -> list[dict]:
    ext = os.path.splitext(fpath)[1].lower()
    page_name = os.path.splitext(os.path.basename(fpath))[0]
    cache_dir = os.path.join(interim_dir, "fragments")
    cache_path = os.path.join(cache_dir, f"{page_name}.json")

    if ext == ".json":
        with open(fpath, encoding="utf-8") as f:
            return json.load(f)

    if ext == ".xml":
        if not force and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        fragments = alto_to_json(fpath)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(fragments, f, indent=2, ensure_ascii=False)
        return fragments

    raise ValueError(f"Unsupported file type: {ext}")


def _load_single(fpath: str) -> list[dict]:
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".xml":
        return alto_to_json(fpath)
    if ext == ".json":
        with open(fpath, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file type: {ext}")


def reconstruct_articles(
    fragments: list[dict],
    client: LLMClient,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
) -> list[dict] | None:
    """Send fragments to LLM and return reconstructed items.

    Each fragment must have "id" and "text" keys.
    Returns a list of {"fragment_ids": [...], "title": str, "class": str}.
    Returns None if the LLM call fails after max_retries attempts.
    """
    frag_text = json.dumps(fragments, ensure_ascii=False, indent=2)
    user_prompt = user_prompt_template.format(fragments=frag_text)

    for attempt in range(max_retries):
        try:
            raw = client.complete(system_prompt, user_prompt)
        except APITimeoutError:
            print(
                f"  Timed out on attempt {attempt + 1}, giving up immediately",
                file=sys.stderr,
            )
            return None
        except APIError as e:
            if "504" in str(e) or "Gateway Timeout" in str(e):
                print(
                    f"  Gateway Timeout on attempt {attempt + 1}, giving up immediately",
                    file=sys.stderr,
                )
                return None
            if attempt < max_retries - 1:
                wait = 5 * (2**attempt)
                print(f"  API error: {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  API error after {max_retries} attempts: {e}", file=sys.stderr)
            return None

        parsed = _parse_json_response(raw)
        if parsed is not None:
            return _validate_items(parsed, fragments)
        if attempt < max_retries - 1:
            continue
    print(
        f"  Failed to parse LLM response as JSON after {max_retries} attempts",
        file=sys.stderr,
    )
    return None


def reconstruct_articles_cached(
    fragments: list[dict],
    client: LLMClient,
    system_prompt: str,
    user_prompt_template: str,
    page_id: str,
    interim_dir: str,
    prompt_name: str,
    model: str,
    force: bool = False,
) -> list[dict] | None:
    """Reconstruct articles with optional output caching.

    If a cached reconstruction exists for this prompt_name + model + page_id,
    returns it without calling the LLM (unless force=True).
    Otherwise calls the LLM, saves the result to cache, and returns it.

    Cache path: {interim_dir}/reconstructions/{prompt_name}/{model}/{page_id}.json
    """
    cache_dir = os.path.join(interim_dir, "reconstructions", prompt_name, model)
    cache_path = os.path.join(cache_dir, f"{page_id}.json")

    if not force and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    result = reconstruct_articles(
        fragments, client, system_prompt, user_prompt_template
    )

    if result is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _parse_json_response(raw: str) -> list[dict] | None:
    """Extract and parse a JSON array from an LLM response.

    Handles markdown fences and surrounding text.
    """
    text = raw.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding the first [ to last ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _validate_items(items: list[dict], fragments: list[dict]) -> list[dict]:
    """Filter LLM output to valid items whose fragment_ids exist in the input.

    Drops items that:
    - Are not dicts or lack a 'fragment_ids' key
    - Have empty or non-list fragment_ids
    - Reference fragment IDs not present in the input
    """
    valid_ids = {f["id"] for f in fragments}
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fids = item.get("fragment_ids")
        if not isinstance(fids, list) or not fids:
            continue
        fids = [fid for fid in fids if fid in valid_ids]
        if not fids:
            continue
        item["fragment_ids"] = fids
        result.append(item)
    return result
