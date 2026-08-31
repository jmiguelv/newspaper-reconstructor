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

LLM_AND_IO_ERRORS = (
    APIError,
    APITimeoutError,
    OSError,
    json.JSONDecodeError,
    ValueError,
)

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


def classify_fragments(
    fragments: list[dict],
    client: LLMClient,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
    prompt_out_path: str | None = None,
) -> dict[str, str] | None:
    """Send fragments to LLM for classification.

    Returns a dict mapping fragment ID to predicted class (e.g., {"fid1": "headline"}).
    Returns None if the LLM call fails.
    """
    frag_text = json.dumps(fragments, ensure_ascii=False, indent=2)
    user_prompt = user_prompt_template.format(fragments=frag_text)

    if prompt_out_path:
        os.makedirs(os.path.dirname(prompt_out_path), exist_ok=True)
        with open(prompt_out_path, "w", encoding="utf-8") as f:
            f.write(f"# System Prompt\n{system_prompt}\n\n# User Prompt\n{user_prompt}")

    for attempt in range(max_retries):
        try:
            raw = client.complete(system_prompt, user_prompt)
        except (APITimeoutError, APIError) as e:
            if _handle_api_error(e, attempt, max_retries):
                continue
            return None

        parsed = _parse_classification_response(raw)
        if parsed is not None:
            return _validate_classification(parsed, fragments)

        if attempt < max_retries - 1:
            continue

    print(
        f"  Failed to parse LLM response as JSON dict after {max_retries} attempts",
        file=sys.stderr,
    )
    return None


def reconstruct_articles(
    fragments: list[dict],
    client: LLMClient,
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int = 3,
    prompt_out_path: str | None = None,
) -> list[dict] | None:
    """Send fragments to LLM and return reconstructed items.

    Returns a list of {"fragment_ids": [...], "title": str, "class": str}.
    Returns None if the LLM call fails.
    """
    frag_text = json.dumps(fragments, ensure_ascii=False, indent=2)
    user_prompt = user_prompt_template.format(fragments=frag_text)

    if prompt_out_path:
        os.makedirs(os.path.dirname(prompt_out_path), exist_ok=True)
        with open(prompt_out_path, "w", encoding="utf-8") as f:
            f.write(f"# System Prompt\n{system_prompt}\n\n# User Prompt\n{user_prompt}")

    for attempt in range(max_retries):
        try:
            raw = client.complete(system_prompt, user_prompt)
        except (APITimeoutError, APIError) as e:
            if _handle_api_error(e, attempt, max_retries):
                continue
            return None

        parsed = _parse_json_response(raw)
        if parsed is not None:
            return _validate_items(parsed, fragments)

        if attempt < max_retries - 1:
            continue

    print(
        f"  Failed to parse LLM response as JSON array after {max_retries} attempts",
        file=sys.stderr,
    )
    return None


def _handle_api_error(e: Exception, attempt: int, max_retries: int) -> bool:
    """Handle API exceptions and return True if should retry."""
    if attempt < max_retries - 1:
        wait = 5 * (2**attempt)
        print(f"  API error: {e}. Retrying in {wait}s...", file=sys.stderr)
        time.sleep(wait)
        return True

    print(f"  API error after {max_retries} attempts: {e}", file=sys.stderr)
    return False


def _parse_classification_response(raw: str) -> dict | None:
    """Extract and parse a JSON dict mapping IDs to classes."""
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Extract from markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Search backwards to find the last valid JSON dict
    end = text.rfind("}")
    if end != -1:
        for start in range(end, -1, -1):
            if text[start] == "{":
                try:
                    result = json.loads(text[start : end + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue

    return None


def _validate_classification(classes: dict, fragments: list[dict]) -> dict:
    """Filter classification dict to only include valid fragment IDs."""
    valid_ids = {f["id"] for f in fragments}
    return {k: v for k, v in classes.items() if k in valid_ids and isinstance(v, str)}


def _parse_json_response(raw: str) -> list[dict] | None:
    """Extract and parse a JSON array from an LLM response.

    Handles markdown fences and surrounding text, ignoring <think> blocks.
    """
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Search backwards to find the last valid JSON array
    end = text.rfind("]")
    if end != -1:
        for start in range(end, -1, -1):
            if text[start] == "[":
                try:
                    result = json.loads(text[start : end + 1])
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue

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
