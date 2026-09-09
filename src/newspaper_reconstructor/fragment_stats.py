"""Statistics for OCR fragment files: corpus profile, script mix, OCR garbage.

Used to characterise fragment sets (whole datasets or individual pages) and to
flag pages whose text is mostly ASCII rather than Jawi, or that contain the
mixed-script garbage that makes reasoning models loop.
"""

import json
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, fields
from itertools import pairwise

ASCII_DOMINANT_THRESHOLD = 0.5
GIBBERISH_MIN_CHARS = 20
GIBBERISH_LATIN_DIGIT_RATIO = 0.30
GIBBERISH_SCRIPT_SWITCH_RATE = 0.20
REPEAT_WINDOW = 12


def _script(ch: str) -> str:
    o = ord(ch)
    if o < 0x80:
        if ch.isalpha():
            return "latin"
        if ch.isdigit():
            return "digit"
        return "punct"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFEFF:
        return "jawi"
    return "other"


def script_ratios(text: str) -> dict[str, float]:
    """Share of non-whitespace characters by script class.

    Keys: ascii, jawi, latin, digit (latin/digit are ASCII subsets), other.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"ascii": 0.0, "jawi": 0.0, "latin": 0.0, "digit": 0.0, "other": 0.0}
    scripts = [_script(c) for c in chars]
    n = len(scripts)
    return {
        "ascii": sum(1 for s in scripts if s in ("latin", "digit", "punct")) / n,
        "jawi": scripts.count("jawi") / n,
        "latin": scripts.count("latin") / n,
        "digit": scripts.count("digit") / n,
        "other": scripts.count("other") / n,
    }


def _max_repeat_window(text: str, k: int = REPEAT_WINDOW) -> int:
    """Highest number of occurrences of any k-character window (loopiness)."""
    if len(text) < k * 2:
        return 0
    return Counter(text[i : i + k] for i in range(len(text) - k + 1)).most_common(1)[0][
        1
    ]


def _is_gibberish(text: str) -> bool:
    """True if a fragment looks like mixed-script OCR garbage."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < GIBBERISH_MIN_CHARS:
        return False
    scripts = [_script(c) for c in chars]
    latin_digit = sum(1 for s in scripts if s in ("latin", "digit")) / len(scripts)
    switches = sum(1 for a, b in pairwise(scripts) if a != b) / (len(scripts) - 1)
    return (
        latin_digit >= GIBBERISH_LATIN_DIGIT_RATIO
        and switches >= GIBBERISH_SCRIPT_SWITCH_RATE
    )


@dataclass
class PageStats:
    page: str
    n_fragments: int
    chars: int
    prompt_chars: int
    max_fragment_chars: int
    mean_fragment_chars: float
    jawi_ratio: float
    ascii_ratio: float
    latin_ratio: float
    digit_ratio: float
    whitespace_ratio: float
    mean_word_chars: float
    longest_word_chars: int
    gibberish_fragments: int
    gibberish_chars: int
    duplicate_fragments: int
    max_repeat_window: int

    def is_ascii_dominant(self, threshold: float = ASCII_DOMINANT_THRESHOLD) -> bool:
        return self.ascii_ratio >= threshold


def page_stats(page: str, fragments: list[dict]) -> PageStats:
    """Compute statistics for one page's fragment list."""
    texts = [f.get("text") or "" for f in fragments]
    joined = "\n".join(texts)
    words = joined.split()
    ratios = script_ratios(joined)
    lengths = [len(t) for t in texts]
    gibberish = [t for t in texts if _is_gibberish(t)]

    return PageStats(
        page=page,
        n_fragments=len(fragments),
        chars=sum(lengths),
        prompt_chars=len(json.dumps(fragments, ensure_ascii=False, indent=2)),
        max_fragment_chars=max(lengths, default=0),
        mean_fragment_chars=statistics.mean(lengths) if lengths else 0.0,
        jawi_ratio=ratios["jawi"],
        ascii_ratio=ratios["ascii"],
        latin_ratio=ratios["latin"],
        digit_ratio=ratios["digit"],
        whitespace_ratio=sum(1 for c in joined if c.isspace()) / max(len(joined), 1),
        mean_word_chars=statistics.mean([len(w) for w in words]) if words else 0.0,
        longest_word_chars=max((len(w) for w in words), default=0),
        gibberish_fragments=len(gibberish),
        gibberish_chars=sum(len(t) for t in gibberish),
        duplicate_fragments=len(texts) - len(set(texts)),
        max_repeat_window=max((_max_repeat_window(t) for t in texts), default=0),
    )


def load_page_stats(path: str) -> PageStats:
    """Load a fragment JSON file and compute its statistics."""
    with open(path, encoding="utf-8") as f:
        fragments = json.load(f)
    return page_stats(os.path.splitext(os.path.basename(path))[0], fragments)


def iter_fragment_files(paths: list[str]) -> list[str]:
    """Expand file/directory paths into fragment JSON file paths.

    Directories contribute their *.json files, ignoring '_'-prefixed files
    (e.g. _metadata.json).
    """
    found: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            found.extend(
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.endswith(".json") and not name.startswith("_")
            )
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return found


_NUMERIC_FIELDS = tuple(
    f.name for f in fields(PageStats) if f.name != "page" and f.type in (int, float)
)


def summarize(
    pages: list[PageStats], ascii_threshold: float = ASCII_DOMINANT_THRESHOLD
) -> dict:
    """Corpus-level summary: per-field mean/median/min/max plus flag counts."""
    summary: dict = {"pages": len(pages)}
    for name in _NUMERIC_FIELDS:
        values = [getattr(p, name) for p in pages]
        if not values:
            continue
        summary[name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    summary["ascii_dominant_pages"] = sum(
        1 for p in pages if p.is_ascii_dominant(ascii_threshold)
    )
    summary["pages_with_gibberish"] = sum(1 for p in pages if p.gibberish_fragments)
    return summary


def _fmt(value: float) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _corpus_table(pages: list[PageStats], ascii_threshold: float) -> list[str]:
    summary = summarize(pages, ascii_threshold)
    lines = [
        f"{'field':22s} {'mean':>10s} {'median':>10s} {'min':>10s} {'max':>10s}",
        "-" * 66,
    ]
    for name in _NUMERIC_FIELDS:
        if name not in summary:
            continue
        s = summary[name]
        lines.append(
            f"{name:22s} {_fmt(s['mean']):>10s} {_fmt(s['median']):>10s} "
            f"{_fmt(s['min']):>10s} {_fmt(s['max']):>10s}"
        )
    lines.append("")
    lines.append(f"pages: {summary['pages']}")
    lines.append(
        f"ascii-dominant pages (ascii_ratio >= {ascii_threshold}): "
        f"{summary['ascii_dominant_pages']}"
    )
    lines.append(
        f"pages with >=1 gibberish fragment: {summary['pages_with_gibberish']}"
    )
    return lines


def _page_table(pages: list[PageStats]) -> list[str]:
    header = (
        f"{'page':22s} {'n':>4s} {'chars':>7s} {'prompt':>7s} {'maxfrag':>7s} "
        f"{'jawi':>6s} {'ascii':>6s} {'gib':>4s} {'dup':>4s} {'rep':>4s}"
    )
    lines = [header, "-" * len(header)]
    for p in pages:
        lines.append(
            f"{p.page:22s} {p.n_fragments:4d} {p.chars:7d} {p.prompt_chars:7d} "
            f"{p.max_fragment_chars:7d} {p.jawi_ratio:6.2f} {p.ascii_ratio:6.2f} "
            f"{p.gibberish_fragments:4d} {p.duplicate_fragments:4d} "
            f"{p.max_repeat_window:4d}"
        )
    return lines


def _split_group(
    pages: list[PageStats], group: set[str]
) -> tuple[list[PageStats], list[PageStats]]:
    ids = {os.path.splitext(os.path.basename(g))[0] for g in group}
    in_group = [p for p in pages if p.page in ids]
    rest = [p for p in pages if p.page not in ids]
    if not in_group:
        raise ValueError(
            f"no pages match any of the {len(ids)} group id(s) "
            f"(e.g. {min(ids)!r}); page ids look like {pages[0].page!r}"
        )
    if not rest:
        raise ValueError(
            f"group matches all {len(pages)} pages; nothing to compare against"
        )
    return in_group, rest


def _group_table(in_group: list[PageStats], rest: list[PageStats]) -> list[str]:
    a, b = summarize(in_group), summarize(rest)
    lines = [
        "",
        f"Group (n={len(in_group)}) vs rest (n={len(rest)}):",
        (
            f"{'field':22s} {'group mean':>12s} {'group med':>11s} "
            f"{'rest mean':>11s} {'rest med':>10s}"
        ),
        "-" * 70,
    ]
    for name in _NUMERIC_FIELDS:
        if name not in a or name not in b:
            continue
        lines.append(
            f"{name:22s} {_fmt(a[name]['mean']):>12s} {_fmt(a[name]['median']):>11s} "
            f"{_fmt(b[name]['mean']):>11s} {_fmt(b[name]['median']):>10s}"
        )
    return lines


def render_report(
    pages: list[PageStats],
    ascii_threshold: float = ASCII_DOMINANT_THRESHOLD,
    top: int = 20,
    sort_by: str = "chars",
    group: set[str] | None = None,
) -> str:
    """Render the text stats report for a set of pages."""
    ordered = sorted(pages, key=lambda p: getattr(p, sort_by), reverse=True)
    lines = [f"Fragment stats — {len(pages)} page(s)", ""]
    lines += _corpus_table(pages, ascii_threshold)
    lines.append("")

    ascii_pages = [p for p in ordered if p.is_ascii_dominant(ascii_threshold)]
    if ascii_pages:
        lines.append(f"ASCII-dominant pages (ascii_ratio >= {ascii_threshold}):")
        lines += _page_table(ascii_pages[:top])
        lines.append("")

    if group:
        in_group, rest = _split_group(pages, group)
        group_pages = sorted(in_group, key=lambda p: getattr(p, sort_by), reverse=True)
        lines.append(f"Group pages ({len(group_pages)}), by {sort_by}:")
        lines += _page_table(group_pages)
        lines += _group_table(in_group, rest)
    else:
        lines.append(f"Top {min(top, len(ordered))} pages by {sort_by}:")
        lines += _page_table(ordered[:top])
    return "\n".join(lines)


def to_csv_rows(pages: list[PageStats]) -> list[dict]:
    """Per-page rows for CSV export."""
    return [asdict(p) for p in pages]
