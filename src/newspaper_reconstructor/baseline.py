"""Deterministic baselines for fragment-to-article reconstruction.

Variants:

- ``geometry``: XY-cut-style gap segmentation. Each fragment casts a ray
  downward and links to its nearest successor whose horizontal extent
  overlaps enough and whose vertical gap is within a threshold (a multiple of
  the page's median fragment height). Articles are the connected components
  of that adjacency graph. Each fragment links to at most one successor, so a
  wide headline cannot chain-merge the articles below it.
- ``geometry-ncd``: as ``geometry``, but successor candidates must also pass
  a normalized compression distance (NCD) test between the two fragments'
  texts.
- ``xy-cut``: classic recursive XY-cut (Nagy et al.). Splits a group at
  every vertical whitespace band that spans the group's full height (X cut,
  width ≥ gap_k × median fragment width), else at every horizontal band
  spanning its full width (Y cut, height ≥ gap_k × median fragment height);
  leaves are articles. Global whitespace analysis — stricter than the
  raycast's local adjacency, and a spanning fragment (e.g. a banner) blocks
  the cut it overlaps, the classic XY-cut failure mode.
- ``xy-cut-plus``: XY-Cut++-inspired (Liu et al., arXiv:2504.10258), adapted
  from reading-order recovery to clustering. Three of their components:
  (1) cross-layout masking — spanners (width > 1.3 × median width with
  horizontal overlap against ≥ 2 blocks) are excluded from the splitting
  projections and remapped afterwards by highest overlap, with centre
  distance as the tie-breaker;
  (2) density-driven axis selection — τ_d = cross-area / single-area in the
  current group, with τ_d > 0.9 preferring a horizontal (Y) split;
  (3) min-projection splits — binary split at the widest uncovered band on
  the chosen axis. Masking is what opens the column gutters that classic
  XY-cut cannot see.
- ``singletons`` / ``one-cluster``: trivial partitions for context.

Baseline commands consume raw module-format OCR. Text-bearing regions are
article candidates; all regions, including images and empty-text regions,
remain as layout context for geometric segmentation but never appear in the
predicted fragment ids.

Reading order within an item is column-based, right-to-left (Jawi RTL),
top-to-bottom within a column. Clustering metrics are order-invariant; the
ordering only affects human-readable output. Column detection uses
transitive horizontal overlap, so pages with wide spanning fragments (e.g.
banners) can interleave visual columns in the emitted reading order.

Default parameters are fixed operating points documented by the local
sensitivity sweep in ``reports/baselines/sweep.md``.

Known limitation: articles continuing across columns are not reconstructed.
"""

from __future__ import annotations

import functools
import statistics
import zlib

DEFAULT_X_OVERLAP = 0.5
DEFAULT_GAP_K = 0.01
DEFAULT_NCD_THRESHOLD = 0.9
DEFAULT_NCD_MIN_CHARS = 20

VARIANTS = (
    "geometry",
    "geometry-ncd",
    "xy-cut",
    "xy-cut-plus",
    "singletons",
    "one-cluster",
)

BETA_CROSS = 1.3
THETA_V = 0.9
_GEOMETRY_KEYS = ("hpos", "vpos", "width", "height")


def baseline_items(
    fragments: list[dict],
    variant: str = "geometry",
    gap_k: float = DEFAULT_GAP_K,
    x_overlap_threshold: float = DEFAULT_X_OVERLAP,
    ncd_threshold: float = DEFAULT_NCD_THRESHOLD,
    ncd_min_chars: int = DEFAULT_NCD_MIN_CHARS,
    layout_regions: list[dict] | None = None,
) -> list[dict]:
    """Cluster text fragments using complete page geometry as layout context.

    All variants require fragments carrying hpos/vpos/width/height. Layout
    regions may additionally contain images and regions without OCR text; they
    affect geometric segmentation but never appear in predicted items.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    if not gap_k >= 0:
        raise ValueError("gap_k must be non-negative")
    if not 0 <= x_overlap_threshold <= 1:
        raise ValueError("x_overlap_threshold must be between 0 and 1")
    if not ncd_threshold >= 0:
        raise ValueError("ncd_threshold must be non-negative")
    if not ncd_min_chars >= 0:
        raise ValueError("ncd_min_chars must be non-negative")
    _require_geometry(fragments)
    _require_ids(fragments, "candidate fragment")
    if variant == "geometry-ncd" and any("text" not in f for f in fragments):
        raise ValueError(
            "geometry-ncd requires candidate fragments with a 'text' field"
        )
    duplicate_candidates = _duplicate_ids(fragments)
    if duplicate_candidates:
        raise ValueError(
            f"duplicate candidate fragment ids: {', '.join(duplicate_candidates)}"
        )
    layout_regions = fragments if layout_regions is None else layout_regions
    _require_geometry(layout_regions)
    _require_ids(layout_regions, "layout region")
    duplicate_layout = _duplicate_ids(layout_regions)
    if duplicate_layout:
        raise ValueError(f"duplicate layout region ids: {', '.join(duplicate_layout)}")
    layout_ids = {region["id"] for region in layout_regions}
    missing_layout = sorted(f["id"] for f in fragments if f["id"] not in layout_ids)
    if missing_layout:
        raise ValueError(
            "layout_regions must include every candidate fragment; "
            f"missing ids: {', '.join(missing_layout)}"
        )
    if not fragments:
        return []
    if variant == "singletons":
        return [
            {"fragment_ids": [f["id"]], "title": "", "class": ""}
            for f in reading_order(fragments)
        ]
    if variant == "one-cluster":
        return [
            {
                "fragment_ids": [f["id"] for f in reading_order(fragments)],
                "title": "",
                "class": "",
            }
        ]
    if variant == "xy-cut":
        return _xy_cut_items(fragments, layout_regions, gap_k)
    if variant == "xy-cut-plus":
        return _xy_cut_plus_items(fragments, layout_regions, gap_k)
    if variant == "geometry-ncd":

        def text_pass(a: str, b: str) -> bool:
            return ncd_pass(a, b, ncd_threshold, ncd_min_chars)

        return _geometry_items(
            fragments, layout_regions, gap_k, x_overlap_threshold, text_pass
        )
    return _geometry_items(fragments, layout_regions, gap_k, x_overlap_threshold)


def _require_geometry(fragments: list[dict]) -> None:
    """Fail fast when fragments lack the geometry every variant needs."""
    missing = sorted({key for f in fragments for key in _GEOMETRY_KEYS if key not in f})
    if missing:
        raise ValueError(
            "baseline variants require fragments with hpos/vpos/width/height "
            "(module-format OCR or ALTO-derived fragments); "
            f"missing keys: {', '.join(missing)}"
        )


def _require_ids(regions: list[dict], kind: str) -> None:
    if any("id" not in region for region in regions):
        raise ValueError(f"every {kind} must carry an 'id'")


def _duplicate_ids(regions: list[dict]) -> list[str]:
    seen = set()
    duplicates = set()
    for region in regions:
        region_id = region["id"]
        if region_id in seen:
            duplicates.add(region_id)
        seen.add(region_id)
    return sorted(duplicates)


def reading_order(fragments: list[dict]) -> list[dict]:
    """Full-page reading order: columns right-to-left, top-to-bottom."""
    ordered: list[dict] = []
    for column in group_columns(fragments):
        ordered.extend(column)
    return ordered


def group_columns(
    fragments: list[dict], x_overlap_threshold: float = DEFAULT_X_OVERLAP
) -> list[list[dict]]:
    """Group fragments into columns by horizontal-extent overlap.

    Columns are returned right-to-left (RTL reading order); fragments within
    a column are sorted top-to-bottom, ties broken by hpos then id.
    """

    def order_key(fragment: dict) -> tuple:
        return fragment["vpos"], fragment["hpos"], fragment["id"]

    remaining = sorted(fragments, key=order_key)
    columns: list[list[dict]] = []
    while remaining:
        column = [remaining.pop(0)]
        frontier = list(column)
        while frontier:
            member = frontier.pop()
            i = 0
            while i < len(remaining):
                if _x_overlap_fraction(member, remaining[i]) >= x_overlap_threshold:
                    added = remaining.pop(i)
                    column.append(added)
                    frontier.append(added)
                else:
                    i += 1
        column.sort(key=order_key)
        columns.append(column)
    columns.sort(key=lambda c: max(f["hpos"] + f["width"] for f in c), reverse=True)
    return columns


def _x_overlap_fraction(a: dict, b: dict) -> float:
    a_left, a_right = a["hpos"], a["hpos"] + a["width"]
    b_left, b_right = b["hpos"], b["hpos"] + b["width"]
    overlap = min(a_right, b_right) - max(a_left, b_left)
    if overlap <= 0:
        return 0.0
    return overlap / min(a["width"], b["width"])


def _geometry_items(
    fragments: list[dict],
    layout_regions: list[dict],
    gap_k: float,
    x_overlap_threshold: float,
    text_pass=None,
) -> list[dict]:
    heights = [f["height"] for f in fragments if f["height"] > 0]
    median_height = statistics.median(heights) if heights else 1.0
    gap_max = gap_k * median_height
    edges = []
    for fragment in fragments:
        successor = _raycast_successor(
            fragment,
            fragments,
            layout_regions,
            gap_max,
            x_overlap_threshold,
            text_pass,
        )
        if successor is not None:
            edges.append((fragment["id"], successor["id"]))
    items = []
    for component in _components(fragments, edges):
        ordered = [f["id"] for f in reading_order(component)]
        items.append({"fragment_ids": ordered, "title": "", "class": ""})
    return items


def _raycast_successor(
    fragment: dict,
    fragments: list[dict],
    layout_regions: list[dict],
    gap_max: float,
    x_overlap_threshold: float,
    text_pass=None,
) -> dict | None:
    """Nearest fragment below `fragment` reachable by a downward ray.

    A geometric tie between successor candidates is ambiguous and yields no
    link, so the result never depends on region id ordering.
    """
    best: dict | None = None
    best_key: tuple | None = None
    tied = False
    for other in fragments:
        if other["vpos"] <= fragment["vpos"]:
            continue
        overlap = _x_overlap_fraction(fragment, other)
        if overlap < x_overlap_threshold:
            continue
        gap = max(0, other["vpos"] - (fragment["vpos"] + fragment["height"]))
        if gap > gap_max:
            continue
        key = (gap, other["vpos"], -overlap)
        if best_key is None or key < best_key:
            best, best_key = other, key
            tied = False
        elif key == best_key:
            tied = True
    if best is None or tied:
        return None
    candidate_ids = {candidate["id"] for candidate in fragments}
    for obstacle in layout_regions:
        if obstacle["id"] in candidate_ids:
            continue
        if obstacle["vpos"] <= fragment["vpos"]:
            continue
        overlap = _x_overlap_fraction(fragment, obstacle)
        if overlap < x_overlap_threshold:
            continue
        gap = max(0, obstacle["vpos"] - (fragment["vpos"] + fragment["height"]))
        if gap > gap_max:
            continue
        if (gap, obstacle["vpos"], -overlap) <= best_key:
            return None
    if text_pass is not None and not text_pass(fragment["text"], best["text"]):
        return None
    return best


def _components(
    fragments: list[dict], edges: list[tuple[str, str]]
) -> list[list[dict]]:
    by_id = {f["id"]: f for f in fragments}
    adjacency: dict[str, set[str]] = {f["id"]: set() for f in fragments}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[str] = set()
    components = []
    for fragment in reading_order(fragments):
        if fragment["id"] in seen:
            continue
        stack = [fragment["id"]]
        component_ids = []
        seen.add(fragment["id"])
        while stack:
            current = stack.pop()
            component_ids.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append([by_id[fid] for fid in component_ids])
    return components


def ncd_pass(
    a: str,
    b: str,
    threshold: float = DEFAULT_NCD_THRESHOLD,
    min_chars: int = DEFAULT_NCD_MIN_CHARS,
) -> bool:
    """Whether two texts are similar enough to merge.

    Fragments shorter than min_chars are too small for a stable NCD estimate
    and pass neutrally.
    """
    if len(a.strip()) < min_chars or len(b.strip()) < min_chars:
        return True
    return ncd(a, b) <= threshold


@functools.lru_cache(maxsize=1 << 18)
def ncd(a: str, b: str) -> float:
    """Normalized compression distance between two strings.

    Uses the cheaper concatenation order for symmetry. Values near 0 mean
    highly similar; values near or above 1 mean unrelated. Results are
    cached (bounded) so parameter sweeps reusing the same fragment pairs
    skip recompression.
    """
    ca, cb = _compressed_len(a), _compressed_len(b)
    denom = max(ca, cb)
    if denom == 0:
        return 0.0
    c_ab = min(_compressed_len(a + b), _compressed_len(b + a))
    return max(0.0, (c_ab - min(ca, cb)) / denom)


def _compressed_len(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8"), 9))


def _xy_cut_items(
    fragments: list[dict], layout_regions: list[dict], gap_k: float
) -> list[dict]:
    widths = [f["width"] for f in fragments if f["width"] > 0]
    heights = [f["height"] for f in fragments if f["height"] > 0]
    x_min = gap_k * (statistics.median(widths) if widths else 1.0)
    y_min = gap_k * (statistics.median(heights) if heights else 1.0)
    order_index = {f["id"]: i for i, f in enumerate(reading_order(fragments))}
    candidate_ids = set(order_index)
    items = []
    for leaf in _xy_cut_leaves(layout_regions, x_min, y_min):
        ordered = sorted(
            (fragment for fragment in leaf if fragment["id"] in candidate_ids),
            key=lambda fragment: order_index[fragment["id"]],
        )
        if ordered:
            items.append(
                {
                    "fragment_ids": [f["id"] for f in ordered],
                    "title": "",
                    "class": "",
                }
            )
    items.sort(key=lambda item: order_index[item["fragment_ids"][0]])
    return items


def _xy_cut_leaves(
    fragments: list[dict], x_min: float, y_min: float
) -> list[list[dict]]:
    for axis, min_width in (("x", x_min), ("y", y_min)):
        groups = _split_at_gaps(fragments, axis, min_width)
        if groups is not None:
            leaves: list[list[dict]] = []
            for group in groups:
                leaves.extend(_xy_cut_leaves(group, x_min, y_min))
            return leaves
    return [fragments]


def _split_at_gaps(
    fragments: list[dict], axis: str, min_width: float
) -> list[list[dict]] | None:
    """Split fragments at every uncovered band of at least min_width.

    Returns None when no qualifying gap exists on the axis.
    """
    bands: list[list[float]] = []
    for start, end in sorted(_axis_interval(f, axis) for f in fragments):
        if bands and start <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], end)
        else:
            bands.append([start, end])
    gaps = [
        (bands[i][1], bands[i + 1][0])
        for i in range(len(bands) - 1)
        if bands[i + 1][0] - bands[i][1] >= min_width
    ]
    if not gaps:
        return None
    groups: list[list[dict]] = [[] for _ in range(len(gaps) + 1)]
    for fragment in fragments:
        end = _axis_interval(fragment, axis)[1]
        for i, (gap_start, _) in enumerate(gaps):
            if end <= gap_start:
                groups[i].append(fragment)
                break
        else:
            groups[len(gaps)].append(fragment)
    return groups


def _axis_interval(fragment: dict, axis: str) -> tuple[float, float]:
    if axis == "x":
        return fragment["hpos"], fragment["hpos"] + fragment["width"]
    return fragment["vpos"], fragment["vpos"] + fragment["height"]


def _xy_cut_plus_items(
    fragments: list[dict], layout_regions: list[dict], gap_k: float
) -> list[dict]:
    widths = [f["width"] for f in fragments if f["width"] > 0]
    heights = [f["height"] for f in fragments if f["height"] > 0]
    x_min = gap_k * (statistics.median(widths) if widths else 1.0)
    y_min = gap_k * (statistics.median(heights) if heights else 1.0)
    cross_ids = detect_cross_layout(layout_regions)
    order_index = {f["id"]: i for i, f in enumerate(reading_order(fragments))}
    candidate_ids = set(order_index)
    leaves = _xy_cut_plus_leaves(layout_regions, cross_ids, x_min, y_min)
    items = []
    for leaf in leaves:
        ordered = sorted(
            (fragment for fragment in leaf if fragment["id"] in candidate_ids),
            key=lambda fragment: order_index[fragment["id"]],
        )
        if ordered:
            items.append(
                {
                    "fragment_ids": [f["id"] for f in ordered],
                    "title": "",
                    "class": "",
                }
            )
    items.sort(key=lambda item: order_index[item["fragment_ids"][0]])
    return items


def detect_cross_layout(fragments: list[dict], beta: float = BETA_CROSS) -> set[str]:
    """Ids of cross-layout (spanning) fragments, after XY-Cut++ phase 1.

    A fragment is cross-layout when its width exceeds beta × the median
    width and it overlaps at least two other fragments horizontally.
    """
    widths = [f["width"] for f in fragments if f["width"] > 0]
    if not widths:
        return set()
    threshold = beta * statistics.median(widths)
    cross = set()
    for fragment in fragments:
        if fragment["width"] <= threshold:
            continue
        overlaps = sum(
            1
            for other in fragments
            if other["id"] != fragment["id"] and _x_intervals_overlap(fragment, other)
        )
        if overlaps >= 2:
            cross.add(fragment["id"])
    return cross


def _x_intervals_overlap(a: dict, b: dict) -> bool:
    a_start, a_end = a["hpos"], a["hpos"] + a["width"]
    b_start, b_end = b["hpos"], b["hpos"] + b["width"]
    return min(a_end, b_end) - max(a_start, b_start) > 0


def _xy_cut_plus_leaves(
    fragments: list[dict], cross_ids: set[str], x_min: float, y_min: float
) -> list[list[dict]]:
    singles = [f for f in fragments if f["id"] not in cross_ids]
    if len(singles) <= 1:
        return [fragments]
    for axis in _axis_order(fragments, cross_ids):
        min_width = y_min if axis == "y" else x_min
        split = _split_at_widest_band(singles, axis, min_width)
        if split is None:
            continue
        groups = [list(split[0]), list(split[1])]
        _remap_cross([f for f in fragments if f["id"] in cross_ids], groups)
        leaves: list[list[dict]] = []
        for group in groups:
            leaves.extend(_xy_cut_plus_leaves(group, cross_ids, x_min, y_min))
        return leaves
    return [fragments]


def _axis_order(fragments: list[dict], cross_ids: set[str]) -> tuple[str, str]:
    """Axis priority: horizontal (Y) first when cross-layout area dominates."""
    return ("y", "x") if _tau_d(fragments, cross_ids) > THETA_V else ("x", "y")


def _tau_d(fragments: list[dict], cross_ids: set[str]) -> float:
    """Cross-layout area over single-layout area in a group (XY-Cut++ tau_d)."""
    singles = [f for f in fragments if f["id"] not in cross_ids]
    cross = [f for f in fragments if f["id"] in cross_ids]
    single_area = _area(singles)
    if not single_area or not cross:
        return 0.0
    return _area(cross) / single_area


def _area(fragments: list[dict]) -> float:
    return sum(f["width"] * f["height"] for f in fragments)


def _split_at_widest_band(
    fragments: list[dict], axis: str, min_width: float
) -> tuple[list[dict], list[dict]] | None:
    """Binary split at the widest uncovered band (XY-Cut++ min projection)."""
    bands = _uncovered_bands(fragments, axis, min_width)
    if not bands:
        return None
    gap_start, _ = bands[0]
    left, right = [], []
    for fragment in fragments:
        if _axis_interval(fragment, axis)[1] <= gap_start:
            left.append(fragment)
        else:
            right.append(fragment)
    if not left or not right:
        return None
    return left, right


def _uncovered_bands(
    fragments: list[dict], axis: str, min_width: float
) -> list[tuple[float, float]]:
    """Maximal bands on the axis covered by no fragment, widest first."""
    bands: list[list[float]] = []
    for start, end in sorted(_axis_interval(f, axis) for f in fragments):
        if bands and start <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], end)
        else:
            bands.append([start, end])
    gaps = [
        (bands[i][1], bands[i + 1][0])
        for i in range(len(bands) - 1)
        if bands[i + 1][0] - bands[i][1] >= min_width
    ]
    return sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)


def _remap_cross(fragments: list[dict], groups: list[list[dict]]) -> None:
    """Assign each fragment by highest overlap, then nearest centre."""
    for fragment in fragments:
        best = max(groups, key=lambda g: _remap_score(fragment, g))
        best.append(fragment)


def _remap_score(fragment: dict, group: list[dict]) -> tuple:
    if not group:
        return (0.0, 0.0)
    overlap = max(_fragment_intersection_over_union(fragment, other) for other in group)
    distance = min(_centre_distance(fragment, other) for other in group)
    return (overlap, -distance)


def _fragment_intersection_over_union(a: dict, b: dict) -> float:
    ix = min(a["hpos"] + a["width"], b["hpos"] + b["width"]) - max(a["hpos"], b["hpos"])
    iy = min(a["vpos"] + a["height"], b["vpos"] + b["height"]) - max(
        a["vpos"], b["vpos"]
    )
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _centre_distance(a: dict, b: dict) -> float:
    ax = a["hpos"] + a["width"] / 2
    ay = a["vpos"] + a["height"] / 2
    bx = b["hpos"] + b["width"] / 2
    by = b["vpos"] + b["height"] / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
