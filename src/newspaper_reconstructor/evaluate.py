"""Evaluation of LLM article reconstruction against ground truth article XML.

Metrics:
    - Pairwise clustering F1 (precision, recall, F1)
    - Class accuracy on exactly-matched items
    - Coverage (fraction of fragments assigned to any item)

Run logging:
    - JSON file per run with config, per-page results, aggregate metrics
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from itertools import combinations

# Classes in ground truth that fold into "miscellaneous"
_FOLD_CLASSES = {"letter", "caption"}


def parse_article_xml(article_xml_path: str) -> list[dict]:
    """Parse ground truth article XML, returning one entry per article.

    Folds 'letter' and 'caption' classes into 'miscellaneous'.
    """
    tree = ET.parse(article_xml_path)
    root = tree.getroot()

    result = []
    for art in root.findall(".//Article"):
        cls = art.get("class", "miscellaneous")
        if cls in _FOLD_CLASSES:
            cls = "miscellaneous"

        topics = [t.text for t in art.findall(".//Topic") if t.text]

        regions = [r.get("ref") for r in art.findall(".//Region")]

        result.append(
            {
                "uuid": art.get("uuid"),
                "class": cls,
                "fragment_ids": regions,
                "topics": topics,
            }
        )
    return result


def load_ground_truth_dir(article_xml_dir: str) -> dict[str, list[dict]]:
    """Load all article XML files from a directory.

    Returns a dict keyed by page name (filename without extension).
    """
    result = {}
    for fname in sorted(os.listdir(article_xml_dir)):
        if not fname.endswith(".xml"):
            continue
        page_name = os.path.splitext(fname)[0]
        fpath = os.path.join(article_xml_dir, fname)
        result[page_name] = parse_article_xml(fpath)
    return result


def _build_pair_set(items: list[dict]) -> set[tuple[str, str]]:
    """Build the set of co-grouped fragment pairs from a list of items.

    Each pair (i, j) is stored with i < j (sorted) for consistency.
    Single-fragment items produce no pairs.
    """
    pairs = set()
    for item in items:
        fids = sorted(item["fragment_ids"])
        for a, b in combinations(fids, 2):
            pairs.add((a, b))
    return pairs


def clustering_f1(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Compute pairwise clustering F1.

    Treats each item as a cluster of fragment IDs. For each pair of fragments,
    checks if they're in the same cluster in predicted vs ground truth.

    Single-fragment items produce no pairs and don't affect the score.
    If both predicted and truth have no multi-fragment items, F1 = 1.0
    (trivially correct — nothing to get wrong).
    """
    pred_pairs = _build_pair_set(predicted)
    truth_pairs = _build_pair_set(ground_truth)

    if not pred_pairs and not truth_pairs:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "false_positives": [],
            "false_negatives": [],
        }

    tp = len(pred_pairs & truth_pairs)
    fp_pairs = pred_pairs - truth_pairs
    fn_pairs = truth_pairs - pred_pairs
    fp = len(fp_pairs)
    fn = len(fn_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_positives": sorted([list(p) for p in fp_pairs]),
        "false_negatives": sorted([list(p) for p in fn_pairs]),
    }


def _bcubed_f1(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Compute B-cubed precision, recall, and F1.

    For each fragment, B-cubed precision = (correct items in its predicted cluster) / (size of predicted cluster).
    B-cubed recall = (correct items in its ground truth cluster) / (size of ground truth cluster).
    Averages over all fragments.

    Unlike pairwise F1, B-cubed gives partial credit when a cluster is mostly right.
    """
    pred_clusters: dict[str, set] = {}
    for item in predicted:
        fids = item.get("fragment_ids", [])
        for fid in fids:
            pred_clusters[fid] = set(fids)

    truth_clusters: dict[str, set] = {}
    for item in ground_truth:
        fids = item.get("fragment_ids", [])
        for fid in fids:
            truth_clusters[fid] = set(fids)

    all_fragments = set(pred_clusters) | set(truth_clusters)
    if not all_fragments:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    precision_sum = 0.0
    recall_sum = 0.0
    for frag in all_fragments:
        pc = pred_clusters.get(frag, {frag})
        tc = truth_clusters.get(frag, {frag})
        overlap = len(pc & tc)
        precision_sum += overlap / len(pc) if pc else 0.0
        recall_sum += overlap / len(tc) if tc else 0.0

    n = len(all_fragments)
    precision = precision_sum / n
    recall = recall_sum / n
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_classification_page(
    predicted: list[dict], ground_truth: list[dict]
) -> dict:
    """Evaluate classification items against ground truth for a single page.

    Args:
        predicted: list of fragments with 'id' and 'predicted_class'
        ground_truth: list of {"uuid": str, "class": str, "fragment_ids": [...], "topics": [...]}

    Returns metrics dict with weighted precision, recall, and f1.
    """
    truth_map = {}
    for item in ground_truth:
        cls = item["class"]
        for fid in item["fragment_ids"]:
            truth_map[fid] = cls

    tp = {}
    fp = {}
    fn = {}

    classes = set(truth_map.values())
    for item in predicted:
        fid = item["id"]
        pred_c = item.get("predicted_class")
        if pred_c is not None:
            classes.add(pred_c)

        if fid in truth_map:
            true_c = truth_map[fid]
            if pred_c == true_c:
                tp[true_c] = tp.get(true_c, 0) + 1
            else:
                if pred_c is not None:
                    fp[pred_c] = fp.get(pred_c, 0) + 1
                fn[true_c] = fn.get(true_c, 0) + 1

    total_tp = sum(tp.values())
    total_samples = len([f for f in predicted if f["id"] in truth_map])

    total_support = 0
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0
    
    valid_classes = [c for c in classes if c is not None]

    for c in valid_classes:
        c_tp = tp.get(c, 0)
        c_fp = fp.get(c, 0)
        c_fn = fn.get(c, 0)
        
        support = c_tp + c_fn
        total_support += support

        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        weighted_p += p * support
        weighted_r += r * support
        weighted_f1 += f1 * support

    if total_support > 0:
        weighted_p /= total_support
        weighted_r /= total_support
        weighted_f1 /= total_support
    else:
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    return {
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "num_fragments": total_samples,
    }


def evaluate_reconstruction_page(
    predicted: list[dict], ground_truth: list[dict]
) -> dict:
    """Evaluate predicted items against ground truth for a single page.

    Args:
        predicted: list of {"fragment_ids": [...], "title": str, "class": str}
        ground_truth: list of {"uuid": str, "class": str, "fragment_ids": [...], "topics": [...]}

    Returns metrics dict with clustering F1 and coverage.
    """
    cluster_metrics = clustering_f1(predicted, ground_truth)
    bcubed_metrics = _bcubed_f1(predicted, ground_truth)

    all_fragment_ids = set()
    for item in ground_truth:
        all_fragment_ids.update(item["fragment_ids"])
    num_fragments = len(all_fragment_ids)

    assigned_fragments = set()
    for item in predicted:
        assigned_fragments.update(item["fragment_ids"])
    coverage = (
        len(assigned_fragments & all_fragment_ids) / num_fragments
        if num_fragments > 0
        else 0.0
    )

    return {
        "num_fragments": num_fragments,
        "num_predicted_items": len(predicted),
        "num_ground_truth_items": len(ground_truth),
        "clustering_precision": cluster_metrics["precision"],
        "clustering_recall": cluster_metrics["recall"],
        "clustering_f1": cluster_metrics["f1"],
        "bcubed_precision": bcubed_metrics["precision"],
        "bcubed_recall": bcubed_metrics["recall"],
        "bcubed_f1": bcubed_metrics["f1"],
        "tp": cluster_metrics["tp"],
        "fp": cluster_metrics["fp"],
        "fn": cluster_metrics["fn"],
        "false_positives": cluster_metrics["false_positives"],
        "false_negatives": cluster_metrics["false_negatives"],
        "coverage": coverage,
    }


def log_evaluation_run(
    results: list[dict],
    config: dict,
    output_dir: str,
    run_id: str | None = None,
) -> str:
    """Log an evaluation run to a JSON file for reproducibility.

    Args:
        results: list of per-page result dicts (page_id, metrics, predicted_items, ground_truth_items)
        config: run configuration (provider, model, prompts, etc.)
        output_dir: directory to write the log file
        run_id: optional run identifier to use instead of generating one

    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().astimezone()

    if not run_id:
        run_id = config.get("run_id")

    if not run_id:
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        provider = config.get("provider", "unknown")
        model = config.get("model", "unknown")
        prompt_name = config.get("prompt_name", "default")
        sample_size = config.get("sample_size")
        seed = config.get("seed")
        run_id = f"{timestamp_str}_{provider}_{model}_{prompt_name}"
        if sample_size is not None:
            run_id = f"{run_id}_sample{sample_size}"
        if seed is not None:
            run_id = f"{run_id}_seed{seed}"

    # Compute aggregate metrics based on task
    task = config.get("task", "reconstruction")

    if task == "classification":
        weighted_ps = [r["metrics"]["weighted_precision"] for r in results]
        weighted_rs = [r["metrics"]["weighted_recall"] for r in results]
        weighted_f1s = [r["metrics"]["weighted_f1"] for r in results]

        aggregate = {
            "mean_weighted_precision": sum(weighted_ps) / len(weighted_ps) if weighted_ps else 0.0,
            "mean_weighted_recall": sum(weighted_rs) / len(weighted_rs) if weighted_rs else 0.0,
            "mean_weighted_f1": sum(weighted_f1s) / len(weighted_f1s) if weighted_f1s else 0.0,
            "total_pages": len(results),
        }
    else:
        f1s = [r["metrics"]["clustering_f1"] for r in results]
        bcubed_f1s = [r["metrics"]["bcubed_f1"] for r in results]
        coverages = [r["metrics"]["coverage"] for r in results]

        aggregate = {
            "mean_clustering_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "mean_bcubed_f1": sum(bcubed_f1s) / len(bcubed_f1s) if bcubed_f1s else 0.0,
            "mean_clustering_precision": sum(
                r["metrics"]["clustering_precision"] for r in results
            )
            / len(results)
            if results
            else 0.0,
            "mean_clustering_recall": sum(
                r["metrics"]["clustering_recall"] for r in results
            )
            / len(results)
            if results
            else 0.0,
            "mean_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
            "total_pages": len(results),
        }

    log = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "config": config,
        "pages": results,
        "aggregate": aggregate,
    }

    filename = f"{run_id}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return path
