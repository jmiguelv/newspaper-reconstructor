"""Offline analysis tool to suggest prompt and heuristic improvements using an LLM Judge."""

import json
import os
import sys
from pathlib import Path


def load_run_data(experiment_id: str, eval_dir: str) -> dict:
    """Load the JSON evaluation log for a specific run ID."""
    log_path = os.path.join(eval_dir, f"{experiment_id}.json")
    if not os.path.exists(log_path):
        print(f"Error: Evaluation log not found at {log_path}", file=sys.stderr)
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_fragments_dir(run_data: dict) -> str | None:
    """Infer the dataset fragments dir from the eval log's input_folder.

    Works for both data/1_interim/<dataset>/<stage>/<exp> and the legacy
    data/1_interim/<stage>/<exp> layouts. Returns None if input_folder is
    absent from the config.
    """
    input_folder = run_data.get("config", {}).get("input_folder")
    if not input_folder:
        return None
    return str(Path(input_folder).parent.parent / "fragments")


def identify_worst_pages(
    run_data: dict, top_k: int = 3, focus: str = "clustering"
) -> list:
    """Identify the worst performing pages based on focus metric."""
    pages = run_data.get("pages", [])

    # Filter out pages that failed reconstruction or don't have metrics
    valid_pages = [p for p in pages if p.get("metrics")]

    if not valid_pages:
        return []

    if focus == "classification":
        valid_pages.sort(key=lambda x: x["metrics"].get("weighted_f1", 1.0))
    elif focus == "both":
        valid_pages.sort(
            key=lambda x: (
                (
                    x["metrics"].get("clustering_f1", 1.0)
                    + x["metrics"].get("weighted_f1", 1.0)
                )
                / 2
            )
        )
    else:
        valid_pages.sort(key=lambda x: x["metrics"].get("clustering_f1", 1.0))
    return valid_pages[:top_k]


def build_judge_prompt(
    run_data: dict,
    worst_pages: list,
    focus: str = "clustering",
    fragments_dir: str | None = None,
) -> str:
    """Construct the prompt for the LLM Judge."""

    config = run_data.get("config", {}) if isinstance(run_data, dict) else {}
    system_prompt = config.get("system_prompt", "N/A")
    user_prompt = config.get("user_prompt_template", "N/A")

    if focus == "classification":
        prompt_focus = "analyze the mismatch between the predicted item classes and the ground truth classes, and provide systemic suggestions for how to improve the classification prompts or heuristics."
    elif focus == "both":
        prompt_focus = "analyze the mismatch between the predicted fragment clusters/classes and the ground truth clusters/classes, and provide systemic suggestions for how to improve the prompts or heuristics."
    else:
        prompt_focus = "analyze the mismatch between the predicted fragment clusters and the ground truth clusters, and provide systemic suggestions for how to improve the prompts or heuristics."

    prompt = f"""You are an expert AI system evaluator and prompt engineer.
We are using an LLM to reconstruct newspaper articles from OCR text fragments in Jawi Malay.
The fragments must be grouped into complete items (articles, advertisements, etc.).

We recently ran an evaluation and some pages performed poorly. I need you to {prompt_focus}

### Current Prompts Used
**System Prompt:**
```
{system_prompt}
```

**User Prompt Template:**
```
{user_prompt}
```

    ### Error Analysis (Worst Performing Pages)
"""

    for page in worst_pages:
        page_id = page.get("page_id")

        if focus == "classification":
            metric_val = page["metrics"].get("weighted_f1", 0)
            prompt += f"\n#### Page: {page_id} (Classification F1: {metric_val:.3f})\n"
        elif focus == "both":
            f1 = page["metrics"].get("clustering_f1", 0)
            cls_f1 = page["metrics"].get("weighted_f1", 0)
            prompt += f"\n#### Page: {page_id} (Clustering F1: {f1:.3f}, Classification F1: {cls_f1:.3f})\n"
        else:
            f1 = page["metrics"].get("clustering_f1", 0)
            prompt += f"\n#### Page: {page_id} (Clustering F1: {f1:.3f})\n"

        # Load fragments from the dataset cache to get the text
        fragment_texts = {}
        fragments_path = (
            os.path.join(fragments_dir, f"{page_id}.json") if fragments_dir else None
        )
        if fragments_path and os.path.exists(fragments_path):
            with open(fragments_path, "r", encoding="utf-8") as f:
                fragments = json.load(f)
                for frag in fragments:
                    fragment_texts[frag["id"]] = frag.get("text", "")
        else:
            prompt += f"(Warning: Fragment text cache missing for {page_id}. Only IDs are available.)\n"

        # List ground truth clusters
        prompt += "\n**Ground Truth Clusters:**\n"
        for i, item in enumerate(page.get("ground_truth_items", [])):
            prompt += f"- Cluster {i + 1} ({item.get('class')}): "
            for fid in item.get("fragment_ids", []):
                text = fragment_texts.get(fid, "").replace("\n", " ")
                prompt += f"[{fid}: '{text[:30]}...'] "
            prompt += "\n"

        # List predicted clusters
        prompt += "\n**Predicted Clusters:**\n"
        for i, item in enumerate(page.get("predicted_items", [])):
            prompt += f"- Cluster {i + 1} ({item.get('class')}): "
            for fid in item.get("fragment_ids", []):
                text = fragment_texts.get(fid, "").replace("\n", " ")
                prompt += f"[{fid}: '{text[:30]}...'] "
            prompt += "\n"

    if focus == "classification":
        error_diagnosis_example = (
            "e.g., misclassifying articles as ads, failing to identify titles"
        )
    elif focus == "both":
        error_diagnosis_example = "e.g., over-grouping adjacent ads, splitting long articles, misclassifying articles as ads"
    else:
        error_diagnosis_example = "e.g., over-grouping adjacent ads, splitting long articles, ignoring headers"

    prompt += f"""
### Your Task
Based on the provided system/user prompts and the specific errors in the worst performing pages, please provide:
1. **Error Diagnosis**: What pattern of errors is the model making? ({error_diagnosis_example}).
2. **Systemic Suggestions**: Concrete recommendations to improve the `System Prompt` or `User Prompt Template` to address these errors. Provide exact wording suggestions if possible.
3. **Heuristic Suggestions**: If prompt engineering isn't enough, suggest programmatic heuristics (e.g., pre-processing bounding boxes, post-processing rules) that could mitigate these errors.

Output your response in Markdown format.
"""
    return prompt


def generate_suggestions(
    experiment_id: str, eval_dir: str, client, model: str, focus: str = "clustering"
) -> int:
    """Generate and save LLM suggestions for a specific evaluation run."""
    print(f"Loading run data for {experiment_id}...", file=sys.stderr)
    run_data = load_run_data(experiment_id, eval_dir)
    if not run_data:
        return 1

    worst_pages = identify_worst_pages(run_data, focus=focus)
    if not worst_pages:
        print("No valid page evaluations found in the run data.", file=sys.stderr)
        return 1

    print(
        f"Identified {len(worst_pages)} worst performing pages. Building prompt...",
        file=sys.stderr,
    )
    fragments_dir = _derive_fragments_dir(run_data)
    if fragments_dir is None:
        print(
            "Warning: eval log has no input_folder; fragment text will be unavailable.",
            file=sys.stderr,
        )

    judge_prompt = build_judge_prompt(
        run_data, worst_pages, focus=focus, fragments_dir=fragments_dir
    )

    print(f"Sending prompt to LLM Judge ({model})...", file=sys.stderr)
    try:
        # Provide a short system prompt and pass the judge_prompt as user
        system = "You are an expert AI system evaluator and prompt engineer."
        suggestions = client.complete(system=system, user=judge_prompt)
    except Exception as e:  # noqa: BLE001
        print(f"Error calling LLM: {e}", file=sys.stderr)
        return 1

    out_dir = os.path.join("reports", "suggestions")
    prompt_path = os.path.join(out_dir, f"{experiment_id}_prompt.md")
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(judge_prompt)

    out_path = os.path.join(out_dir, f"{experiment_id}_suggestions.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Suggestions for Run `{experiment_id}`\n\n")
        f.write(suggestions)

    print(f"Successfully saved prompt to {prompt_path}", file=sys.stderr)
    print(f"Successfully saved suggestions to {out_path}", file=sys.stderr)
    return 0
