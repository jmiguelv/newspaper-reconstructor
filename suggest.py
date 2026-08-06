"""Offline analysis tool to suggest prompt and heuristic improvements using an LLM Judge."""

import json
import os
import sys

def load_run_data(run_id: str, eval_dir: str) -> dict:
    """Load the JSON evaluation log for a specific run ID."""
    log_path = os.path.join(eval_dir, f"{run_id}.json")
    if not os.path.exists(log_path):
        print(f"Error: Evaluation log not found at {log_path}", file=sys.stderr)
        return None
        
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def identify_worst_pages(run_data: dict, top_k: int = 3) -> list:
    """Identify the worst performing pages based on clustering F1 score."""
    pages = run_data.get("pages", [])
    if not pages and "items" in run_data:
         pages = [run_data]
         
    # Handle batch evaluate vs single evaluate structures
    # Batch evaluate puts pages array at root or wraps them in an array?
    # Oh wait, evaluate.py logs paged_results array.
    if isinstance(run_data, list):
         pages = run_data
    elif isinstance(run_data, dict):
         if "page_id" in run_data:
              pages = [run_data]
         elif "pages" in run_data:
              pages = run_data["pages"]
    
    # Filter out pages that failed reconstruction or don't have metrics
    valid_pages = [p for p in pages if p.get("metrics")]
    
    if not valid_pages:
        return []
        
    # Sort by clustering F1 ascending (worst first)
    valid_pages.sort(key=lambda x: x["metrics"].get("clustering_f1", 1.0))
    return valid_pages[:top_k]


def build_judge_prompt(run_data: dict, worst_pages: list) -> str:
    """Construct the prompt for the LLM Judge."""
    
    config = run_data.get("config", {}) if isinstance(run_data, dict) else {}
    system_prompt = config.get("system_prompt", "N/A")
    user_prompt = config.get("user_prompt_template", "N/A")
    
    prompt = f"""You are an expert AI system evaluator and prompt engineer.
We are using an LLM to reconstruct newspaper articles from OCR text fragments in Jawi Malay.
The fragments must be grouped into complete items (articles, advertisements, etc.).

We recently ran an evaluation and some pages performed poorly. I need you to analyze the mismatch between the predicted fragment clusters and the ground truth clusters, and provide systemic suggestions for how to improve the prompts or heuristics.

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
        f1 = page["metrics"].get("clustering_f1", 0)
        
        prompt += f"\n#### Page: {page_id} (Clustering F1: {f1:.3f})\n"
        
        # Load fragments from interim cache to get the text
        interim_path = os.path.join("data", "1_interim", "fragments", f"{page_id}.json")
        fragment_texts = {}
        if os.path.exists(interim_path):
            with open(interim_path, "r", encoding="utf-8") as f:
                fragments = json.load(f)
                for frag in fragments:
                    fragment_texts[frag["id"]] = frag.get("text", "")
        else:
            prompt += f"(Warning: Fragment text cache missing for {page_id}. Only IDs are available.)\n"

        # List ground truth clusters
        prompt += "\n**Ground Truth Clusters:**\n"
        for i, item in enumerate(page.get("ground_truth_items", [])):
            prompt += f"- Cluster {i+1} ({item.get('class')}): "
            for fid in item.get("fragment_ids", []):
                text = fragment_texts.get(fid, "").replace("\n", " ")
                prompt += f"[{fid}: '{text[:30]}...'] "
            prompt += "\n"

        # List predicted clusters
        prompt += "\n**Predicted Clusters:**\n"
        for i, item in enumerate(page.get("predicted_items", [])):
            prompt += f"- Cluster {i+1} ({item.get('class')}): "
            for fid in item.get("fragment_ids", []):
                text = fragment_texts.get(fid, "").replace("\n", " ")
                prompt += f"[{fid}: '{text[:30]}...'] "
            prompt += "\n"

    prompt += """
### Your Task
Based on the provided system/user prompts and the specific fragment clustering errors in the worst performing pages, please provide:
1. **Error Diagnosis**: What pattern of errors is the model making? (e.g., over-grouping adjacent ads, splitting long articles, ignoring headers).
2. **Systemic Suggestions**: Concrete recommendations to improve the `System Prompt` or `User Prompt Template` to address these errors. Provide exact wording suggestions if possible.
3. **Heuristic Suggestions**: If prompt engineering isn't enough, suggest programmatic heuristics (e.g., pre-processing bounding boxes, post-processing rules) that could mitigate these errors.

Output your response in Markdown format.
"""
    return prompt


def generate_suggestions(run_id: str, eval_dir: str, client, model: str) -> int:
    """Generate and save LLM suggestions for a specific evaluation run."""
    print(f"Loading run data for {run_id}...", file=sys.stderr)
    run_data = load_run_data(run_id, eval_dir)
    if not run_data:
        return 1
        
    worst_pages = identify_worst_pages(run_data)
    if not worst_pages:
        print("No valid page evaluations found in the run data.", file=sys.stderr)
        return 1
        
    print(f"Identified {len(worst_pages)} worst performing pages. Building prompt...", file=sys.stderr)
    judge_prompt = build_judge_prompt(run_data, worst_pages)
    
    print(f"Sending prompt to LLM Judge ({model})...", file=sys.stderr)
    try:
        # Provide a short system prompt and pass the judge_prompt as user
        system = "You are an expert AI system evaluator and prompt engineer."
        suggestions = client.complete(system=system, user=judge_prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        return 1
        
    out_dir = os.path.join("data", "3_reports")
    os.makedirs(out_dir, exist_ok=True)
    
    prompt_path = os.path.join(out_dir, f"{run_id}_prompt.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(judge_prompt)
        
    out_path = os.path.join(out_dir, f"{run_id}_suggestions.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Suggestions for Run `{run_id}`\n\n")
        f.write(suggestions)
        
    print(f"Successfully saved prompt to {prompt_path}", file=sys.stderr)
    print(f"Successfully saved suggestions to {out_path}", file=sys.stderr)
    return 0
