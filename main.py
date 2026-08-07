import json
import os
import random
import sys
import time

import typer
from dotenv import load_dotenv

load_dotenv()

from src.newspaper_reconstructor.evaluate import (
    evaluate_page,
    load_ground_truth_dir,
    log_evaluation_run,
)
from src.newspaper_reconstructor.llm import make_client
from src.newspaper_reconstructor.reconstruct import (
    alto_to_json,
    classify_fragments,
    reconstruct_articles,
)
from src.newspaper_reconstructor.sort import sort_fragments

_MD_SYSTEM_HEADING = "# System Prompt"
_MD_USER_HEADING = "# User Prompt Template"

app = typer.Typer(
    help="Reconstruct articles from ALTO XML fragments using LLM pipelines.",
    no_args_is_help=True,
)


def _parse_md_prompt(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        if line.strip() in (_MD_SYSTEM_HEADING, _MD_USER_HEADING):
            current = line.strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    system_prompt = "\n".join(sections.get(_MD_SYSTEM_HEADING, [])).strip()
    user_prompt_template = "\n".join(sections.get(_MD_USER_HEADING, [])).strip()
    return system_prompt, user_prompt_template


def _load_prompt(prompt_file: str) -> tuple[str, str]:
    with open(prompt_file, encoding="utf-8") as f:
        content = f.read()
    if prompt_file.endswith(".json"):
        data = json.loads(content)
        return data["system_prompt"], data.get("user_prompt_template", "")
    elif prompt_file.endswith(".md"):
        return _parse_md_prompt(content)
    else:
        return content, ""


@app.command()
def parse(
    input_folder: str = typer.Option(
        ..., "--input-folder", "-i", help="Directory of ALTO XML files"
    ),
    output_folder: str = typer.Option(
        ..., "--output-folder", "-o", help="Directory to save JSON fragments"
    ),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
):
    """Parse ALTO XML files into JSON fragment lists."""
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    files = sorted(os.listdir(input_folder))
    if page_id:
        files = [f for f in files if f.startswith(page_id)]
    for fname in files:
        if not fname.endswith(".xml"):
            continue
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.json")

        fragments = alto_to_json(in_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fragments, f, indent=2, ensure_ascii=False)
        count += 1
    typer.echo(f"Parsed {count} files to {output_folder}")


@app.command()
def classify(
    input_folder: str = typer.Option(
        ..., "--input-folder", "-i", help="Directory of JSON fragments"
    ),
    prompt_file: str = typer.Option(
        ..., "--prompt-file", "-p", help="Classification prompt file (.md or .json)"
    ),
    output_folder: str = typer.Option(
        ..., "--output-folder", "-o", help="Directory to save classified JSON fragments"
    ),
    model: str = typer.Option(..., envvar="LLM_MODEL", help="LLM model name"),
    base_url: str | None = typer.Option(
        None, envvar="LLM_BASE_URL", help="API base URL"
    ),
    api_key: str | None = typer.Option(None, envvar="LLM_API_KEY", help="API key"),
    timeout: float = typer.Option(300.0, help="API timeout in seconds"),
    sample_size: int | None = typer.Option(None, help="Randomly sample N pages"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
):
    """Classify fragments using an LLM. Output is fragments enriched with 'predicted_class'."""
    os.makedirs(output_folder, exist_ok=True)
    client = make_client(model, base_url, api_key, timeout)
    sys_prompt, user_prompt = _load_prompt(prompt_file)

    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".json")]
    if page_id:
        files = [f for f in files if f.startswith(page_id)]
    if sample_size and sample_size < len(files):
        random.seed(seed)
        files = random.sample(files, sample_size)

    success = 0
    for fname in files:
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        with open(in_path, encoding="utf-8") as f:
            fragments = json.load(f)

        typer.echo(f"Classifying {fname}...")
        classes = classify_fragments(fragments, client, sys_prompt, user_prompt)

        if classes:
            # Enrich fragments
            for frag in fragments:
                if frag["id"] in classes:
                    frag["predicted_class"] = classes[frag["id"]]

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(fragments, f, indent=2, ensure_ascii=False)
            success += 1
        else:
            typer.echo(f"Failed to classify {fname}", err=True)

    typer.echo(f"Classified {success}/{len(files)} files to {output_folder}")


@app.command()
def cluster(
    input_folder: str = typer.Option(
        ...,
        "--input-folder",
        "-i",
        help="Directory of JSON fragments (optionally classified)",
    ),
    prompt_file: str = typer.Option(
        ..., "--prompt-file", "-p", help="Clustering prompt file (.md or .json)"
    ),
    output_folder: str = typer.Option(
        ..., "--output-folder", "-o", help="Directory to save predicted articles"
    ),
    model: str = typer.Option(..., envvar="LLM_MODEL", help="LLM model name"),
    base_url: str | None = typer.Option(
        None, envvar="LLM_BASE_URL", help="API base URL"
    ),
    api_key: str | None = typer.Option(None, envvar="LLM_API_KEY", help="API key"),
    timeout: float = typer.Option(300.0, help="API timeout in seconds"),
    sort_fragments_flag: bool = typer.Option(
        False, "--sort-fragments", help="Sort fragments spatially before clustering"
    ),
    sample_size: int | None = typer.Option(None, help="Randomly sample N pages"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
):
    """Cluster fragments into articles using an LLM."""
    os.makedirs(output_folder, exist_ok=True)
    client = make_client(model, base_url, api_key, timeout)
    sys_prompt, user_prompt = _load_prompt(prompt_file)

    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".json")]
    if page_id:
        files = [f for f in files if f.startswith(page_id)]
    if sample_size and sample_size < len(files):
        random.seed(seed)
        files = random.sample(files, sample_size)

    success = 0
    for fname in files:
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        with open(in_path, encoding="utf-8") as f:
            fragments = json.load(f)

        if sort_fragments_flag:
            fragments = sort_fragments(fragments)

        typer.echo(f"Clustering {fname}...")
        articles = reconstruct_articles(fragments, client, sys_prompt, user_prompt)

        if articles is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            success += 1
        else:
            typer.echo(f"Failed to cluster {fname}", err=True)

    typer.echo(f"Clustered {success}/{len(files)} files to {output_folder}")


@app.command()
def evaluate(
    input_folder: str = typer.Option(
        ..., "--input-folder", "-i", help="Directory of predicted articles JSON"
    ),
    ground_truth_folder: str = typer.Option(
        ..., "--ground-truth-folder", "-g", help="Directory of ground truth XML"
    ),
    eval_dir: str = typer.Option(
        "reports/evaluations", help="Directory for evaluation logs"
    ),
    run_id: str = typer.Option(
        None, help="Identifier for this evaluation run (e.g., model_v1_sample16)"
    ),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
):
    """Evaluate predicted articles against ground truth XML."""
    os.makedirs(eval_dir, exist_ok=True)

    if run_id is None:
        run_id = f"eval_{int(time.time())}"

    gt_data = load_ground_truth_dir(ground_truth_folder)

    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".json")]
    if page_id:
        files = [f for f in files if f.startswith(page_id)]

    results = {}
    for fname in files:
        page_id = os.path.splitext(fname)[0]
        if page_id not in gt_data:
            typer.echo(f"Warning: No ground truth for {page_id}", err=True)
            continue

        with open(os.path.join(input_folder, fname), encoding="utf-8") as f:
            predicted_items = json.load(f)

        page_metrics = evaluate_page(predicted_items, gt_data[page_id])
        results[page_id] = page_metrics

    if not results:
        typer.echo("No matching pages found for evaluation.", err=True)
        raise typer.Exit(1)

    # Mock config for log since we decoupled it
    config = {
        "run_id": run_id,
        "input_folder": input_folder,
        "ground_truth_folder": ground_truth_folder,
    }
    log_path = log_evaluation_run(results, config, eval_dir, run_id)
    typer.echo(f"Evaluation complete. Logs saved to {log_path}")


@app.command()
def suggest(
    run_id: str = typer.Option(..., help="Run ID of the evaluation to analyze"),
    model: str = typer.Option(..., envvar="LLM_MODEL", help="LLM model name"),
    base_url: str | None = typer.Option(
        None, envvar="LLM_BASE_URL", help="API base URL"
    ),
    api_key: str | None = typer.Option(None, envvar="LLM_API_KEY", help="API key"),
):
    """Analyze evaluation logs and suggest improvements using LLM judge."""
    # We will invoke the suggest logic here. Let's just run it as a subprocess for now
    # to avoid modifying suggest.py directly, or we can import it.
    from src.newspaper_reconstructor.suggest import main as suggest_main

    # suggest.py uses sys.argv, so we override it temporarily
    old_argv = sys.argv
    sys.argv = ["suggest.py", run_id, "--model", model]
    if base_url:
        sys.argv.extend(["--base-url", base_url])
    if api_key:
        sys.argv.extend(["--api-key", api_key])

    try:
        suggest_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    app()
