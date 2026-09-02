import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

from src.newspaper_reconstructor.evaluate import (
    evaluate_classification_page,
    evaluate_reconstruction_page,
    load_ground_truth_dir,
    log_evaluation_experiment,
)
from src.newspaper_reconstructor.ingest import load_article_json
from src.newspaper_reconstructor.llm import LLMClient, make_client
from src.newspaper_reconstructor.reconstruct import (
    LLM_AND_IO_ERRORS,
    alto_to_json,
    classify_fragments,
    reconstruct_articles,
)
from src.newspaper_reconstructor.suggest import generate_suggestions

_MD_SYSTEM_HEADING = "# System Prompt"
_MD_USER_HEADING = "# User Prompt Template"

app = typer.Typer(
    help="Reconstruct articles from newspaper text fragments using LLM pipelines.",
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


def _build_model_kwargs(
    model_kwargs: str | None,
    max_tokens: int | None,
    frequency_penalty: float | None,
) -> dict:
    parsed: dict = {}
    if model_kwargs:
        try:
            parsed = json.loads(model_kwargs)
        except json.JSONDecodeError as e:
            typer.echo(f"Error parsing --model-kwargs as JSON: {e}", err=True)
            raise typer.Exit(1)
    if max_tokens is not None:
        parsed["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        parsed["frequency_penalty"] = frequency_penalty
    return parsed


@dataclass
class StageContext:
    """Shared context passed to a per-file stage processor."""

    client: LLMClient
    sys_prompt: str
    user_prompt: str
    input_folder: str
    output_folder: str
    save_prompts: bool


def _run_batch(
    files: list[str],
    process_fn: "Callable[[str, threading.Lock, StageContext], bool]",
    max_workers: int,
    ctx: StageContext,
) -> int:
    lock = threading.Lock()
    success = 0

    def _wrapped(fname: str) -> bool:
        return process_fn(fname, lock, ctx)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_wrapped, fname): fname for fname in files}
        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
            except LLM_AND_IO_ERRORS as e:
                fname = futures[future]
                typer.echo(f"Error processing {fname}: {e}", err=True)
    return success


def _iter_input_files(input_folder: str, page_id: str | None, ext: str) -> list[str]:
    """List files in input_folder matching ext, optionally filtered by page_id."""
    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(ext)]
    if page_id:
        files = [f for f in files if f.startswith(page_id)]
    return files


def _list_stage_files(
    input_folder: str,
    page_id: str | None,
    sample_size: int | None,
    seed: int,
) -> list[str]:
    """List JSON fragments for an LLM stage, optionally sampled."""
    files = [
        f
        for f in _iter_input_files(input_folder, page_id, ".json")
        if not f.startswith("_")
    ]
    if sample_size and sample_size < len(files):
        random.seed(seed)
        files = random.sample(files, sample_size)
    return files


def _save_stage_metadata(
    output_folder: str,
    model: str,
    prompt_file: str,
    sample_size: int | None,
    provider: str | None,
    tag: str | None,
    parsed_model_kwargs: dict,
    execution_time_seconds: float,
) -> None:
    prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
    metadata_path = os.path.join(output_folder, "_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "execution_time_seconds": execution_time_seconds,
                "model": model,
                "prompt_name": prompt_name,
                "sample_size": sample_size,
                "provider": provider,
                "tag": tag,
                "model_kwargs": parsed_model_kwargs,
            },
            f,
            indent=2,
        )


def _prompt_out_path(output_folder: str, fname: str) -> str:
    return os.path.join(
        output_folder, "prompts", f"{os.path.splitext(fname)[0]}.prompt.txt"
    )


def _run_llm_stage(
    *,
    input_folder: str,
    output_folder: str,
    prompt_file: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    provider: str | None,
    timeout: float,
    sample_size: int | None,
    seed: int,
    page_id: str | None,
    save_prompts: bool,
    tag: str | None,
    model_kwargs: str | None,
    max_workers: int,
    max_tokens: int | None,
    frequency_penalty: float | None,
    process_fn: "Callable[[str, threading.Lock, StageContext], bool]",
    action_past: str,
) -> None:
    """Shared runner for classify/cluster: client, batching, and metadata."""
    os.makedirs(output_folder, exist_ok=True)

    parsed_model_kwargs = _build_model_kwargs(
        model_kwargs, max_tokens, frequency_penalty
    )
    try:
        client = make_client(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            provider=provider,
            model_kwargs=parsed_model_kwargs,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    sys_prompt, user_prompt = _load_prompt(prompt_file)
    files = _list_stage_files(input_folder, page_id, sample_size, seed)

    ctx = StageContext(
        client=client,
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        input_folder=input_folder,
        output_folder=output_folder,
        save_prompts=save_prompts,
    )

    start_time = time.time()
    success = _run_batch(files, process_fn, max_workers, ctx)
    execution_time_seconds = time.time() - start_time

    _save_stage_metadata(
        output_folder,
        model,
        prompt_file,
        sample_size,
        provider,
        tag,
        parsed_model_kwargs,
        execution_time_seconds,
    )

    typer.echo(
        f"{action_past} {success}/{len(files)} files to {output_folder} in {execution_time_seconds:.1f}s"
    )


def _classify_process(fname: str, lock: threading.Lock, ctx: StageContext) -> bool:
    in_path = os.path.join(ctx.input_folder, fname)
    out_path = os.path.join(ctx.output_folder, fname)

    if os.path.exists(out_path):
        with lock:
            typer.echo(f"Skipping {fname}, already classified.")
        return True

    with open(in_path, encoding="utf-8") as f:
        fragments = json.load(f)

    prompt_out_path = (
        _prompt_out_path(ctx.output_folder, fname) if ctx.save_prompts else None
    )

    with lock:
        typer.echo(f"Classifying {fname}...")

    classes = classify_fragments(
        fragments,
        ctx.client,
        ctx.sys_prompt,
        ctx.user_prompt,
        prompt_out_path=prompt_out_path,
    )

    if classes:
        for frag in fragments:
            if frag["id"] in classes:
                frag["predicted_class"] = classes[frag["id"]]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fragments, f, indent=2, ensure_ascii=False)
        return True

    with lock:
        typer.echo(f"Failed to classify {fname}", err=True)
    return False


def _cluster_process(fname: str, lock: threading.Lock, ctx: StageContext) -> bool:
    in_path = os.path.join(ctx.input_folder, fname)
    out_path = os.path.join(ctx.output_folder, fname)

    if os.path.exists(out_path):
        with lock:
            typer.echo(f"Skipping {fname}, already clustered.")
        return True

    with open(in_path, encoding="utf-8") as f:
        fragments = json.load(f)

    prompt_out_path = (
        _prompt_out_path(ctx.output_folder, fname) if ctx.save_prompts else None
    )

    with lock:
        typer.echo(f"Clustering {fname}...")

    articles = reconstruct_articles(
        fragments,
        ctx.client,
        ctx.sys_prompt,
        ctx.user_prompt,
        prompt_out_path=prompt_out_path,
    )

    if articles is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        return True

    with lock:
        typer.echo(f"Failed to cluster {fname}", err=True)
    return False


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
    for fname in _iter_input_files(input_folder, page_id, ".xml"):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.json")

        fragments = alto_to_json(in_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fragments, f, indent=2, ensure_ascii=False)
        count += 1
    typer.echo(f"Parsed {count} files to {output_folder}")


@app.command()
def etl(
    input_folder: str = typer.Option(
        ..., "--input-folder", "-i", help="Directory of article JSON files"
    ),
    output_folder: str = typer.Option(
        ..., "--output-folder", "-o", help="Directory to save fragment lists"
    ),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
):
    """Convert article JSON files ({id: text}) into fragment lists."""
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for fname in _iter_input_files(input_folder, page_id, ".json"):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        fragments = load_article_json(in_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fragments, f, indent=2, ensure_ascii=False)
        count += 1
    typer.echo(f"Converted {count} files to {output_folder}")


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
    provider: str | None = typer.Option(
        None, envvar="LLM_PROVIDER", help="Provider name (e.g. create, openrouter)"
    ),
    timeout: float = typer.Option(300.0, help="API timeout in seconds"),
    sample_size: int | None = typer.Option(None, help="Randomly sample N pages"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
    save_prompts: bool = typer.Option(
        False, "--save-prompts", help="Save individual prompts sent to the LLM"
    ),
    tag: str | None = typer.Option(
        None, "--tag", help="Optional tag for this run (e.g., think_high)"
    ),
    model_kwargs: str | None = typer.Option(
        None, "--model-kwargs", help="JSON string for extra model arguments"
    ),
    max_workers: int = typer.Option(
        1, "--max-workers", "-w", help="Max concurrent workers for batch processing"
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate (prevents infinite loops)",
    ),
    frequency_penalty: float | None = typer.Option(
        None, "--frequency-penalty", help="Frequency penalty to reduce repetition"
    ),
):
    """Classify fragments using an LLM. Output is fragments enriched with 'predicted_class'."""
    _run_llm_stage(
        input_folder=input_folder,
        output_folder=output_folder,
        prompt_file=prompt_file,
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        timeout=timeout,
        sample_size=sample_size,
        seed=seed,
        page_id=page_id,
        save_prompts=save_prompts,
        tag=tag,
        model_kwargs=model_kwargs,
        max_workers=max_workers,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        process_fn=_classify_process,
        action_past="Classified",
    )


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
    provider: str | None = typer.Option(
        None, envvar="LLM_PROVIDER", help="Provider name (e.g. create, openrouter)"
    ),
    timeout: float = typer.Option(300.0, help="API timeout in seconds"),
    sample_size: int | None = typer.Option(None, help="Randomly sample N pages"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
    save_prompts: bool = typer.Option(
        False, "--save-prompts", help="Save individual prompts sent to the LLM"
    ),
    tag: str | None = typer.Option(
        None, "--tag", help="Optional tag for this run (e.g., think_high)"
    ),
    model_kwargs: str | None = typer.Option(
        None, "--model-kwargs", help="JSON string for extra model arguments"
    ),
    max_workers: int = typer.Option(
        1, "--max-workers", "-w", help="Max concurrent workers for batch processing"
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate (prevents infinite loops)",
    ),
    frequency_penalty: float | None = typer.Option(
        None, "--frequency-penalty", help="Frequency penalty to reduce repetition"
    ),
):
    """Cluster fragments into articles using an LLM."""
    _run_llm_stage(
        input_folder=input_folder,
        output_folder=output_folder,
        prompt_file=prompt_file,
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        timeout=timeout,
        sample_size=sample_size,
        seed=seed,
        page_id=page_id,
        save_prompts=save_prompts,
        tag=tag,
        model_kwargs=model_kwargs,
        max_workers=max_workers,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        process_fn=_cluster_process,
        action_past="Clustered",
    )


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
    experiment_id: str = typer.Option(
        None, help="Identifier for this evaluation run (e.g., model_v1_sample16)"
    ),
    page_id: str | None = typer.Option(None, help="Process a single page ID"),
    task: str = typer.Option(
        ..., help="Task to evaluate ('classification' or 'reconstruction')"
    ),
):
    """Evaluate predicted results against ground truth XML."""
    if task not in ("classification", "reconstruction"):
        typer.echo(
            "Error: --task must be 'classification' or 'reconstruction'", err=True
        )
        raise typer.Exit(1)

    os.makedirs(eval_dir, exist_ok=True)

    if experiment_id is None:
        experiment_id = f"eval_{int(time.time())}"

    gt_data = load_ground_truth_dir(ground_truth_folder)

    files = [
        f
        for f in sorted(os.listdir(input_folder))
        if f.endswith(".json") and not f.startswith("_")
    ]
    if page_id:
        files = [f for f in files if f.startswith(page_id)]

    results = []
    for fname in files:
        page_id_match = os.path.splitext(fname)[0]
        if page_id_match not in gt_data:
            typer.echo(f"Warning: No ground truth for {page_id_match}", err=True)
            continue

        with open(os.path.join(input_folder, fname), encoding="utf-8") as f:
            predicted_items = json.load(f)

        if task == "classification":
            page_metrics = evaluate_classification_page(
                predicted_items, gt_data[page_id_match]
            )
        else:
            page_metrics = evaluate_reconstruction_page(
                predicted_items, gt_data[page_id_match]
            )

        results.append(
            {
                "page_id": page_id_match,
                "metrics": page_metrics,
                "predicted_items": predicted_items,
                "ground_truth_items": gt_data[page_id_match],
            }
        )

    if not results:
        typer.echo("No matching pages found for evaluation.", err=True)
        raise typer.Exit(1)

    # Mock config for log since we decoupled it
    config = {
        "experiment_id": experiment_id,
        "task": task,
        "input_folder": input_folder,
        "ground_truth_folder": ground_truth_folder,
    }

    metadata_path = os.path.join(input_folder, "_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
                config.update(meta)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            typer.echo(f"Warning: Failed to read metadata.json: {e}", err=True)

    log_path = log_evaluation_experiment(results, config, eval_dir, experiment_id)
    typer.echo(f"Evaluation complete. Logs saved to {log_path}")


@app.command()
def suggest(
    experiment_id: str = typer.Option(
        ..., help="Experiment ID of the evaluation to analyze"
    ),
    model: str = typer.Option(..., envvar="LLM_MODEL", help="LLM model name"),
    base_url: str | None = typer.Option(
        None, envvar="LLM_BASE_URL", help="API base URL"
    ),
    api_key: str | None = typer.Option(None, envvar="LLM_API_KEY", help="API key"),
    provider: str | None = typer.Option(
        None, envvar="LLM_PROVIDER", help="Provider name (e.g. create, openrouter)"
    ),
    timeout: float = typer.Option(300.0, help="API timeout in seconds"),
    focus: str = typer.Option(
        "clustering", help="Focus of the analysis: clustering, classification, or both"
    ),
    tag: str | None = typer.Option(
        None, "--tag", help="Optional tag for this run (e.g., think_high)"
    ),
    model_kwargs: str | None = typer.Option(
        None, "--model-kwargs", help="JSON string for extra model arguments"
    ),
):
    """Analyze evaluation logs and suggest improvements using LLM judge."""
    parsed_model_kwargs = None
    if model_kwargs:
        try:
            parsed_model_kwargs = json.loads(model_kwargs)
        except json.JSONDecodeError as e:
            typer.echo(f"Error parsing --model-kwargs as JSON: {e}", err=True)
            raise typer.Exit(1)

    eval_dir = os.path.join("reports", "evaluations")
    try:
        client = make_client(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            provider=provider,
            model_kwargs=parsed_model_kwargs,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    sys.exit(generate_suggestions(experiment_id, eval_dir, client, model, focus))


@app.command()
def plan(
    input_folder: str = typer.Option(
        ..., "-i", "--input", help="Folder containing input fragment JSONs"
    ),
):
    """Estimate hardware requirements and token usage based on raw input fragments."""
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        typer.echo(
            f"Error: Input folder {input_folder} does not exist or is not a directory.",
            err=True,
        )
        raise typer.Exit(1)

    total_pages = 0
    total_fragments = 0
    total_chars = 0

    for file_path in input_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_pages += 1
                    total_fragments += len(data)
                    for item in data:
                        total_chars += len(item.get("text", ""))
        except (OSError, json.JSONDecodeError) as e:
            typer.echo(f"Warning: Failed to process {file_path.name}: {e}", err=True)

    if total_pages == 0:
        typer.echo("No valid fragment JSONs found in the input folder.")
        raise typer.Exit(1)

    avg_fragments = total_fragments / total_pages
    avg_chars = total_chars / total_pages
    avg_tokens = avg_chars / 2.5

    typer.echo(f"Analyzed {total_pages} pages in '{input_folder}'.")
    typer.echo(f"Average fragments per page: {avg_fragments:.1f}")
    typer.echo(f"Average characters per page: {avg_chars:.1f}")
    typer.echo(
        f"Estimated input tokens per page: {avg_tokens:,.0f} (assuming ~2.5 chars/token)"
    )
    typer.echo("")
    typer.echo(
        "To estimate precise VRAM requirements for a 16K output budget, visit: https://vramcalculator.com/"
    )


if __name__ == "__main__":
    app()
