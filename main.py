"""CLI entry point for article reconstruction and evaluation.

Usage:
    # Convert ALTO to JSON (no LLM call)
    uv run python main.py --alto alto/UM-1956-01-09-6.xml --json-only

    # Reconstruct a single page
    uv run python main.py --alto alto/UM-1956-01-09-6.xml

    # Reconstruct a directory of ALTO files
    uv run python main.py --input-dir data/0_external/alto/

    # Reconstruct + evaluate a single page
    uv run python main.py --evaluate --alto alto/UM-1956-01-09-6.xml --article-xml article_xml/UM-1956-01-09-6.xml

    # Reconstruct + evaluate a directory
    uv run python main.py --evaluate --input-dir data/0_external/alto/ --ground-truth-dir data/0_external/article_xml/

Options:
    --model              Model name (or LLM_MODEL env var)
    --base-url           OpenAI-compatible API base URL (or LLM_BASE_URL env var)
    --api-key            API key (or LLM_API_KEY env var)
    --output             Save results to file instead of stdout
    --output-dir         Write one JSON file per page to this directory
    --eval-dir           Directory for evaluation logs (default: reports/evaluations/)
    --prompt-file        Read system prompt (and optionally user_prompt_template) from file
"""

import argparse
import json
import os
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.newspaper_reconstructor.evaluate import (
    evaluate_page,
    load_ground_truth_dir,
    log_evaluation_run,
    parse_article_xml,
)
from src.newspaper_reconstructor.llm import make_client
from src.newspaper_reconstructor.reconstruct import (
    load_fragments_cached,
    reconstruct_articles_cached,
)

_MD_SYSTEM_HEADING = "# System Prompt"
_MD_USER_HEADING = "# User Prompt Template"


def _sort_fragments(fragments: list[dict]) -> list[dict]:
    """Sort fragments top to bottom (vpos asc) and then right to left (hpos desc)."""
    return sorted(fragments, key=lambda f: (f.get("vpos", 0), -f.get("hpos", 0)))


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct articles from ALTO XML fragments using LLM prompting."
    )
    parser.add_argument("--alto", help="Path to a single ALTO XML file")
    parser.add_argument("--input-dir", help="Directory of ALTO or JSON fragment files")
    parser.add_argument(
        "--article-xml", help="Path to ground truth article XML (for single-page eval)"
    )
    parser.add_argument(
        "--ground-truth-dir",
        help="Directory of ground truth article XML files (for batch eval)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Convert ALTO to JSON and exit (no LLM call)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate against ground truth"
    )
    parser.add_argument(
        "--sort-fragments", action="store_true", help="Sort fragments top-to-bottom, right-to-left before reconstruction"
    )
    parser.add_argument(
        "--suggest", action="store_true", help="Generate improvement suggestions using LLM judge"
    )
    parser.add_argument(
        "--run-id", default=None, help="Specific evaluation run ID to analyze (used with --suggest)"
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible API base URL"
    )
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="LLM API request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read system prompt (and optionally user_prompt_template) from file",
    )
    parser.add_argument("--output", default=None, help="Save results to file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write one JSON file per page to this directory (for batch modes)",
    )
    parser.add_argument(
        "--eval-dir", default="reports/evaluations", help="Directory for evaluation logs"
    )
    parser.add_argument(
        "--interim-dir",
        default="data/1_interim",
        help="Directory for cached JSON fragments (default: data/1_interim)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse ALTO XML even if cached JSON exists",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample N pages from the input directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (use with --sample-size)",
    )
    args = parser.parse_args(argv)

    model = args.model or os.environ.get("LLM_MODEL")

    if not model:
        print(
            "Error: Model not set. Provide --model or set LLM_MODEL env var.",
            file=sys.stderr,
        )
        return 1

    prompt_file = args.prompt_file or "prompts/v00.md"
    prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]

    with open(prompt_file, encoding="utf-8") as f:
        content = f.read()
    if prompt_file.endswith(".json"):
        data = json.loads(content)
        system_prompt = data["system_prompt"]
        user_prompt_template = data.get("user_prompt_template", "")
    elif prompt_file.endswith(".md"):
        system_prompt, user_prompt_template = _parse_md_prompt(content)
    else:
        system_prompt = content
        user_prompt_template = ""

    if args.sort_fragments:
        prompt_name = f"{prompt_name}_sorted"

    # --json-only: convert and exit
    if args.json_only:
        if args.alto:
            result = load_fragments_cached(args.alto, args.interim_dir, args.force)
            _output(result, args.output)
            return 0
        if args.input_dir:
            result = load_fragments_cached(args.input_dir, args.interim_dir, args.force)
            if args.output_dir and isinstance(result, dict):
                _output_dir(result, args.output_dir)
                return 0
            _output(result, args.output)
            return 0
        print("Error: --json-only requires --alto or --input-dir", file=sys.stderr)
        return 1

    if args.suggest:
        if not args.run_id:
            print("Error: --suggest requires --run-id to specify which evaluation log to analyze.", file=sys.stderr)
            return 1
            
        # We import here to avoid circular dependencies or loading LLM judge if not needed
        from src.newspaper_reconstructor.suggest import generate_suggestions
        client = make_client(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model,
            timeout=args.timeout,
        )
        return generate_suggestions(args.run_id, args.eval_dir, client, model)

    # Determine mode: single page or directory
    if args.input_dir:
        return _run_input_dir(
            args, model, system_prompt, user_prompt_template, prompt_name
        )
    elif args.alto:
        return _run_single_page(
            args, model, system_prompt, user_prompt_template, prompt_name
        )

    parser.print_help()
    return 1


def _run_single_page(
    args, model: str, system_prompt: str, user_prompt_template: str, prompt_name: str
) -> int:
    fragments = load_fragments_cached(args.alto, args.interim_dir, args.force)
    if args.sort_fragments:
        fragments = _sort_fragments(fragments)
    page_id = os.path.splitext(os.path.basename(args.alto))[0]

    if args.evaluate and args.article_xml:
        return _reconstruct_and_eval_single(
            args,
            model,
            fragments,
            page_id,
            system_prompt,
            user_prompt_template,
            prompt_name,
        )

    client = make_client(
        base_url=args.base_url,
        api_key=args.api_key,
        model=model,
        timeout=args.timeout,
    )
    result = reconstruct_articles_cached(
        fragments,
        client,
        system_prompt,
        user_prompt_template,
        page_id,
        args.interim_dir,
        prompt_name,
        model,
        args.force,
    )
    if result is None:
        print(f"[{page_id}] Reconstruction failed", file=sys.stderr)
        return 1
    _output({"page_id": page_id, "items": result}, args.output)
    return 0


def _run_input_dir(
    args, model: str, system_prompt: str, user_prompt_template: str, prompt_name: str
) -> int:
    pages = load_fragments_cached(args.input_dir, args.interim_dir, args.force)
    if not isinstance(pages, dict):
        print("Error: --input-dir expects a directory", file=sys.stderr)
        return 1

    all_page_ids = sorted(pages.keys())
    if args.sample_size is not None and args.sample_size < len(all_page_ids):
        rng = random.Random(args.seed)
        all_page_ids = rng.sample(all_page_ids, args.sample_size)
        print(f"Sampled {len(all_page_ids)} of {len(pages)} pages", file=sys.stderr)

    client = make_client(
        base_url=args.base_url,
        api_key=args.api_key,
        model=model,
        timeout=args.timeout,
    )

    ground_truth = {}
    if args.evaluate and args.ground_truth_dir:
        ground_truth = load_ground_truth_dir(args.ground_truth_dir)

    eval_results = []
    failed_pages = []

    start_time = time.time()

    for page_id in all_page_ids:
        print(f"[{page_id}] Reconstructing...", file=sys.stderr)
        fragments = pages[page_id]
        if args.sort_fragments:
            fragments = _sort_fragments(fragments)
        predicted = reconstruct_articles_cached(
            fragments,
            client,
            system_prompt,
            user_prompt_template,
            page_id,
            args.interim_dir,
            prompt_name,
            model,
            args.force,
        )

        if predicted is None:
            failed_pages.append(page_id)
            print(f"[{page_id}] SKIPPED (reconstruction failed)", file=sys.stderr)
            continue

        if page_id in ground_truth:
            print(f"[{page_id}] Evaluating...", file=sys.stderr)
            metrics = evaluate_page(predicted, ground_truth[page_id])
            print(
                f"[{page_id}] F1={metrics['clustering_f1']:.3f}  "
                f"BCF1={metrics['bcubed_f1']:.3f}  "
                f"coverage={metrics['coverage']:.3f}  "
                f"class_acc={metrics['class_accuracy']}",
                file=sys.stderr,
            )
        else:
            metrics = None
            if args.evaluate:
                print(
                    f"[{page_id}] No ground truth found, skipping evaluation",
                    file=sys.stderr,
                )

        eval_results.append(
            {
                "page_id": page_id,
                "metrics": metrics,
                "predicted_items": predicted,
                "ground_truth_items": ground_truth.get(page_id),
            }
        )

    if failed_pages:
        print(
            f"\nFailed pages ({len(failed_pages)}): {', '.join(failed_pages)}",
            file=sys.stderr,
        )

    if args.evaluate and args.ground_truth_dir:
        paged_results = [r for r in eval_results if r["metrics"] is not None]
        config = {
            "provider": "openai",
            "model": model,
            "base_url": args.base_url or os.environ.get("LLM_BASE_URL"),
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "prompt_name": prompt_name,
            "sample_size": args.sample_size,
            "pages_processed": len(eval_results),
            "pages_failed": len(failed_pages),
            "seed": args.seed,
            "execution_time_seconds": time.time() - start_time,
        }
        log_path = log_evaluation_run(paged_results, config, args.eval_dir)
        print(f"\nEvaluation log saved to: {log_path}", file=sys.stderr)

        if paged_results:
            f1s = [r["metrics"]["clustering_f1"] for r in paged_results]
            bcubed_f1s = [r["metrics"]["bcubed_f1"] for r in paged_results]
            coverages = [r["metrics"]["coverage"] for r in paged_results]
            class_accs = [
                r["metrics"]["class_accuracy"]
                for r in paged_results
                if r["metrics"]["class_accuracy"] is not None
            ]
            requested = args.sample_size or len(all_page_ids)
            print(
                f"\n=== Summary ({len(paged_results)}/{requested} pages, {len(failed_pages)} skipped) ===",
                file=sys.stderr,
            )
            print(
                f"  Mean clustering F1:    {sum(f1s) / len(f1s):.4f}", file=sys.stderr
            )
            print(
                f"  Mean B-cubed F1:       {sum(bcubed_f1s) / len(bcubed_f1s):.4f}",
                file=sys.stderr,
            )
            print(
                f"  Mean coverage:         {sum(coverages) / len(coverages):.4f}",
                file=sys.stderr,
            )
            if class_accs:
                print(
                    f"  Mean class accuracy:   {sum(class_accs) / len(class_accs):.4f}",
                    file=sys.stderr,
                )

    all_results = {r["page_id"]: r["predicted_items"] for r in eval_results}
    if args.output_dir:
        _output_dir(all_results, args.output_dir)
    else:
        _output(eval_results if args.evaluate else all_results, args.output)
    return 0


def _reconstruct_and_eval_single(
    args, model, fragments, page_id, system_prompt, user_prompt_template, prompt_name
):
    client = make_client(
        base_url=args.base_url,
        api_key=args.api_key,
        model=model,
        timeout=args.timeout,
    )
    start_time = time.time()
    predicted = reconstruct_articles_cached(
        fragments,
        client,
        system_prompt,
        user_prompt_template,
        page_id,
        args.interim_dir,
        prompt_name,
        model,
        args.force,
    )
    if predicted is None:
        print(
            f"[{page_id}] Reconstruction failed, skipping evaluation", file=sys.stderr
        )
        return 1

    truth = parse_article_xml(args.article_xml)
    metrics = evaluate_page(predicted, truth)

    print(f"\n=== {page_id} ===", file=sys.stderr)
    print(f"  Clustering F1:  {metrics['clustering_f1']:.4f}", file=sys.stderr)
    print(f"  B-cubed F1:      {metrics['bcubed_f1']:.4f}", file=sys.stderr)
    print(f"  Precision:      {metrics['clustering_precision']:.4f}", file=sys.stderr)
    print(f"  Recall:         {metrics['clustering_recall']:.4f}", file=sys.stderr)
    print(f"  Class accuracy: {metrics['class_accuracy']}", file=sys.stderr)
    print(f"  Coverage:       {metrics['coverage']:.4f}", file=sys.stderr)
    print(
        f"  Items:          {metrics['num_predicted_items']} pred / {metrics['num_ground_truth_items']} truth",
        file=sys.stderr,
    )

    config = {
        "provider": "openai",
        "model": model,
        "base_url": args.base_url or os.environ.get("LLM_BASE_URL"),
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "prompt_name": prompt_name,
        "sample_size": None,
        "seed": args.seed,
        "execution_time_seconds": time.time() - start_time,
    }
    log_path = log_evaluation_run(
        [
            {
                "page_id": page_id,
                "metrics": metrics,
                "predicted_items": predicted,
                "ground_truth_items": truth,
            }
        ],
        config,
        args.eval_dir,
    )
    print(f"\nEvaluation log: {log_path}", file=sys.stderr)

    _output({"page_id": page_id, "items": predicted, "metrics": metrics}, args.output)
    return 0


def _output(data, output_path):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}", file=sys.stderr)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def _output_dir(pages: dict, output_dir: str):
    """Write one JSON file per page to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for page_id, data in pages.items():
        path = os.path.join(output_dir, f"{page_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(pages)} files to {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
