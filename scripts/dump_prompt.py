"""Render the exact prompt sent to the LLM for one page, for offline debugging.

Usage:
    uv run python scripts/dump_prompt.py --dataset 1956_ocr_out_20260907 \
        --page-id UM-1956-02-11-6 --prompt prompts/v01.01.02.md --payload
"""

import argparse
import json
import os

from newspaper_reconstructor.prompts import load_prompt


def render_prompts(fragments_path: str, prompt_file: str) -> tuple[str, str]:
    with open(fragments_path, encoding="utf-8") as f:
        fragments = json.load(f)
    sys_prompt, user_template = load_prompt(prompt_file)
    frag_text = json.dumps(fragments, ensure_ascii=False, indent=2)
    return sys_prompt, user_template.format(fragments=frag_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--prompt", default="prompts/v01.01.02.md")
    parser.add_argument("--fragments-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--payload",
        nargs="?",
        const="auto",
        default=None,
        help="Also write an OpenAI-compatible request body (default: <out>.payload.json)",
    )
    parser.add_argument("--model", default="aisingapore/Qwen-SEA-LION-v4.5-27B-IT")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    fragments_dir = args.fragments_dir or f"data/1_interim/{args.dataset}/fragments"
    fragments_path = os.path.join(fragments_dir, f"{args.page_id}.json")
    out_path = args.out or os.path.join(
        "reports", "debug", f"{args.page_id}.prompt.txt"
    )

    sys_prompt, user_prompt = render_prompts(fragments_path, args.prompt)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# System Prompt\n{sys_prompt}\n\n# User Prompt\n{user_prompt}")
    print(f"Wrote prompt ({len(sys_prompt) + len(user_prompt)} chars) to {out_path}")

    if args.payload is not None:
        payload_path = (
            args.payload if args.payload != "auto" else f"{out_path}.payload.json"
        )
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote request payload to {payload_path}")


if __name__ == "__main__":
    main()
