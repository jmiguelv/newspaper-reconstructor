"""Look up the string representation of a token ID for a given model.

Usage:
    uv run python scripts/token_lookup.py <model_name> <token_id>

Example:
    uv run python scripts/token_lookup.py aisingapore/Qwen-SEA-LION-v4.5-27B-IT 248044
"""

import sys

from transformers import AutoTokenizer


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_name> <token_id>")
        sys.exit(1)

    model_name = sys.argv[1]
    token_id = int(sys.argv[2])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    token = tokenizer.decode([token_id])
    print(f"Token {token_id}: {token}")
    print(f"Characters: {' '.join(token)}")
    print(f"Unicode: {' '.join(f'U+{ord(c):04X} ({c})' for c in token)}")


if __name__ == "__main__":
    main()
