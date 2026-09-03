"""Loading of LLM prompt files (.md, .json, or plain text)."""

import json

_MD_SYSTEM_HEADING = "# System Prompt"
_MD_USER_HEADING = "# User Prompt Template"


def parse_md_prompt(content: str) -> tuple[str, str]:
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
    user_prompt = "\n".join(sections.get(_MD_USER_HEADING, [])).strip()
    return system_prompt, user_prompt


def load_prompt(prompt_file: str) -> tuple[str, str]:
    with open(prompt_file, encoding="utf-8") as f:
        content = f.read()
    if prompt_file.endswith(".json"):
        data = json.loads(content)
        return data["system_prompt"], data.get("user_prompt_template", "")
    elif prompt_file.endswith(".md"):
        return parse_md_prompt(content)
    else:
        return content, ""
