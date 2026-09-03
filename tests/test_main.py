"""Unit tests for main.py helpers."""

from src.newspaper_reconstructor.prompts import parse_md_prompt as _parse_md_prompt


class TestParseMdPrompt:
    def test_both_sections(self):
        content = """# System Prompt

You are given text fragments.

# User Prompt Template

Fragments:

{fragments}
"""
        system, user = _parse_md_prompt(content)
        assert system == "You are given text fragments."
        assert "Fragments:" in user
        assert "{fragments}" in user

    def test_missing_user_section(self):
        content = """# System Prompt

You are given text fragments.
"""
        system, user = _parse_md_prompt(content)
        assert system == "You are given text fragments."
        assert user == ""

    def test_missing_system_section(self):
        content = """# User Prompt Template

Fragments:

{fragments}
"""
        system, user = _parse_md_prompt(content)
        assert system == ""
        assert user == "Fragments:\n\n{fragments}"

    def test_empty_content(self):
        system, user = _parse_md_prompt("")
        assert system == ""
        assert user == ""

    def test_empty_sections(self):
        content = "# System Prompt\n\n# User Prompt Template\n"
        system, user = _parse_md_prompt(content)
        assert system == ""
        assert user == ""

    def test_text_before_first_heading_ignored(self):
        content = """stray preamble

# System Prompt

sys text

# User Prompt Template

user text
"""
        system, user = _parse_md_prompt(content)
        assert system == "sys text"
        assert user == "user text"
