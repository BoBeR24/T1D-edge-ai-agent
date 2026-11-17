"""Skill loading for the benchmark agent.

A skill is a folder under benchmark/skills/ that contains a SKILL.md file:

    ---
    name: dawn-phenomenon
    description: One line saying when the skill applies.
    tools: [tool_a, tool_b]
    ---
    Step-by-step instructions for the model (Markdown).

"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Folder that contains one sub-folder per skill (benchmark/skills/).
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Oracle mapping: benchmark task category -> skill folder name.
# Categories not listed here run exactly as the baseline (no skill).
ORACLE_SKILL_FOR_CATEGORY: dict[str, str] = {
    "pattern_analysis": "dawn-phenomenon",
}


def parse_skill(path: Path) -> dict[str, Any]:
    """Read a SKILL.md file and return its parts as a dictionary.

    Returns:
        name:        skill name (from the header, or the folder name)
        description: one-line description (from the header)
        tools:       list of tool names the model may use with this skill
        body:        the instructions, ready to be appended to the system prompt
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: SKILL.md must start with a '---' header block")

    # The file looks like "---<header>---<body>": split it into those two parts.
    _, header, body = text.split("---", 2)

    # Header lines have the form "key: value".
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    name = meta.get("name") or path.parent.name
    # "[tool_a, tool_b]" -> ["tool_a", "tool_b"]
    tools = [t.strip() for t in meta.get("tools", "").strip("[]").split(",") if t.strip()]

    return {
        "name": name,
        "description": meta.get("description", ""),
        "tools": tools,
        "body": f"## Skill: {name}\n{body.strip()}",
    }


def load_oracle_skill(category: str) -> dict[str, Any] | None:
    """Return the skill mapped to a task category, or None (= baseline behaviour)."""
    name = ORACLE_SKILL_FOR_CATEGORY.get(category)
    if name is None:
        return None
    return parse_skill(SKILLS_DIR / name / "SKILL.md")