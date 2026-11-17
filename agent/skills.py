# agent/skills.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
import yaml

from agent.tools import ALL_TOOLS
from benchmark import __main__


@dataclass(frozen=True)
class Skill:
    name: str
    description: str        # Keep under ~20 words (always in system prompt)
    instructions: str   # Loaded into system prompt ONLY when active
    tools: list[BaseTool] = field(default_factory=list)

def parse_skill_file(file_path: str | Path) -> Skill:
    """Parses a SKILL.md file into a Skill object."""
    raw_text = Path(file_path).read_text(encoding="utf-8").strip()

    # Strip optional leading/trailing '---' YAML frontmatter fences
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        yaml_content = parts[1]
    elif raw_text.endswith("---"):
        yaml_content = raw_text[:-3]
    else:
        yaml_content = raw_text

    data = yaml.safe_load(yaml_content) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {file_path}")

    # Validate required fields
    missing = [k for k in ("name", "description", "instructions") if not data.get(k)]
    if missing:
        raise ValueError(f"Missing required field(s) {missing} in {file_path}")

    # Ensure tools is always a clean list of strings
    raw_tools = data.get("tools") or []
    if isinstance(raw_tools, str):
        tools_str = [t.strip() for t in raw_tools.split(",") if t.strip()]
    else:
        tools_str = [str(t).strip() for t in raw_tools]

    tools = []
    for tool_name in tools_str:
        for tool_object in ALL_TOOLS:
            if tool_object.name == tool_name:
                tools.append(tool_object)

    return Skill(
        name=str(data["name"]).strip(),
        description=str(data["description"]).strip(),
        instructions=str(data["instructions"]).strip(),
        tools=tools,
    )

def load_skills_from_dir(skills_dir: str | Path) -> dict[str, Skill]:
    """Scans a directory for SKILL.md files (both flat and nested folders)."""
    root = Path(skills_dir)
    if not root.exists():
        return {}

    skills: dict[str, Skill] = {}
    for skill_file in sorted(root.rglob("SKILL.md")):
        skill = parse_skill_file(skill_file)
        if skill.name in skills:
            raise ValueError(f"Duplicate skill name '{skill.name}' found in {skill_file}")
        skills[skill.name] = skill

    return skills


SKILLS_REGISTRY: dict[str, Skill] = load_skills_from_dir("agent/skills_dir")


@tool
def activate_skill(
    skill_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Activate a specialized skill to load its instructions and tools."""
    if skill_name not in SKILLS_REGISTRY:
        available = ", ".join(SKILLS_REGISTRY.keys()) or "none"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: Unknown skill '{skill_name}'. Available skills: {available}.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            "active_skill": skill_name,
            "messages": [
                ToolMessage(
                    content=f"Skill '{skill_name}' activated.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def get_skill_tools(skill_name: str | None) -> list[BaseTool]:
    """Collects all tools across all skills for ToolNode execution (Option A)."""
    return SKILLS_REGISTRY[skill_name].tools if skill_name is not None else []


def build_skill_aware_prompt(base_prompt: str, active_skill_name: str | None) -> str:
    """Appends the compact skill index and, if active, the active skill's full instructions."""
    if not SKILLS_REGISTRY:
        return base_prompt

    skill_index = "\n".join(
        f"- {skill.name}: {skill.description}" for skill in SKILLS_REGISTRY.values()
    )
    prompt = (
        f"{base_prompt}\n\n"
        f"Available Skills (call `activate_skill` when a task matches a skill):\n"
        f"{skill_index}"
    )

    if active_skill_name and active_skill_name in SKILLS_REGISTRY:
        active = SKILLS_REGISTRY[active_skill_name]
        prompt += f"\n\n[Active Skill: {active.name}]\n{active.instructions}"

    return prompt

if __name__ == "__main__":
    print(SKILLS_REGISTRY)