# agent/graph.py
# Wires the LLM and tools into a ReAct loop using LangGraph.
#
# Flow:
#   START → [agent node] ──has tool_calls?──> [ToolNode] ──> [agent node] (loop)
#                        ──no tool_calls?──> END (resets active_skill)
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import get_llm
from agent.skills import (
    SKILLS_REGISTRY,
    activate_skill,
    build_skill_aware_prompt,
    get_skill_tools,
)
from agent.tools import CORE_TOOLS

SYSTEM_PROMPT = (
    "You are an assistant that helps manage diabetes. You have access to the user's CGM sensor data. "
    "For historical and time-range CGM analysis, prefer SQLite-backed tools "
    "(get_recent_cgm_readings, get_cgm_readings, get_cgm_summary_from_db, "
    "get_cgm_spikes_from_db, get_nearest_cgm_reading). "
    "You can help users calculate insulin dosing for meals using their personal insulin-to-carb ratios. "
    "When a user mentions a meal with carbs, proactively use calculate_insulin_dosage. "
    "Always remind users this is informational and not medical advice."
)


class AgentState(MessagesState):
    active_skill: str | None = None


def _agent_node(state: AgentState, llm_by_skill: dict):
    active_skill = state.get("active_skill")

    # Select the LLM bound only to core tools + active skill's tools
    llm = llm_by_skill.get(active_skill, llm_by_skill[None])

    # Ephemerally construct system prompt (never saved to state["messages"])
    # TODO: Right now the active skill is injected in the initial system prompt, this may affect logical flow of the agent.
    #  Perhaps a better approach would be adding skill context with a new tool message and then delete it once skill is deactivated.
    dynamic_system_prompt = build_skill_aware_prompt(SYSTEM_PROMPT, active_skill)
    messages = [
        m for m in state["messages"] if getattr(m, "type", None) != "system"
    ]
    messages = [SystemMessage(content=dynamic_system_prompt)] + messages

    response = llm.invoke(messages)

    # Reset active_skill when the agent finishes its turn (no more tool calls).
    # This assumes that there is no model calls afterward that need a loaded skill.
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    next_active_skill = active_skill if has_tool_calls else None

    return {
        "messages": [response],
        "active_skill": next_active_skill,
    }

def _tool_node(state: AgentState, core_tools):
    # Tool node only exposes core tools + tools from active skill
    return ToolNode([*core_tools, *get_skill_tools(state['active_skill'])])


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(model_name: str | None = None):
    base_llm = get_llm(model_name=model_name)
    core_tools = [*CORE_TOOLS, activate_skill]

    # Pre-bind LLM variants so only relevant tool schemas are sent per state
    llm_by_skill = {
        None: base_llm.bind_tools(core_tools),
        **{
            name: base_llm.bind_tools([*core_tools, *skill.tools])
            for name, skill in SKILLS_REGISTRY.items()
        },
    } # NOTE careful with this solution, may result in unexpected behavior

    def tool_node(state: AgentState):
        return _tool_node(state, core_tools)

    def agent_node(state: AgentState):
        return _agent_node(state, llm_by_skill)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


graph = build_graph()