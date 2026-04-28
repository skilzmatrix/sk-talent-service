"""Graph of Thoughts (GoT) scaffolding: structured multi-step reasoning over tool use.

The full GoT paper (Yao et al., variants) uses explicit thought vertices and edges
with keep/merge/refine operators. We encode the same *behavior* in the system prompt
(brief verbal decomposition → act → merge with observations) while ReAct (LangGraph
`create_react_agent`) handles the execution loop. Types below document a future
state/SSE layer if we persist a literal thought graph per turn.
"""

from __future__ import annotations

from typing import List, Literal, NotRequired, TypedDict


class ThoughtNode(TypedDict):
    """A single node in a reasoning graph."""

    id: str
    parents: List[str]  # upstream thought ids; empty = root
    content: str
    kind: Literal["plan", "hypothesis", "critique", "aggregate", "post_tool"]


class ThoughtLink(TypedDict):
    """Directed edge (parent → child) for visualization or audit."""

    source: str
    target: str
    label: NotRequired[str]


# Prompt appendix: GoT-style behavior on top of ReAct (no extra graph nodes).
GRAPH_OF_THOUGHTS_INSTRUCTIONS = """

## Graph-of-Thoughts discipline (use mentally before you answer)

1. **Decompose** the user’s request into small sub-questions (implicit “nodes”). Note what must be true before a tool call and what is unknown.
2. **Relate** sub-questions: if one answer depends on another, order tools accordingly (edges). Avoid redundant searches.
3. **Act** with tools only when the plan requires fresh data. After each **ToolMessage**, **revise** your internal plan: keep, refine, or replace earlier thoughts—do not contradict evidence.
4. **Aggregate** partial conclusions into a single, coherent answer for the recruiter. Prefer clarity over long chain-of-thought in the final reply; keep intermediate reasoning private unless the user asked for it.

When the task is simple (e.g. one clear search or one follow-up), you may skip explicit enumeration but still follow order (plan → tools → final answer).
"""


def build_react_got_system_prompt(base_system: str) -> str:
    """Prepend/append GoT section to the recruiter system prompt."""
    return base_system.rstrip() + "\n" + GRAPH_OF_THOUGHTS_INSTRUCTIONS
