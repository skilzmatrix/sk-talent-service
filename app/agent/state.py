"""Agent state notes.

The compiled ReAct graph (`create_react_agent`) uses LangGraph’s internal schema
(`messages`, managed step counters). API routes and persistence only serialize
`messages` (LangChain `BaseMessage` list).

For Graph-of-Thoughts *structure* (ThoughtNode / edges), see `got.py`; optional
future extension could add a `thought_graph` channel with reducers.
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Minimal chat-shaped state used when reasoning about persistence."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
