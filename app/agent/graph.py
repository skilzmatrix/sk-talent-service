"""ReAct agent (LangGraph `create_react_agent`) + Graph-of-Thoughts (GoT) prompt layer."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL_AGENT
from .tools import search_candidates, get_candidate_details
from .got import build_react_got_system_prompt

tools = [search_candidates, get_candidate_details]

# include_thoughts=True is required for Gemini 3.x multi-turn tool calls:
# the API attaches a thought signature to each model turn; subsequent messages
# must echo it back or the API returns a 400. langchain-google-genai 3.1+
# handles this automatically when the flag is set.
#
# IMPORTANT: with include_thoughts=True, streaming chunk.content is a list of
# typed parts [{"type":"thinking",...}, {"type":"text","text":"..."}], NOT a
# plain string. The SSE stream handler in chat.py handles this correctly.
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_AGENT,
    api_key=GEMINI_API_KEY,
    temperature=0.2,
    include_thoughts=True,
)

BASE_SYSTEM_PROMPT = """You are an expert AI Recruiter Copilot. Your role is to help recruiters \
find and evaluate candidates.

You have access to a database of candidate resumes.

Rules:
- When asked to find candidates, ALWAYS call `search_candidates` first.
- When asked for more detail on a specific candidate, ALWAYS call `get_candidate_details`.
- Never fabricate candidates, names, or skills. Only report what the tools return.
- If a tool returns an error, report it clearly and suggest what the user can try next.
- Keep final responses concise: lead with the most relevant candidates, then summarise \
  strengths and gaps objectively.
"""

RECRUITER_SYSTEM_PROMPT = build_react_got_system_prompt(BASE_SYSTEM_PROMPT)

# recursion_limit caps the number of agent ↔ tool round-trips to prevent runaway
# loops; 15 is generous for any realistic recruiter query.
app = create_react_agent(
    llm,
    tools,
    prompt=RECRUITER_SYSTEM_PROMPT,
)
