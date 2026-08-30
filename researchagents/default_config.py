import os

_RESEARCHAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".researchagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv(
        "RESEARCHAGENTS_RESULTS_DIR", os.path.join(_RESEARCHAGENTS_HOME, "results")
    ),
    # LLM settings
    # Default: OpenAI via OPENAI_API_KEY. Also supported: "anthropic" (via
    # ANTHROPIC_API_KEY), "google", "oracle" (Oracle-hosted ChatGPT), and any
    # OpenAI-compatible provider.
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5-mini",
    # When None, each provider's client falls back to its own default endpoint.
    "backend_url": None,
    # Per-role LLM overrides. Keys are graph node names; each entry may set
    # "provider", "model", and optionally "base_url" (all fall back to the
    # defaults above). Roles not listed use deep_think_llm (Research Manager,
    # Program Director) or quick_think_llm (everyone else).
    # Example:
    # "role_llms": {
    #     "Critic": {"provider": "anthropic", "model": "claude-opus-5"},
    #     "Research Manager": {"provider": "anthropic", "model": "claude-opus-5"},
    #     "Novelty Analyst": {"provider": "openai", "model": "gpt-5.4"},
    # },
    "role_llms": {},
    # Web search for the Novelty Analyst (in addition to arXiv).
    # provider: "openai" uses the Responses API web_search tool;
    # "anthropic" uses Claude's server-side web search tool.
    # model None -> provider default (gpt-5.4 / claude-opus-5).
    "web_search": {
        "enabled": True,
        "provider": "openai",
        "model": None,
        "max_uses": 3,
    },
    # Debate settings
    "max_debate_rounds": 5,       # advocate <-> critic back-and-forth rounds
    "max_scope_rounds": 3,        # ambitious/conservative/pragmatic rounds
    "max_recur_limit": 100,
    # Literature search
    "arxiv_max_results": 20,       # results returned per arXiv query
    "max_lit_search_calls": 10,    # tool-call budget for the novelty analyst
}
