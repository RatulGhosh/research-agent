import os

_RESEARCHAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".researchagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv(
        "RESEARCHAGENTS_RESULTS_DIR", os.path.join(_RESEARCHAGENTS_HOME, "results")
    ),
    # LLM settings
    "llm_provider": "oracle",
    "deep_think_llm": "caa-gpt-5.4",
    "quick_think_llm": "caa-gpt-5-mini",
    # When None, each provider's client falls back to its own default endpoint.
    "backend_url": None,
    # Debate settings
    "max_debate_rounds": 2,       # advocate <-> critic back-and-forth rounds
    "max_scope_rounds": 1,        # ambitious/conservative/pragmatic rounds
    "max_recur_limit": 100,
    # Literature search
    "arxiv_max_results": 8,       # results returned per arXiv query
    "max_lit_search_calls": 4,    # tool-call budget for the novelty analyst
}
