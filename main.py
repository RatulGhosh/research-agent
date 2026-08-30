from dotenv import load_dotenv

from researchagents.default_config import DEFAULT_CONFIG
from researchagents.graph.research_graph import ResearchAgentsGraph

# Load environment variables from .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # default provider for unlisted roles
config["deep_think_llm"] = "gpt-5.6"
config["quick_think_llm"] = "gpt-5.6-terra"

# Mix providers per role: e.g. Claude as the adversarial critic and the two
# judges, OpenAI everywhere else. Any role can name any provider/model.
config["role_llms"] = {
    "Critic": {"provider": "anthropic", "model": "claude-opus-5"},
    "Research Manager": {"provider": "anthropic", "model": "claude-opus-5"},
    "Program Director": {"provider": "anthropic", "model": "claude-opus-5"},
}

# Web search for the Novelty Analyst (alongside arXiv):
# "openai" -> Responses API web_search tool; "anthropic" -> Claude server-side search
config["web_search"] = {"enabled": True, "provider": "openai", "model": None, "max_uses": 3}

ra = ResearchAgentsGraph(debug=True, config=config)

research_idea = """
Investigate whether small language models (<3B params) can learn to use tools
reliably through synthetic curriculum distillation from a frontier model,
matching the tool-use accuracy of models 10x their size on multi-step tasks.
"""

resources = """
- Compute: 4x NVIDIA A100 80GB for 6 weeks (dedicated), plus small-scale
  debugging on 1x RTX 4090
- API budget: $2,000 for frontier-model distillation data generation
- Data: open tool-use benchmarks (ToolBench, BFCL, API-Bank)
- Team: 1 PhD student full-time, 1 advisor at 2 hrs/week
- Timeline: aiming for a top ML conference deadline in ~4 months
"""

_, recommendation = ra.propagate(research_idea, resources)
print(recommendation)
