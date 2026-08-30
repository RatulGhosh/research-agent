from dotenv import load_dotenv

from researchagents.default_config import DEFAULT_CONFIG
from researchagents.graph.research_graph import ResearchAgentsGraph

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "oracle"       # Oracle-hosted ChatGPT
config["deep_think_llm"] = "caa-gpt-5.4"
config["quick_think_llm"] = "caa-gpt-5-mini"
config["max_debate_rounds"] = 2

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
