# ResearchAgents

A multi-agent LLM review board for research ideas. You give it a
high-level research idea plus the resources you actually have (GPUs, time,
data, team, budget); a board of specialist agents analyzes it, debates it, and
returns a final recommendation — **PURSUE**, **PURSUE WITH MODIFICATIONS**, or
**DO NOT PURSUE** — together with a refined, executable version of the research
problem and a staged resource plan.

## How it works

```
                          ┌─────────────────────┐
        arXiv search ───► │   Novelty Analyst   │
                          └─────────┬───────────┘
                          ┌─────────▼───────────┐
                          │ Feasibility Analyst │  (against your GPU/time budget)
                          └─────────┬───────────┘
                          ┌─────────▼───────────┐
                          │   Impact Analyst    │
                          └─────────┬───────────┘
                          ┌─────────▼───────────┐
                          │ Methodology Analyst │
                          └─────────┬───────────┘
                    ┌───────────────▼────────────────┐
                    │   Advocate  ⇄  Critic debate   │  (N rounds)
                    └───────────────┬────────────────┘
                          ┌─────────▼───────────┐
                          │  Research Manager   │ ──► refined research problem
                          └─────────┬───────────┘
              ┌───────────────────────────────────────────┐
              │ Ambitious ⇄ Conservative ⇄ Pragmatic scope │
              └───────────────────┬───────────────────────┘
                          ┌───────▼─────────────┐
                          │  Program Director   │ ──► final recommendation
                          └─────────────────────┘
```

- **Analysts** produce structured reports. The Novelty Analyst grounds its
  assessment in real literature via two tools: live arXiv search (no API key
  required) and provider-native web search (OpenAI's Responses API `web_search`
  tool or Claude's server-side web search) for very recent results, published
  venue versions, lab blog posts, and open-source projects.
- **Advocate vs. Critic** debate the idea's merit over configurable rounds;
  the **Research Manager** judges the debate and rewrites the research problem
  to keep what survived scrutiny.
- **Three scopers** debate how big the project should be given your declared
  resources; the **Program Director** issues the final go/no-go decision with
  a staged execution plan and go/no-go gates.

Every run saves a full Markdown report (all analyst reports, complete debate
transcripts, and the final decision) plus a `summary.json` to
`~/.researchagents/results/`.

## Install

```bash
cd research-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in credentials
```

Default provider is OpenAI (set `OPENAI_API_KEY` in `.env`). Anthropic/Claude
is fully supported via `ANTHROPIC_API_KEY`; Google and Oracle-hosted ChatGPT
also work. Every role can use a different provider and model — see
Configuration below.

## Usage

### CLI

```bash
python -m cli.main \
  --idea "Can small LMs learn reliable tool use via curriculum distillation?" \
  --gpus "4x A100 80GB for 6 weeks" \
  --time "4 months to conference deadline" \
  --team "1 PhD student full-time" \
  --budget "\$2000 API credits" \
  --debug
```

Or keep the idea and resources in files:

```bash
python -m cli.main --idea-file idea.md --resources-file resources.md
```

### Python

```python
from researchagents.graph.research_graph import ResearchAgentsGraph
from researchagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["max_debate_rounds"] = 2

ra = ResearchAgentsGraph(debug=True, config=config)
state, recommendation = ra.propagate(
    research_idea="...",
    resources="- Compute: 4x A100 80GB\n- Time: 4 months\n- Team: 1 PhD student",
)
print(recommendation)
```

See `main.py` for a complete example.

## Configuration

Key options in `researchagents/default_config.py`:

| Option | Default | Meaning |
| --- | --- | --- |
| `llm_provider` | `openai` | `openai`, `anthropic`, `google`, `oracle`, ... |
| `deep_think_llm` | `gpt-5.2` | Default model for the two judges |
| `quick_think_llm` | `gpt-5-mini` | Default model for analysts and debaters |
| `role_llms` | `{}` | Per-role provider/model overrides (see below) |
| `web_search` | enabled, `openai` | Web search tool for the Novelty Analyst |
| `max_debate_rounds` | `5` | Advocate/critic back-and-forth rounds |
| `max_scope_rounds` | `3` | Scoping-team rounds |
| `max_lit_search_calls` | `10` | Search tool-call budget for the Novelty Analyst |
| `results_dir` | `~/.researchagents/results` | Where reports are written |

### Per-role LLMs

Any of the eleven roles can run on its own provider and model. Roles not
listed fall back to `deep_think_llm` (Research Manager, Program Director) or
`quick_think_llm` (everyone else):

```python
config["role_llms"] = {
    "Critic":           {"provider": "anthropic", "model": "claude-opus-5"},
    "Research Manager": {"provider": "anthropic", "model": "claude-opus-5"},
    "Program Director": {"provider": "anthropic", "model": "claude-opus-5"},
    "Novelty Analyst":  {"provider": "openai",    "model": "gpt-5.2"},
}
```

Valid role names: `Novelty Analyst`, `Feasibility Analyst`, `Impact Analyst`,
`Methodology Analyst`, `Advocate`, `Critic`, `Research Manager`,
`Ambitious Scoper`, `Conservative Scoper`, `Pragmatic Scoper`,
`Program Director`. Identical provider/model specs share one client instance.

### Web search

The Novelty Analyst gets a `web_search` tool alongside arXiv, backed by the
LLM provider's native search — no separate search-API key needed:

```python
config["web_search"] = {
    "enabled": True,
    "provider": "openai",     # "openai" (Responses API) or "anthropic" (Claude server-side)
    "model": None,             # None -> gpt-5.2 / claude-opus-5
    "max_uses": 3,             # searches per query (anthropic provider)
}
```

If the search call fails (e.g. missing key), the tool returns an error string
and the review degrades gracefully to arXiv-only.

## Tests

```bash
pytest
```

The test suite runs the full graph end-to-end with fake LLMs — no network or
API keys needed.

## Roadmap

- Memory/reflection: feed back actual project outcomes to calibrate future
  reviews (mirroring TradingAgents' reflection loop)
- Semantic Scholar / OpenReview search alongside arXiv
- Structured (JSON) final decisions for programmatic pipelines
- Batch mode: rank several candidate ideas against one resource budget
