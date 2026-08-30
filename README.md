# ResearchAgents

A multi-agent LLM review board for research ideas, modeled on
[TradingAgents](https://github.com/TauricResearch/TradingAgents). You give it a
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

- **Analysts** produce structured reports. The Novelty Analyst runs live
  arXiv searches (no API key required) to ground its assessment in real
  literature.
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

Default provider is Oracle-hosted ChatGPT (set `ORACLE_OPENAI_USERNAME` /
`ORACLE_OPENAI_PASSWORD` in `.env`). Standard OpenAI, Anthropic, and Google
providers are also supported via `config["llm_provider"]`.

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
| `llm_provider` | `oracle` | `oracle`, `openai`, `anthropic`, `google`, ... |
| `deep_think_llm` | `caa-gpt-5.4` | Model for the two judges |
| `quick_think_llm` | `caa-gpt-5-mini` | Model for analysts and debaters |
| `max_debate_rounds` | `2` | Advocate/critic back-and-forth rounds |
| `max_scope_rounds` | `1` | Scoping-team rounds |
| `max_lit_search_calls` | `4` | arXiv tool-call budget for the Novelty Analyst |
| `results_dir` | `~/.researchagents/results` | Where reports are written |

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
