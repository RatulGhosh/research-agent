"""Command-line interface for the research review board.

Usage:
    python -m cli.main --idea "..." --gpus "4x A100 80GB" --time "6 weeks"
    python -m cli.main --idea-file idea.md --resources-file resources.md
"""

import argparse
import sys

from dotenv import load_dotenv

from researchagents.default_config import DEFAULT_CONFIG
from researchagents.graph.research_graph import ResearchAgentsGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchagents",
        description="Evaluate a research idea with a multi-agent review board.",
    )
    idea = parser.add_mutually_exclusive_group(required=True)
    idea.add_argument("--idea", help="Research idea as inline text")
    idea.add_argument("--idea-file", help="Path to a file containing the research idea")

    parser.add_argument(
        "--resources-file", help="Path to a file describing all available resources"
    )
    parser.add_argument("--gpus", help="GPU resources, e.g. '4x A100 80GB for 6 weeks'")
    parser.add_argument("--time", help="Time budget, e.g. '4 months to deadline'")
    parser.add_argument("--team", help="Team, e.g. '1 PhD student, advisor 2h/week'")
    parser.add_argument("--data", help="Data access, e.g. 'public benchmarks only'")
    parser.add_argument("--budget", help="Money budget, e.g. '$2000 API credits'")

    parser.add_argument(
        "--venue", help="Target conference/workshop name, e.g. 'NeurIPS 2027'"
    )
    parser.add_argument(
        "--track",
        help="Target track: main, findings, workshop, industry, demo, short, ...",
    )
    parser.add_argument("--venue-url", help="URL of the venue or its call for papers")

    parser.add_argument("--provider", default=None, help="LLM provider (default: config)")
    parser.add_argument("--deep-model", default=None, help="Deep-thinking model name")
    parser.add_argument("--quick-model", default=None, help="Quick-thinking model name")
    parser.add_argument(
        "--debate-rounds", type=int, default=None, help="Advocate/critic rounds"
    )
    parser.add_argument("--debug", action="store_true", help="Stream progress to stdout")
    return parser


def resolve_venue(args) -> str:
    parts = []
    if args.venue:
        parts.append(f"- Venue: {args.venue}")
    if args.track:
        parts.append(f"- Track: {args.track}")
    if args.venue_url:
        parts.append(f"- URL: {args.venue_url}")
    return "\n".join(parts)


def resolve_resources(args) -> str:
    if args.resources_file:
        with open(args.resources_file) as f:
            return f.read()

    parts = []
    for label, value in (
        ("Compute", args.gpus),
        ("Time budget", args.time),
        ("Team", args.team),
        ("Data access", args.data),
        ("Budget", args.budget),
    ):
        if value:
            parts.append(f"- {label}: {value}")
    if not parts:
        raise SystemExit(
            "Describe your resources via --resources-file or at least one of "
            "--gpus/--time/--team/--data/--budget."
        )
    return "\n".join(parts)


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.idea_file:
        with open(args.idea_file) as f:
            research_idea = f.read()
    else:
        research_idea = args.idea

    resources = resolve_resources(args)

    config = DEFAULT_CONFIG.copy()
    if args.provider:
        config["llm_provider"] = args.provider
    if args.deep_model:
        config["deep_think_llm"] = args.deep_model
    if args.quick_model:
        config["quick_think_llm"] = args.quick_model
    if args.debate_rounds is not None:
        config["max_debate_rounds"] = args.debate_rounds

    graph = ResearchAgentsGraph(debug=args.debug, config=config)
    _, recommendation = graph.propagate(research_idea, resources, venue=resolve_venue(args))

    print("\n" + "=" * 72)
    print("FINAL RECOMMENDATION")
    print("=" * 72 + "\n")
    print(recommendation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
