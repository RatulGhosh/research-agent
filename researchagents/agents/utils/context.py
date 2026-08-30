def format_venue_context(state) -> str:
    """Venue block for agent prompts; empty string when no venue was given."""
    venue = (state.get("target_venue") or "").strip()
    if not venue:
        return ""
    venue_report = state.get("venue_report", "")
    block = f"\nTarget venue/track:\n{venue}\n"
    if venue_report:
        block += f"\nVenue analyst's report (CFP scope, deadlines, acceptance bar):\n{venue_report}\n"
    return block
