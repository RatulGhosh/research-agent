"""Minimal arXiv search client using the public Atom API (no API key needed)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, max_results: int = 8) -> str:
    """Search arXiv and return a formatted digest of matching papers.

    Args:
        query: Free-text search query (title/abstract keywords).
        max_results: Maximum number of papers to return.

    Returns:
        A human-readable digest with title, authors, date, and abstract for
        each match, or a message if nothing was found / the request failed.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"arXiv search failed for query '{query}': {exc}"

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        return f"arXiv returned an unparseable response for query '{query}': {exc}"

    entries = root.findall(f"{_ATOM_NS}entry")
    if not entries:
        return f"No arXiv results found for query '{query}'."

    sections = []
    for entry in entries:
        title = _text(entry, "title")
        summary = _text(entry, "summary")
        published = _text(entry, "published")[:10]
        link = _text(entry, "id")
        authors = ", ".join(
            _text(author, "name") for author in entry.findall(f"{_ATOM_NS}author")
        )
        sections.append(
            f"### {title}\n"
            f"- Authors: {authors}\n"
            f"- Published: {published}\n"
            f"- Link: {link}\n"
            f"- Abstract: {summary}"
        )

    return f"## arXiv results for '{query}'\n\n" + "\n\n".join(sections)


def _text(element, tag: str) -> str:
    node = element.find(f"{_ATOM_NS}{tag}")
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())
