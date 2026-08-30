from unittest.mock import patch

import requests

from researchagents.dataflows.arxiv_search import search_arxiv

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <published>2024-01-15T00:00:00Z</published>
    <title>Sample Paper on Tool Use</title>
    <summary>We study tool use in small language models.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
</feed>
"""


class FakeResponse:
    status_code = 200
    text = SAMPLE_FEED

    def raise_for_status(self):
        pass


def test_search_arxiv_formats_results():
    with patch("researchagents.dataflows.arxiv_search.requests.get", return_value=FakeResponse()):
        result = search_arxiv("tool use")
    assert "Sample Paper on Tool Use" in result
    assert "Ada Lovelace, Alan Turing" in result
    assert "2024-01-15" in result
    assert "http://arxiv.org/abs/1234.5678v1" in result


def test_search_arxiv_handles_request_failure():
    with patch(
        "researchagents.dataflows.arxiv_search.requests.get",
        side_effect=requests.ConnectionError("boom"),
    ):
        result = search_arxiv("tool use")
    assert "arXiv search failed" in result


def test_search_arxiv_handles_empty_feed():
    class EmptyResponse(FakeResponse):
        text = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    with patch("researchagents.dataflows.arxiv_search.requests.get", return_value=EmptyResponse()):
        result = search_arxiv("nonexistent topic")
    assert "No arXiv results" in result
