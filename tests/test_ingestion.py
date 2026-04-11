import os
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.ingestion import (
    ArxivAtomSourceProvider,
    CompositeSourceProvider,
    GitHubRepositorySourceProvider,
)
from research_engine.models import ResearchTopic


ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>  Test Paper Title  </title>
    <updated>2026-04-10T12:34:56Z</updated>
    <summary>
      This is a sample abstract.
    </summary>
  </entry>
</feed>
"""


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict:
        self.calls.append((url, headers))
        return {
            "items": [
                {
                    "full_name": "openai/example-repo",
                    "html_url": "https://github.com/openai/example-repo",
                    "description": "Example repository",
                    "updated_at": "2026-04-09T08:00:00Z",
                }
            ]
        }

    def get_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        self.calls.append((url, headers))
        return ARXIV_SAMPLE


class IngestionProviderTests(unittest.TestCase):
    def test_arxiv_provider_parses_atom_entries(self) -> None:
        client = FakeHttpClient()
        provider = ArxivAtomSourceProvider(client=client, max_results=2)

        sources = provider.collect(
            ResearchTopic(name="long-context reasoning", objective="discover sources")
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].kind, "paper")
        self.assertEqual(sources[0].title, "Test Paper Title")
        self.assertEqual(sources[0].updated_at, "2026-04-10T12:34:56Z")
        self.assertIn("search_query=", client.calls[0][0])

    def test_github_provider_parses_repository_results(self) -> None:
        client = FakeHttpClient()
        provider = GitHubRepositorySourceProvider(client=client, max_results=2)

        sources = provider.collect(
            ResearchTopic(name="tool use", objective="discover sources")
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].identifier, "github://openai/example-repo")
        self.assertEqual(sources[0].kind, "repository")
        self.assertEqual(sources[0].updated_at, "2026-04-09T08:00:00Z")
        self.assertEqual(
            client.calls[0][1]["X-GitHub-Api-Version"],
            "2022-11-28",
        )

    def test_github_provider_uses_auth_token_when_available(self) -> None:
        client = FakeHttpClient()
        provider = GitHubRepositorySourceProvider(client=client, max_results=1)
        previous = os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = "test-token"
        try:
            provider.collect(ResearchTopic(name="memory", objective="discover"))
        finally:
            if previous is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = previous

        self.assertEqual(
            client.calls[0][1]["Authorization"],
            "Bearer test-token",
        )

    def test_composite_provider_deduplicates_by_identifier(self) -> None:
        topic = ResearchTopic(name="memory", objective="discover sources")

        class DuplicateProvider:
            def collect(self, topic: ResearchTopic):
                del topic
                return [
                    GitHubRepositorySourceProvider(client=FakeHttpClient(), max_results=1).collect(
                        ResearchTopic(name="memory", objective="discover sources")
                    )[0]
                ]

        source = GitHubRepositorySourceProvider(client=FakeHttpClient(), max_results=1).collect(
            topic
        )[0]

        class StaticProvider:
            def collect(self, topic: ResearchTopic):
                del topic
                return [source]

        composite = CompositeSourceProvider(
            providers=[StaticProvider(), DuplicateProvider()]
        )

        sources = composite.collect(topic)
        self.assertEqual(len(sources), 1)

    def test_composite_provider_continues_when_one_provider_fails(self) -> None:
        topic = ResearchTopic(name="memory", objective="discover sources")

        class BrokenProvider:
            def collect(self, topic: ResearchTopic):
                del topic
                raise RuntimeError("boom")

        client = FakeHttpClient()
        good_provider = GitHubRepositorySourceProvider(client=client, max_results=1)
        composite = CompositeSourceProvider(providers=[BrokenProvider(), good_provider])

        with self.assertLogs("research_engine.ingestion", level="WARNING") as captured:
            sources = composite.collect(topic)

        self.assertEqual(len(sources), 1)
        self.assertIn("Source provider BrokenProvider failed", captured.output[0])


if __name__ == "__main__":
    unittest.main()
