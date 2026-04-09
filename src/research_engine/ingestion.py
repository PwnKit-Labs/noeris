from __future__ import annotations

import json
from dataclasses import dataclass
import logging
import os
from typing import Protocol
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .components import SourceProvider
from .models import ResearchSource, ResearchTopic


ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
LOGGER = logging.getLogger(__name__)


class HttpClient(Protocol):
    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Fetch a JSON payload."""

    def get_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Fetch a text payload."""


class UrllibHttpClient:
    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict:
        return json.loads(self.get_text(url, headers=headers))

    def get_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": "noeris/0.1",
                **(headers or {}),
            },
        )
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")


@dataclass(slots=True)
class ArxivAtomSourceProvider(SourceProvider):
    """Source provider backed by the official arXiv Atom API."""

    client: HttpClient
    max_results: int = 5

    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        query = quote_plus(f'all:"{topic.name}"')
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query={query}&start=0&max_results={self.max_results}"
            "&sortBy=lastUpdatedDate&sortOrder=descending"
        )
        payload = self.client.get_text(url)
        root = ET.fromstring(payload)

        sources: list[ResearchSource] = []
        for entry in root.findall("atom:entry", ATOM_NAMESPACE):
            identifier = self._entry_text(entry, "atom:id")
            title = self._entry_text(entry, "atom:title")
            summary = self._entry_text(entry, "atom:summary")
            if not identifier or not title:
                continue
            sources.append(
                ResearchSource(
                    identifier=identifier,
                    kind="paper",
                    title=self._clean_text(title),
                    locator=identifier,
                    excerpt=self._clean_text(summary),
                )
            )
        return sources

    def _entry_text(self, entry: ET.Element, path: str) -> str:
        node = entry.find(path, ATOM_NAMESPACE)
        return "" if node is None or node.text is None else node.text

    def _clean_text(self, value: str) -> str:
        return " ".join(value.split())


@dataclass(slots=True)
class GitHubRepositorySourceProvider(SourceProvider):
    """Source provider backed by the public GitHub REST search API."""

    client: HttpClient
    max_results: int = 5

    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        query = quote_plus(f"{topic.name} in:name,description,readme")
        url = (
            "https://api.github.com/search/repositories"
            f"?q={query}&sort=updated&order=desc&per_page={self.max_results}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload = self.client.get_json(url, headers=headers)
        items = payload.get("items", [])
        sources: list[ResearchSource] = []
        for item in items:
            full_name = item.get("full_name")
            html_url = item.get("html_url")
            if not full_name or not html_url:
                continue
            sources.append(
                ResearchSource(
                    identifier=f"github://{full_name}",
                    kind="repository",
                    title=full_name,
                    locator=html_url,
                    excerpt=(item.get("description") or "").strip(),
                )
            )
        return sources


@dataclass(slots=True)
class CompositeSourceProvider(SourceProvider):
    providers: list[SourceProvider]

    def collect(self, topic: ResearchTopic) -> list[ResearchSource]:
        sources: list[ResearchSource] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                provider_sources = provider.collect(topic)
            except Exception:
                LOGGER.warning(
                    "Source provider %s failed for topic %r",
                    type(provider).__name__,
                    topic.name,
                    exc_info=True,
                )
                continue
            for source in provider_sources:
                if source.identifier in seen:
                    continue
                seen.add(source.identifier)
                sources.append(source)
        return sources
