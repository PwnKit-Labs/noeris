from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tests import _pathfix  # noqa: F401

from research_engine.llm import (
    LlmConfigurationError,
    LlmHypothesisPlanner,
    LlmResearchMemory,
    ResponsesApiClient,
    ResponsesProviderConfig,
    load_responses_provider_config,
)
from research_engine.models import Claim, ResearchContext, ResearchSource, ResearchTopic


class _FakeHttpClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        self.calls.append((url, payload, headers))
        return self.response


class ResponsesConfigTests(unittest.TestCase):
    def test_load_responses_provider_config_uses_codex_provider_shape(self) -> None:
        with patch("research_engine.llm.load_codex_provider_config") as mock_load:
            mock_load.return_value = type(
                "Config",
                (),
                {
                    "provider_name": "azure",
                    "secret_env_var": "AZURE_OPENAI_API_KEY",
                    "base_url": "https://example-resource.openai.azure.com/openai/v1",
                    "model": "gpt-5.4",
                    "wire_api": "responses",
                },
            )()
            with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "azure-key"}, clear=False):
                config = load_responses_provider_config()

        self.assertEqual(config.provider_name, "azure")
        self.assertEqual(config.model, "gpt-5.4")
        self.assertEqual(config.responses_url, "https://example-resource.openai.azure.com/openai/v1/responses")
        self.assertEqual(config.headers()["api-key"], "azure-key")

    def test_load_responses_provider_config_rejects_missing_values(self) -> None:
        with patch("research_engine.llm.load_codex_provider_config") as mock_load:
            mock_load.return_value = type(
                "Config",
                (),
                {
                    "provider_name": "azure",
                    "secret_env_var": "AZURE_OPENAI_API_KEY",
                    "base_url": None,
                    "model": "gpt-5.4",
                    "wire_api": "responses",
                },
            )()
            with self.assertRaises(LlmConfigurationError):
                load_responses_provider_config()


class ResponsesClientTests(unittest.TestCase):
    def test_generate_json_builds_structured_output_request(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"claims":[],"open_questions":[],"contradictions":[]}',
                            }
                        ],
                    }
                ]
            }
        )
        client = ResponsesApiClient(
            config=ResponsesProviderConfig(
                provider_name="azure",
                api_key="azure-key",
                base_url="https://example-resource.openai.azure.com/openai/v1",
                model="gpt-5.4",
            ),
            http_client=http_client,
        )

        payload = client.generate_json(
            schema_name="research_context",
            schema={"type": "object"},
            instructions="system text",
            prompt="user text",
        )

        self.assertEqual(payload["claims"], [])
        _, request_body, headers = http_client.calls[0]
        self.assertEqual(headers["api-key"], "azure-key")
        self.assertEqual(request_body["model"], "gpt-5.4")
        self.assertEqual(request_body["text"]["format"]["type"], "json_schema")
        self.assertEqual(request_body["input"][0]["content"][0]["type"], "input_text")

    def test_generate_json_supports_reasoning_and_token_controls(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": '{"ok": true}',
            }
        )
        client = ResponsesApiClient(
            config=ResponsesProviderConfig(
                provider_name="openai",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1-mini",
            ),
            http_client=http_client,
        )

        client.generate_json(
            schema_name="probe",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            instructions="system",
            prompt="user",
            max_output_tokens=123,
            reasoning_effort="low",
            text_verbosity="low",
        )

        _, request_body, _ = http_client.calls[0]
        self.assertEqual(request_body["max_output_tokens"], 123)
        self.assertEqual(request_body["reasoning"]["effort"], "low")
        self.assertEqual(request_body["text"]["verbosity"], "low")


class LlmPlannerTests(unittest.TestCase):
    def test_research_memory_sanitizes_unknown_sources(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Claim","source":"bad-source","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[]}'
                )
            }
        )
        client = ResponsesApiClient(
            config=ResponsesProviderConfig(
                provider_name="openai",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1-mini",
            ),
            http_client=http_client,
        )
        planner = LlmResearchMemory(client=client)
        topic = ResearchTopic(name="tool use", objective="improve reliability")
        sources = [
            ResearchSource(
                identifier="source-1",
                kind="paper",
                title="Source",
                locator="https://example.com",
                excerpt="Example",
            )
        ]

        context = planner.build_context(topic, sources)

        self.assertEqual(context.claims[0].source, "source-1")
        self.assertEqual(context.claims[0].evidence_refs, ["source-1"])

    def test_hypothesis_planner_filters_unknown_supporting_claims(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"hypotheses":['
                    '{"title":"Try retrieval routing","rationale":"Because",'
                    '"novelty_reason":"Gap in current memory routing","expected_signal":"Higher accuracy",'
                    '"supporting_claims":["Known claim","Missing claim"]},'
                    '{"title":"Try memory routing","rationale":"Because too",'
                    '"novelty_reason":"Gap 2","expected_signal":"Higher recall",'
                    '"supporting_claims":["Known claim"]},'
                    '{"title":"Try context pruning","rationale":"Because three",'
                    '"novelty_reason":"Gap 3","expected_signal":"Lower latency",'
                    '"supporting_claims":["Known claim"]},'
                    '{"title":"Too many","rationale":"Because four",'
                    '"novelty_reason":"Gap 4","expected_signal":"Other",'
                    '"supporting_claims":["Known claim"]}'
                    ']}'
                )
            }
        )
        client = ResponsesApiClient(
            config=ResponsesProviderConfig(
                provider_name="openai",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1-mini",
            ),
            http_client=http_client,
        )
        context = ResearchContext(
            topic="tool use",
            sources=[
                ResearchSource(
                    identifier="source-1",
                    kind="paper",
                    title="Source",
                    locator="https://example.com",
                    excerpt="Example",
                )
            ],
            claims=[
                Claim(
                    title="Known claim",
                    source="source-1",
                    summary="Summary",
                    evidence_refs=["source-1"],
                )
            ],
        )
        planner = LlmHypothesisPlanner(client=client)

        hypotheses = planner.plan(
            ResearchTopic(name="tool use", objective="improve reliability"),
            context,
        )

        self.assertEqual(len(hypotheses), 3)
        self.assertEqual(hypotheses[0].supporting_claims, ["Known claim"])


if __name__ == "__main__":
    unittest.main()
