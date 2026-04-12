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
from research_engine.models import Contradiction, SourceAssessment


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

    def test_load_responses_provider_config_uses_azure_env_without_codex_config(self) -> None:
        with patch("research_engine.llm.load_codex_provider_config", return_value=None):
            with patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_KEY": "azure-key",
                    "AZURE_OPENAI_BASE_URL": "https://example-resource.openai.azure.com/openai/v1",
                    "AZURE_OPENAI_MODEL": "gpt-5.4",
                },
                clear=False,
            ):
                config = load_responses_provider_config()

        self.assertEqual(config.provider_name, "azure")
        self.assertEqual(config.headers()["api-key"], "azure-key")

    def test_load_responses_provider_config_rejects_partial_azure_env(self) -> None:
        with patch("research_engine.llm.load_codex_provider_config", return_value=None):
            with patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_KEY": "azure-key",
                    "AZURE_OPENAI_BASE_URL": "",
                    "AZURE_OPENAI_MODEL": "",
                },
                clear=False,
            ):
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

    def test_generate_json_reads_output_text_from_message_content_when_top_level_missing(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"ok": true}',
                            }
                        ],
                    },
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
            schema_name="probe",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            instructions="system",
            prompt="user",
        )

        self.assertTrue(payload["ok"])


class LlmPlannerTests(unittest.TestCase):
    def test_research_memory_sanitizes_unknown_sources(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Claim","source":"bad-source","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[{"title":"Conflict","summary":"Sources disagree","claim_titles":["Claim"],"severity":"high"}],'
                    '"source_assessments":[{"source_id":"source-1","confidence":"high","rationale":"Direct primary source"}]}'
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
                updated_at="2026-04-10T12:00:00Z",
            )
        ]

        context = planner.build_context(topic, sources)

        self.assertEqual(context.claims[0].source, "source-1")
        self.assertEqual(context.claims[0].evidence_refs, ["source-1"])
        self.assertEqual(context.source_assessments[0].confidence, "high")
        self.assertEqual(context.contradictions[0].severity, "high")
        _, request_body, _ = http_client.calls[0]
        self.assertEqual(
            request_body["input"][1]["content"][0]["text"].count("updated_at"),
            1,
        )

    def test_research_memory_infers_claim_link_for_single_claim_contradiction(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Only claim","source":"source-1","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[{"title":"Conflict","summary":"Sources disagree","claim_titles":[],"severity":"high"}],'
                    '"source_assessments":[{"source_id":"source-1","confidence":"high","rationale":"Direct primary source"}]}'
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
        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            [
                ResearchSource(
                    identifier="source-1",
                    kind="paper",
                    title="Source",
                    locator="https://example.com",
                    excerpt="Example",
                )
            ],
        )

        self.assertEqual(context.contradictions[0].claim_titles, ["Only claim"])

    def test_research_memory_infers_claim_links_from_text_when_titles_missing(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":['
                    '{"title":"Fresh memory claim","source":"source-1","summary":"Summary","evidence_refs":["source-1"]},'
                    '{"title":"Older retrieval claim","source":"source-2","summary":"Summary","evidence_refs":["source-2"]}'
                    '],"open_questions":["What next?"],'
                    '"contradictions":[{"title":"Fresh memory claim conflict","summary":"Fresh memory claim conflicts with older retrieval claim","claim_titles":[],"severity":"medium"}],'
                    '"source_assessments":['
                    '{"source_id":"source-1","confidence":"high","rationale":"Direct primary source"},'
                    '{"source_id":"source-2","confidence":"medium","rationale":"Secondary source"}'
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
        planner = LlmResearchMemory(client=client)
        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            [
                ResearchSource(
                    identifier="source-1",
                    kind="paper",
                    title="Fresh source",
                    locator="https://example.com/1",
                    excerpt="Example",
                ),
                ResearchSource(
                    identifier="source-2",
                    kind="paper",
                    title="Older source",
                    locator="https://example.com/2",
                    excerpt="Example",
                ),
            ],
        )

        self.assertIn("Fresh memory claim", context.contradictions[0].claim_titles)

    def test_research_memory_backfills_assessments_for_unscored_sources(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Claim","source":"source-1","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[],"source_assessments":[]}'
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
        sources = [
            ResearchSource(
                identifier="source-1",
                kind="paper",
                title="Source",
                locator="https://example.com",
                excerpt="Example",
                updated_at="2026-04-10T12:00:00Z",
            )
        ]

        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            sources,
        )

        self.assertEqual(len(context.source_assessments), 1)
        self.assertEqual(context.source_assessments[0].confidence, "medium")
        self.assertIn("Backfilled default assessment", context.source_assessments[0].rationale)

    def test_research_memory_normalizes_confidence_synonyms(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Claim","source":"source-1","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[],"source_assessments":[{"source_id":"source-1","confidence":"strong","rationale":"Direct primary source"}]}'
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
        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            [
                ResearchSource(
                    identifier="source-1",
                    kind="paper",
                    title="Source",
                    locator="https://example.com",
                    excerpt="Example",
                )
            ],
        )

        self.assertEqual(context.source_assessments[0].confidence, "high")

    def test_research_memory_keeps_stronger_duplicate_assessment(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[{"title":"Claim","source":"source-1","summary":"Summary",'
                    '"evidence_refs":["source-1"]}],"open_questions":["What next?"],'
                    '"contradictions":[],"source_assessments":['
                    '{"source_id":"source-1","confidence":"low","rationale":"Weak hint"},'
                    '{"source_id":"source-1","confidence":"high","rationale":"Direct primary source with explicit benchmark evidence"}'
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
        planner = LlmResearchMemory(client=client)
        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            [
                ResearchSource(
                    identifier="source-1",
                    kind="paper",
                    title="Source",
                    locator="https://example.com",
                    excerpt="Example",
                )
            ],
        )

        self.assertEqual(len(context.source_assessments), 1)
        self.assertEqual(context.source_assessments[0].confidence, "high")
        self.assertIn("benchmark evidence", context.source_assessments[0].rationale)

    def test_research_memory_falls_back_to_source_claims_when_model_returns_none(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"claims":[],"open_questions":["What next?"],"contradictions":[],"source_assessments":[]}'
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
        sources = [
            ResearchSource(
                identifier="source-1",
                kind="repository",
                title="Source repo",
                locator="https://example.com",
                excerpt="Example",
            )
        ]

        context = planner.build_context(
            ResearchTopic(name="tool use", objective="improve reliability"),
            sources,
        )

        self.assertEqual(context.claims[0].evidence_kind, "source-derived")
        self.assertIn("implementation evidence", context.claims[0].title)

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

    def test_hypothesis_planner_ranks_by_confidence_and_contradictions(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"hypotheses":['
                    '{"title":"Weak idea","rationale":"Because",'
                    '"novelty_reason":"Gap","expected_signal":"Higher accuracy",'
                    '"supporting_claims":["Weak claim"]},'
                    '{"title":"Strong contradiction idea","rationale":"Because too",'
                    '"novelty_reason":"Gap 2","expected_signal":"Higher recall",'
                    '"supporting_claims":["Strong claim"]}'
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
                    title="Strong source",
                    locator="https://example.com/1",
                    excerpt="Example",
                    updated_at="2026-04-11T12:00:00Z",
                ),
                ResearchSource(
                    identifier="source-2",
                    kind="paper",
                    title="Weak source",
                    locator="https://example.com/2",
                    excerpt="Example",
                    updated_at="2026-04-08T12:00:00Z",
                ),
            ],
            source_assessments=[
                SourceAssessment(
                    source_id="source-1",
                    confidence="high",
                    rationale="Primary evidence",
                ),
                SourceAssessment(
                    source_id="source-2",
                    confidence="low",
                    rationale="Weak evidence",
                ),
            ],
            claims=[
                Claim(
                    title="Strong claim",
                    source="source-1",
                    summary="Summary",
                    evidence_refs=["source-1"],
                ),
                Claim(
                    title="Weak claim",
                    source="source-2",
                    summary="Summary",
                    evidence_refs=["source-2"],
                ),
            ],
            contradictions=[
                Contradiction(
                    title="Conflict",
                    summary="Strong claim conflicts with prior work",
                    claim_titles=["Strong claim"],
                    severity="high",
                )
            ],
        )
        planner = LlmHypothesisPlanner(client=client)

        hypotheses = planner.plan(
            ResearchTopic(name="tool use", objective="improve reliability"),
            context,
        )

        self.assertEqual(hypotheses[0].title, "Strong contradiction idea")
        self.assertGreater(hypotheses[0].priority_score, hypotheses[1].priority_score)
        self.assertIn("contradiction bonus", hypotheses[0].ranking_rationale)

    def test_hypothesis_planner_prefers_fresher_support_when_confidence_ties(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"hypotheses":['
                    '{"title":"Older-source idea","rationale":"Because",'
                    '"novelty_reason":"Gap","expected_signal":"Higher accuracy",'
                    '"supporting_claims":["Older claim"]},'
                    '{"title":"Fresher-source idea","rationale":"Because too",'
                    '"novelty_reason":"Gap 2","expected_signal":"Higher recall",'
                    '"supporting_claims":["Fresher claim"]}'
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
                    title="Fresh source",
                    locator="https://example.com/1",
                    excerpt="Example",
                    updated_at="2026-04-11T12:00:00Z",
                ),
                ResearchSource(
                    identifier="source-2",
                    kind="paper",
                    title="Older source",
                    locator="https://example.com/2",
                    excerpt="Example",
                    updated_at="2026-04-08T12:00:00Z",
                ),
            ],
            source_assessments=[
                SourceAssessment(
                    source_id="source-1",
                    confidence="medium",
                    rationale="Recent evidence",
                ),
                SourceAssessment(
                    source_id="source-2",
                    confidence="medium",
                    rationale="Older evidence",
                ),
            ],
            claims=[
                Claim(
                    title="Fresher claim",
                    source="source-1",
                    summary="Summary",
                    evidence_refs=["source-1"],
                    evidence_kind="llm-extracted",
                ),
                Claim(
                    title="Older claim",
                    source="source-2",
                    summary="Summary",
                    evidence_refs=["source-2"],
                    evidence_kind="llm-extracted",
                ),
            ],
        )
        planner = LlmHypothesisPlanner(client=client)

        hypotheses = planner.plan(
            ResearchTopic(name="tool use", objective="improve reliability"),
            context,
        )

        self.assertEqual(hypotheses[0].title, "Fresher-source idea")
        self.assertIn("freshness bonus", hypotheses[0].ranking_rationale)

    def test_hypothesis_planner_penalizes_much_older_support(self) -> None:
        http_client = _FakeHttpClient(
            {
                "output_text": (
                    '{"hypotheses":['
                    '{"title":"Very old source idea","rationale":"Because",'
                    '"novelty_reason":"Gap","expected_signal":"Higher accuracy",'
                    '"supporting_claims":["Very old claim"]},'
                    '{"title":"Recent source idea","rationale":"Because too",'
                    '"novelty_reason":"Gap 2","expected_signal":"Higher recall",'
                    '"supporting_claims":["Recent claim"]}'
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
                    title="Recent source",
                    locator="https://example.com/1",
                    excerpt="Example",
                    updated_at="2026-04-11T12:00:00Z",
                ),
                ResearchSource(
                    identifier="source-2",
                    kind="paper",
                    title="Very old source",
                    locator="https://example.com/2",
                    excerpt="Example",
                    updated_at="2026-01-01T12:00:00Z",
                ),
            ],
            source_assessments=[
                SourceAssessment(
                    source_id="source-1",
                    confidence="medium",
                    rationale="Recent evidence",
                ),
                SourceAssessment(
                    source_id="source-2",
                    confidence="medium",
                    rationale="Older evidence",
                ),
            ],
            claims=[
                Claim(
                    title="Recent claim",
                    source="source-1",
                    summary="Summary",
                    evidence_refs=["source-1"],
                    evidence_kind="llm-extracted",
                ),
                Claim(
                    title="Very old claim",
                    source="source-2",
                    summary="Summary",
                    evidence_refs=["source-2"],
                    evidence_kind="llm-extracted",
                ),
            ],
        )
        planner = LlmHypothesisPlanner(client=client)

        hypotheses = planner.plan(
            ResearchTopic(name="tool use", objective="improve reliability"),
            context,
        )

        self.assertEqual(hypotheses[0].title, "Recent source idea")


if __name__ == "__main__":
    unittest.main()
