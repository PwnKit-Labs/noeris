from __future__ import annotations

import json
from dataclasses import dataclass
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .codex_config import load_codex_provider_config
from .components import HypothesisPlanner, ResearchMemory
from .models import (
    Claim,
    Contradiction,
    Hypothesis,
    ResearchContext,
    ResearchSource,
    ResearchTopic,
    SourceAssessment,
)


class LlmConfigurationError(RuntimeError):
    """Raised when the configured LLM provider is not usable."""


class LlmApiError(RuntimeError):
    """Raised when the upstream LLM API returns an error."""


class JsonHttpClient(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        """POST a JSON payload and return a JSON response."""


@dataclass(slots=True)
class UrllibJsonHttpClient:
    timeout_seconds: int = 60

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlmApiError(detail or str(exc)) from exc
        except URLError as exc:
            raise LlmApiError(str(exc.reason)) from exc


@dataclass(slots=True)
class ResponsesProviderConfig:
    provider_name: str
    api_key: str
    base_url: str
    model: str

    @property
    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider_name == "azure":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


@dataclass(slots=True)
class ResponsesJsonResult:
    data: dict[str, object]
    raw_response: dict[str, object]


def load_responses_provider_config() -> ResponsesProviderConfig:
    provider = load_codex_provider_config()
    if provider is not None:
        if provider.wire_api != "responses":
            raise LlmConfigurationError(
                f"Configured provider {provider.provider_name!r} does not use the Responses API."
            )
        missing_fields = []
        if not provider.base_url:
            missing_fields.append("base_url")
        if not provider.model:
            missing_fields.append("model")
        api_key = os.getenv(provider.secret_env_var)
        if not api_key:
            missing_fields.append(provider.secret_env_var)
        if missing_fields:
            raise LlmConfigurationError(
                "Configured Codex provider is missing required values: "
                + ", ".join(missing_fields)
            )
        return ResponsesProviderConfig(
            provider_name=provider.provider_name,
            api_key=api_key,
            base_url=provider.base_url,
            model=provider.model,
        )

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    azure_model = os.getenv("AZURE_OPENAI_MODEL")
    if azure_api_key:
        missing_fields = []
        if not azure_base_url:
            missing_fields.append("AZURE_OPENAI_BASE_URL")
        if not azure_model:
            missing_fields.append("AZURE_OPENAI_MODEL")
        if missing_fields:
            raise LlmConfigurationError(
                "Azure OpenAI environment is missing required values: "
                + ", ".join(missing_fields)
            )
        return ResponsesProviderConfig(
            provider_name="azure",
            api_key=azure_api_key,
            base_url=azure_base_url,
            model=azure_model,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LlmConfigurationError(
            "No Codex provider config was found, AZURE_OPENAI_API_KEY is not set, and OPENAI_API_KEY is not set."
        )
    return ResponsesProviderConfig(
        provider_name="openai",
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )


@dataclass(slots=True)
class ResponsesApiClient:
    config: ResponsesProviderConfig
    http_client: JsonHttpClient

    @classmethod
    def from_environment(cls) -> ResponsesApiClient:
        return cls(
            config=load_responses_provider_config(),
            http_client=UrllibJsonHttpClient(),
        )

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        instructions: str,
        prompt: str,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        text_verbosity: str = "medium",
    ) -> dict[str, object]:
        return self.generate_json_result(
            schema_name=schema_name,
            schema=schema,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        ).data

    def generate_json_result(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        instructions: str,
        prompt: str,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        text_verbosity: str = "medium",
    ) -> ResponsesJsonResult:
        payload = {
            "model": self.config.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
                "verbosity": text_verbosity,
            },
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        response = self.http_client.post_json(
            self.config.responses_url,
            payload=payload,
            headers=self.config.headers(),
        )
        error = response.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or json.dumps(error)
            else:
                message = str(error)
            raise LlmApiError(message)
        return ResponsesJsonResult(
            data=json.loads(self._extract_output_text(response)),
            raw_response=response,
        )

    def _extract_output_text(self, payload: dict[str, object]) -> str:
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])

        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return str(content["text"])
        raise LlmApiError("Responses API reply did not include output_text.")


def extract_usage(payload: dict[str, object]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


@dataclass(slots=True)
class LlmResearchMemory(ResearchMemory):
    client: ResponsesApiClient
    max_sources: int = 6

    def build_context(
        self,
        topic: ResearchTopic,
        sources: list[ResearchSource],
    ) -> ResearchContext:
        if not sources:
            return ResearchContext(topic=topic.name, sources=[], claims=[], open_questions=[])

        selected_sources = sources[: self.max_sources]
        payload = self.client.generate_json(
            schema_name="research_context",
            schema=_RESEARCH_CONTEXT_SCHEMA,
            instructions=(
                "You are building a structured research context for an ML/LLM research engine. "
                "Only use the supplied sources. Prefer explicit uncertainty over invention."
            ),
            prompt=json.dumps(
                {
                    "topic": topic.name,
                    "objective": topic.objective,
                    "constraints": topic.constraints,
                    "sources": [
                        {
                            "identifier": source.identifier,
                            "kind": source.kind,
                            "title": source.title,
                            "locator": source.locator,
                            "excerpt": source.excerpt,
                        }
                        for source in selected_sources
                    ],
                },
                indent=2,
            ),
        )
        claims = _sanitize_claims(payload.get("claims", []), selected_sources)
        return ResearchContext(
            topic=topic.name,
            sources=selected_sources,
            claims=claims,
            open_questions=_sanitize_string_list(payload.get("open_questions", []), limit=5),
            contradictions=_sanitize_contradictions(
                payload.get("contradictions", []),
                claims=claims,
            ),
            source_assessments=_sanitize_source_assessments(
                payload.get("source_assessments", []),
                selected_sources,
            ),
        )


@dataclass(slots=True)
class LlmHypothesisPlanner(HypothesisPlanner):
    client: ResponsesApiClient
    max_hypotheses: int = 3

    def plan(
        self,
        topic: ResearchTopic,
        context: ResearchContext,
    ) -> list[Hypothesis]:
        if not context.claims:
            return []

        payload = self.client.generate_json(
            schema_name="research_hypotheses",
            schema=_HYPOTHESIS_SCHEMA,
            instructions=(
                "You are generating bounded, testable ML/LLM research hypotheses. "
                "Do not claim experiments have run. Keep ideas concrete and benchmarkable. "
                "Return at most 3 hypotheses and use exact supporting-claim titles when citing claims."
            ),
            prompt=json.dumps(
                {
                    "topic": topic.name,
                    "objective": topic.objective,
                    "constraints": topic.constraints,
                    "claims": [
                        {
                            "title": claim.title,
                            "source": claim.source,
                            "summary": claim.summary,
                            "evidence_refs": claim.evidence_refs,
                        }
                        for claim in context.claims
                    ],
                    "source_assessments": [
                        {
                            "source_id": assessment.source_id,
                            "confidence": assessment.confidence,
                            "rationale": assessment.rationale,
                        }
                        for assessment in context.source_assessments
                    ],
                    "open_questions": context.open_questions,
                    "contradictions": [
                        {
                            "title": contradiction.title,
                            "summary": contradiction.summary,
                            "claim_titles": contradiction.claim_titles,
                            "severity": contradiction.severity,
                        }
                        for contradiction in context.contradictions
                    ],
                },
                indent=2,
            ),
        )
        hypotheses: list[Hypothesis] = []
        available_claims = {claim.title for claim in context.claims}
        for item in payload.get("hypotheses", []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title"))
            rationale = _clean_text(item.get("rationale"))
            novelty_reason = _clean_text(item.get("novelty_reason"))
            expected_signal = _clean_text(item.get("expected_signal")) or topic.objective
            if not title or not rationale or not novelty_reason:
                continue
            supporting_claims = [
                claim
                for claim in _sanitize_string_list(item.get("supporting_claims", []), limit=4)
                if claim in available_claims
            ]
            hypotheses.append(
                Hypothesis(
                    title=title,
                    rationale=rationale,
                    novelty_reason=novelty_reason,
                    expected_signal=expected_signal,
                    supporting_claims=supporting_claims,
                )
            )
        return _rank_hypotheses(hypotheses, context)[: self.max_hypotheses]


def _sanitize_claims(
    payload: object,
    sources: list[ResearchSource],
) -> list[Claim]:
    if not isinstance(payload, list):
        return []

    valid_sources = {source.identifier for source in sources}
    fallback_source = sources[0].identifier
    claims: list[Claim] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        summary = _clean_text(item.get("summary"))
        source = _clean_text(item.get("source"))
        evidence_refs = [
            ref
            for ref in _sanitize_string_list(item.get("evidence_refs", []), limit=4)
            if ref in valid_sources
        ]
        if not title or not summary:
            continue
        if source not in valid_sources:
            source = evidence_refs[0] if evidence_refs else fallback_source
        if not evidence_refs:
            evidence_refs = [source]
        claims.append(
            Claim(
                title=title,
                source=source,
                summary=summary,
                evidence_refs=evidence_refs,
            )
        )
    return claims[:5]


def _sanitize_source_assessments(
    payload: object,
    sources: list[ResearchSource],
) -> list[SourceAssessment]:
    if not isinstance(payload, list):
        return []
    valid_sources = {source.identifier for source in sources}
    assessments: list[SourceAssessment] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        source_id = _clean_text(item.get("source_id"))
        rationale = _clean_text(item.get("rationale"))
        confidence = _sanitize_level(item.get("confidence"), allowed={"low", "medium", "high"})
        if source_id not in valid_sources or not rationale:
            continue
        assessments.append(
            SourceAssessment(
                source_id=source_id,
                confidence=confidence,
                rationale=rationale,
            )
        )
    return assessments[: len(sources)]


def _sanitize_contradictions(
    payload: object,
    *,
    claims: list[Claim],
) -> list[Contradiction]:
    if not isinstance(payload, list):
        return []
    valid_claims = {claim.title for claim in claims}
    contradictions: list[Contradiction] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        summary = _clean_text(item.get("summary"))
        severity = _sanitize_level(item.get("severity"), allowed={"low", "medium", "high"})
        claim_titles = [
            claim_title
            for claim_title in _sanitize_string_list(item.get("claim_titles", []), limit=4)
            if claim_title in valid_claims
        ]
        if not title or not summary:
            continue
        contradictions.append(
            Contradiction(
                title=title,
                summary=summary,
                claim_titles=claim_titles,
                severity=severity,
            )
        )
    return contradictions[:5]


def _sanitize_string_list(payload: object, *, limit: int) -> list[str]:
    if not isinstance(payload, list):
        return []
    items = []
    for item in payload:
        text = _clean_text(item)
        if text:
            items.append(text)
    return items[:limit]


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _sanitize_level(value: object, *, allowed: set[str]) -> str:
    if not isinstance(value, str):
        return "medium"
    text = value.strip().lower()
    return text if text in allowed else "medium"


def _rank_hypotheses(
    hypotheses: list[Hypothesis],
    context: ResearchContext,
) -> list[Hypothesis]:
    claim_to_source = {claim.title: claim.source for claim in context.claims}
    source_confidence = {
        assessment.source_id: _confidence_weight(assessment.confidence)
        for assessment in context.source_assessments
    }
    contradiction_map = {}
    for contradiction in context.contradictions:
        for claim_title in contradiction.claim_titles:
            contradiction_map[claim_title] = max(
                contradiction_map.get(claim_title, 0.0),
                _severity_weight(contradiction.severity),
            )

    ranked: list[Hypothesis] = []
    for hypothesis in hypotheses:
        claim_scores = []
        contradiction_bonus = 0.0
        for claim_title in hypothesis.supporting_claims:
            source_id = claim_to_source.get(claim_title)
            if source_id is not None:
                claim_scores.append(source_confidence.get(source_id, 0.4))
            contradiction_bonus += contradiction_map.get(claim_title, 0.0)
        support_score = sum(claim_scores)
        coverage_bonus = 0.35 * len(hypothesis.supporting_claims)
        score = round(support_score + contradiction_bonus + coverage_bonus, 3)
        rationale_bits = []
        rationale_bits.append(f"{len(hypothesis.supporting_claims)} supporting claims")
        if claim_scores:
            rationale_bits.append(f"source confidence {support_score:.2f}")
        if contradiction_bonus:
            rationale_bits.append(f"contradiction bonus {contradiction_bonus:.2f}")
        if not claim_scores:
            rationale_bits.append("weak evidence support")
        ranked.append(
            Hypothesis(
                title=hypothesis.title,
                rationale=hypothesis.rationale,
                novelty_reason=hypothesis.novelty_reason,
                expected_signal=hypothesis.expected_signal,
                supporting_claims=hypothesis.supporting_claims,
                priority_score=score,
                ranking_rationale=", ".join(rationale_bits),
            )
        )
    return sorted(
        ranked,
        key=lambda item: (-item.priority_score, item.title),
    )


def _confidence_weight(confidence: str) -> float:
    return {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
    }.get(confidence, 0.4)


def _severity_weight(severity: str) -> float:
    return {
        "low": 0.1,
        "medium": 0.25,
        "high": 0.4,
    }.get(severity, 0.2)


_RESEARCH_CONTEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "source": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["title", "source", "summary", "evidence_refs"],
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "claim_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "severity": {"type": "string"},
                },
                "required": ["title", "summary", "claim_titles", "severity"],
            },
        },
        "source_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_id": {"type": "string"},
                    "confidence": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["source_id", "confidence", "rationale"],
            },
        },
    },
    "required": ["claims", "open_questions", "contradictions", "source_assessments"],
}


_HYPOTHESIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "novelty_reason": {"type": "string"},
                    "expected_signal": {"type": "string"},
                    "supporting_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "rationale",
                    "novelty_reason",
                    "expected_signal",
                    "supporting_claims",
                ],
            },
        }
    },
    "required": ["hypotheses"],
}
