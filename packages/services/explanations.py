from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from packages.llm.openrouter_client import OpenRouterClient, default_openrouter_model
from packages.llm.prompts import PROMPT_VERSION, build_recommendation_explanation_messages


@dataclass(frozen=True)
class ExplanationResult:
    provider: str
    model: str
    prompt_version: str
    content_type: str
    content: str
    request: dict[str, Any]
    response: dict[str, Any] | None


def generate_recommendation_explanation(*, run_payload: dict[str, Any]) -> ExplanationResult:
    client = OpenRouterClient.from_env()
    model = default_openrouter_model()

    messages = build_recommendation_explanation_messages(run=run_payload)
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }

    response = client.chat_completions(request_payload)

    content = ""
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        content = json.dumps(response, indent=2)

    return ExplanationResult(
        provider="openrouter",
        model=model,
        prompt_version=PROMPT_VERSION,
        content_type="text/markdown",
        content=content,
        request=request_payload,
        response=response,
    )

