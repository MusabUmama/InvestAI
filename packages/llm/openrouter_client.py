from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str
    base_url: str = OPENROUTER_BASE_URL
    timeout_seconds: float = 60.0
    app_name: str = "InvestAI"

    @staticmethod
    def from_env() -> "OpenRouterClient":
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return OpenRouterClient(api_key=key)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends these headers for attribution; safe to send even if blank.
            "X-Title": self.app_name,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()


def default_openrouter_model() -> str:
    # Use a user-provided model if set; otherwise default to a free-ish route.
    # This can be adjusted later without touching code paths.
    return os.getenv("OPENROUTER_MODEL", "").strip() or "openrouter/free"

