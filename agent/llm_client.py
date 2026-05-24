"""
agent/llm_client.py
────────────────────
Provider-agnostic LLM client with automatic fallback chain.

Free provider priority order (best quality → fastest):
  1. Groq API          — free tier, DeepSeek-Coder-33B, ~500 tok/s
  2. Google Gemini     — free tier, 1M context, 15 RPM
  3. Ollama (local)    — fully offline, DeepSeek-Coder-7B/33B
  4. HuggingFace TGI   — free inference API
  5. OpenAI            — paid fallback (only if key is set)

Why Groq over GPT-4o for this project:
  - DeepSeek-Coder-33B-Instruct scores HIGHER than GPT-4o on HumanEval
    (79.3% vs 67.0%), EvalPlus, and LiveCodeBench for code tasks
  - Inference is 10× faster (~500 tok/s vs ~50 tok/s)
  - Free tier: 30 RPM, 14,400 RPD, 6,000 tokens/min
  - This is a QUALITY UPGRADE, not just a cost-cutting measure

Usage:
    from agent.llm_client import get_llm_client
    client = get_llm_client()   # auto-detects from environment
    patch = client.complete(system=SYSTEM_PROMPT, user=ISSUE_TEXT)
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

# Auto-load .env so scripts work without manually exporting env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Provider-agnostic LLM interface."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[str, dict]:
        """
        Generate completion.
        Returns: (text, usage_dict)
        usage_dict keys: prompt_tokens, completion_tokens, total_tokens
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""


# ── Groq client (FREE — recommended) ─────────────────────────────────────────

class GroqClient(LLMClient):
    """
    Groq Cloud API — free tier.
    Best model for code: deepseek-r1-distill-llama-70b or
    llama-3.3-70b-versatile or deepseek-coder models.

    Free limits: 30 requests/min · 14,400 requests/day · 6,000 tokens/min
    Sign up: https://console.groq.com (no credit card required)

    Set env var: GROQ_API_KEY=gsk_...
    """

    # Best free models for code generation on Groq (ranked by code quality)
    RECOMMENDED_MODELS = [
        "deepseek-r1-distill-llama-70b",    # DeepSeek R1 reasoning — best for bugs
        "llama-3.3-70b-versatile",          # Llama 3.3 70B — excellent general code
        "llama-3.1-70b-versatile",          # Llama 3.1 70B fallback
    ]

    def __init__(self, model: str = "deepseek-r1-distill-llama-70b"):
        self._model = model
        self._client = None

    @property
    def model_name(self) -> str:
        return f"groq/{self._model}"

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            except ImportError:
                raise ImportError("Install groq: pip install groq")
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 0.2) -> tuple[str, dict]:
        client = self._get_client()
        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            logger.info("Groq %s: %.1fs | %d tokens", self._model, time.monotonic() - start, usage["total_tokens"])
            return text, usage
        except Exception as e:
            logger.warning("Groq error: %s", e)
            raise


# ── Google Gemini client (FREE) ───────────────────────────────────────────────

class GeminiClient(LLMClient):
    """
    Google Gemini API via direct REST calls — no SDK needed.
    Uses httpx which is already in requirements.txt.

    gemini-2.0-flash: fast, generous free tier (15 RPM, 1M tokens/day)
    gemini-2.5-flash: newest model, same free tier

    Sign up: https://aistudio.google.com (no credit card required)
    Set env var: GEMINI_API_KEY=AIza...
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str = "gemini-2.0-flash"):
        self._model = model

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model}"

    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 0.2) -> tuple[str, dict]:
        import httpx

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")

        url = f"{self.BASE_URL}/{self._model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        start = time.monotonic()
        try:
            resp = httpx.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            text = data["candidates"][0]["content"]["parts"][0]["text"]
            meta = data.get("usageMetadata", {})
            usage = {
                "prompt_tokens":     meta.get("promptTokenCount", 0),
                "completion_tokens": meta.get("candidatesTokenCount", 0),
                "total_tokens":      meta.get("totalTokenCount", 0),
            }
            logger.info("Gemini %s: %.1fs | %d tokens", self._model, time.monotonic() - start, usage["total_tokens"])
            return text, usage
        except Exception as e:
            logger.warning("Gemini error: %s", e)
            raise


# ── Ollama client (100% local, offline) ──────────────────────────────────────

class OllamaClient(LLMClient):
    """
    Ollama — run models 100% locally, no API key, no cost, no rate limits.
    Best model for code: deepseek-coder-v2:16b or deepseek-coder:33b
    Install: https://ollama.com
    Run:     ollama pull deepseek-coder-v2:16b

    Required: Ollama server running at localhost:11434
    """

    def __init__(self, model: str = "deepseek-coder-v2:16b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 0.2) -> tuple[str, dict]:
        try:
            import requests
        except ImportError:
            raise ImportError("Install: pip install requests")

        start = time.monotonic()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        usage = {
            "prompt_tokens":     data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens":      total_tokens,
        }
        logger.info("Ollama %s: %.1fs | %d tokens", self._model, time.monotonic() - start, total_tokens)
        return text, usage


# ── HuggingFace Inference API client (fine-tuned adapter) ────────────────────

class HFInferenceClient(LLMClient):
    """
    HuggingFace Inference API — runs any public model/adapter hosted on HF Hub.
    Free for public models (rate-limited but sufficient for evaluation).

    Best for: testing fine-tuned adapters without a local GPU.
    Set env var: HF_TOKEN=hf_...
                 LLM_PROVIDER=hf
                 LLM_MODEL=SouravNath/repomind-deepseek-coder-7b-lora
    """

    HF_API_BASE = "https://api-inference.huggingface.co/models"

    def __init__(self, model: str = "SouravNath/repomind-deepseek-coder-7b-lora"):
        self._model = model

    @property
    def model_name(self) -> str:
        return f"hf/{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> tuple[str, dict]:
        import httpx

        api_key = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # Format as ChatML for DeepSeek models
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "stop": ["<|im_end|>"],
            },
        }

        start = time.monotonic()
        url = f"{self.HF_API_BASE}/{self._model}"
        try:
            with httpx.Client(timeout=180.0) as http:
                resp = http.post(url, json=payload, headers=headers)
                if resp.status_code == 503:
                    # Model is loading — wait and retry once
                    logger.warning("HF model loading, waiting 30s...")
                    time.sleep(30)
                    resp = http.post(url, json=payload, headers=headers)
                resp.raise_for_status()

            data = resp.json()
            if isinstance(data, list):
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")

            # Strip the stop token if present
            text = text.split("<|im_end|>")[0].strip()

            total = len(text.split())
            usage = {"prompt_tokens": 0, "completion_tokens": total, "total_tokens": total}
            logger.info(
                "HF Inference %s: %.1fs | ~%d tokens",
                self._model, time.monotonic() - start, total,
            )
            return text, usage
        except Exception as e:
            logger.warning("HF Inference error: %s", e)
            raise


# ── OpenAI client (paid, kept as optional fallback) ───────────────────────────

class OpenAIClient(LLMClient):
    """OpenAI client — kept as optional fallback if OPENAI_API_KEY is set."""

    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._client = None

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError("Install: pip install openai")
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 0.2) -> tuple[str, dict]:
        client = self._get_client()
        start = time.monotonic()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }
        logger.info("OpenAI %s: %.1fs | %d tokens", self._model, time.monotonic() - start, usage["total_tokens"])
        return text, usage


# ── Auto-detect factory ────────────────────────────────────────────────────────

def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """
    Auto-detect and return the best available free LLM client.

    Priority (set LLM_PROVIDER env var to override):
      groq → gemini → ollama → openai

    Args:
        provider: "groq" | "gemini" | "ollama" | "openai" | None (auto)
        model:    model name override
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "auto")

    if provider == "auto":
        # Try each free provider in priority order
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
            logger.info("Auto-selected provider: Groq (GROQ_API_KEY found)")
        elif os.environ.get("HF_TOKEN") and os.environ.get("HF_MODEL"):
            provider = "hf"
            logger.info("Auto-selected provider: HF Inference (HF_TOKEN + HF_MODEL found)")
        elif os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
            logger.info("Auto-selected provider: Gemini (GEMINI_API_KEY found)")
        elif _ollama_available():
            provider = "ollama"
            logger.info("Auto-selected provider: Ollama (local server detected)")
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            logger.info("Auto-selected provider: OpenAI (OPENAI_API_KEY found, note: paid)")
        else:
            raise EnvironmentError(
                "No LLM provider configured. Set one of:\n"
                "  GROQ_API_KEY   — free at https://console.groq.com\n"
                "  HF_TOKEN       — free at https://huggingface.co/settings/tokens\n"
                "  GEMINI_API_KEY — free at https://aistudio.google.com\n"
                "  Install Ollama — https://ollama.com (fully local, free)\n"
                "  OPENAI_API_KEY — paid"
            )

    default_hf_model = os.environ.get("HF_MODEL", "SouravNath/repomind-deepseek-coder-7b-lora")
    clients = {
        "groq":   lambda: GroqClient(model or "deepseek-r1-distill-llama-70b"),
        "hf":     lambda: HFInferenceClient(model or default_hf_model),
        "gemini": lambda: GeminiClient(model or "gemini-2.0-flash"),
        "ollama": lambda: OllamaClient(model or "deepseek-coder-v2:16b"),
        "openai": lambda: OpenAIClient(model or "gpt-4o"),
    }
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(clients)}")

    return clients[provider]()


def _ollama_available() -> bool:
    """Check if Ollama server is running locally."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False
