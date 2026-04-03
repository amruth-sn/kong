"""Endpoint probing for LLM providers.

Validates connectivity and authentication before committing
to expensive operations like Ghidra startup.
"""

from __future__ import annotations

import logging

import anthropic
import openai

from kong.config import LLMConfig, LLMProvider
from kong.llm.codex_client import DEFAULT_MODEL as DEFAULT_CODEX_MODEL
from kong.llm.codex_client import CodexClient, CodexClientError
from kong.llm.openai_auth import resolve_openai_credential

logger = logging.getLogger(__name__)

_PROBE_DUMMY_KEY = "not-needed"


def probe_endpoint(config: LLMConfig) -> bool:
    if config.provider is LLMProvider.CUSTOM:
        return _probe_custom(config)
    if config.provider is LLMProvider.CODEX:
        return _probe_codex(config)
    if config.provider is LLMProvider.OPENAI:
        return _probe_openai(config)
    return _probe_anthropic(config)


def _probe_custom(config: LLMConfig) -> bool:
    api_key = config.api_key if config.api_key else _PROBE_DUMMY_KEY
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("Custom endpoint rejected API key")
        return False
    except openai.APIConnectionError:
        logger.warning("Could not connect to %s", config.base_url)
        return False
    except openai.APIError as e:
        logger.warning("Custom endpoint error: %s", e.message)
        return False
    except Exception as e:
        logger.warning("Could not validate %s: %s", config.base_url, e)
        return False


def _probe_openai(config: LLMConfig) -> bool:
    credential = resolve_openai_credential()
    api_key = config.api_key
    if api_key is None and credential is not None:
        api_key = credential.token

    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.PermissionDeniedError as e:
        message = str(e)
        if "api.model.read" not in message:
            logger.warning("OpenAI credential was rejected: %s", message)
            return False
        return _probe_openai_inference(config, api_key)
    except openai.AuthenticationError:
        logger.warning("OpenAI API key is invalid")
        return False
    except openai.APIError as e:
        logger.warning("OpenAI API error: %s", e.message)
        return False


def _probe_openai_inference(config: LLMConfig, api_key: str | None) -> bool:
    model = config.model or "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.base_url)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except openai.AuthenticationError:
        logger.warning("OpenAI credential is invalid for inference")
        return False
    except openai.APIError as e:
        logger.warning("OpenAI inference probe failed: %s", e.message)
        return False


def _probe_codex(config: LLMConfig) -> bool:
    try:
        client = CodexClient(
            model=config.model or DEFAULT_CODEX_MODEL,
            max_output_tokens=config.max_output_tokens,
        )
        return client.probe()
    except CodexClientError as e:
        logger.warning("Codex probe failed: %s", e)
        return False


def _probe_anthropic(config: LLMConfig) -> bool:
    try:
        client = anthropic.Anthropic(api_key=config.api_key)
        client.models.list()
        return True
    except anthropic.AuthenticationError:
        logger.warning("Anthropic API key is invalid")
        return False
    except anthropic.APIError as e:
        logger.warning("Anthropic API error: %s", e.message)
        return False
