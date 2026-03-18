"""Endpoint probing for LLM providers.

Validates connectivity and authentication before committing
to expensive operations like Ghidra startup.
"""

from __future__ import annotations

import logging

import anthropic
import httpx
import openai

from kong.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


def probe_endpoint(config: LLMConfig) -> bool:
    if config.provider is LLMProvider.CUSTOM:
        return _probe_custom(config)
    if config.provider is LLMProvider.OPENAI:
        return _probe_openai(config)
    return _probe_anthropic(config)


def _probe_custom(config: LLMConfig) -> bool:
    url = f"{config.base_url}/models"
    try:
        response = httpx.get(url, timeout=10.0)
        if response.status_code == 200:
            return True
        logger.warning("Custom endpoint returned status %d", response.status_code)
        return False
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("Could not connect to %s", config.base_url)
        return False
    except httpx.HTTPError as e:
        logger.warning("HTTP error probing %s: %s", config.base_url, e)
        return False


def _probe_openai(config: LLMConfig) -> bool:
    try:
        client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
        client.models.list()
        return True
    except openai.AuthenticationError:
        logger.warning("OpenAI API key is invalid")
        return False
    except openai.APIError as e:
        logger.warning("OpenAI API error: %s", e)
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
        logger.warning("Anthropic API error: %s", e)
        return False
