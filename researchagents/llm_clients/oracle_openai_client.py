"""Client for Oracle-hosted ChatGPT deployments.

Adapted from the TradingAgents Oracle client: the gateway speaks the Azure
OpenAI API and authenticates with a bearer token obtained via an OAuth2
password grant against IDCS.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

DEFAULT_API_VERSION = "2024-09-01-preview"
DEFAULT_BASE_URL = "http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000"
DEFAULT_AUTH_URL = (
    "https://idcs-30d0b9fa41c34d5abb4733cead145c80.identity.pint.oc9qadev.com/oauth2/v1/token"
)
DEFAULT_AUTH_SCOPE = (
    "http://LLMResourceServer1.digitalassistant.us-phoenix-1.oci.oc-test.com/management-api/v1"
)
DEFAULT_AUTH_BASIC = (
    "Basic "
    "TExNUmVzb3VyY2VTZXJ2ZXIxX0FQUElEOjU4MTQ1ZjIzLWRhMGEtNDhiNC04MjExLWU4NWMwNzdjYjAzMQ=="
)

MODEL_NAME_TO_DEPLOYMENT_NAME = {
    "gpt-4": "oracle-gpt-4",
    "gpt-4-turbo": "oracle-gpt-4-turbo",
    "gpt-4o": "oracle-gpt-4o-2024-05-13",
    "gpt-4-32k": "oracle-gpt4-32k-dev",
    "gpt-4o-mini": "oracle-gpt-4o-mini-2024-07-18-dev",
    "gpt-o1": "oracle-o1-preview-2024-09-12-dev",
    "gpt-o1-mini": "oracle-o1-mini-2024-09-12-dev",
    "gpt-4.1": "oracle-gpt-4.1-2025-04-14-dev",
    "gpt-4.1-mini": "oracle-gpt-4.1-mini-2025-04-14-dev",
    "gpt-4.1-nano": "oracle-gpt-4.1-nano-2025-04-14-dev",
    "caa-gpt-5": "caa-gpt-5-2025-08-07-dev",
    "caa-gpt-5-mini": "caa-gpt-5-mini-2025-08-dev",
    "caa-gpt-5.1": "caa-gpt-5.1-2025-11-13-dev",
    "caa-gpt-5.2": "gpt-5.2-2025-12-11",
    "caa-gpt-5.4": "gpt-5.4-2026-03-05",
}

USERNAME_ENV_CANDIDATES = (
    "ORACLE_OPENAI_USERNAME",
    "OPEN_AI_PROXY_USER",
    "OPENAI_PROXY_USER",
    "ODA_OPENAI_USERNAME",
    "AZURE_OAI_USERNAME",
    "MODEL_API_USERNAME",
)
PASSWORD_ENV_CANDIDATES = (
    "ORACLE_OPENAI_PASSWORD",
    "OPEN_AI_PROXY_PASSWORD",
    "OPENAI_PROXY_PASSWORD",
    "ODA_OPENAI_PASSWORD",
    "AZURE_OAI_PASSWORD",
    "MODEL_API_PASSWORD",
)


class OracleBearerTokenProvider:
    """Callable token provider compatible with AzureChatOpenAI."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        *,
        auth_url: Optional[str] = None,
        auth_scope: Optional[str] = None,
        auth_basic: Optional[str] = None,
    ):
        self.username = username
        self.password = password
        self.auth_url = auth_url
        self.auth_scope = auth_scope
        self.auth_basic = auth_basic

    def __call__(self) -> str:
        return get_bearer_token(
            username=self.username,
            password=self.password,
            auth_url=self.auth_url,
            auth_scope=self.auth_scope,
            auth_basic=self.auth_basic,
        )


def get_bearer_token(
    username: Optional[str],
    password: Optional[str],
    *,
    auth_url: Optional[str] = None,
    auth_scope: Optional[str] = None,
    auth_basic: Optional[str] = None,
) -> str:
    resolved_username = _resolve_credential(value=username, env_names=USERNAME_ENV_CANDIDATES)
    resolved_password = _resolve_credential(value=password, env_names=PASSWORD_ENV_CANDIDATES)
    resolved_auth_url = _resolve_text(
        value=auth_url,
        env_names=("ORACLE_OPENAI_AUTH_URL", "OPENAI_AUTH_URL"),
        default=DEFAULT_AUTH_URL,
    )
    resolved_auth_scope = _resolve_text(
        value=auth_scope,
        env_names=("ORACLE_OPENAI_AUTH_SCOPE", "OPENAI_AUTH_SCOPE"),
        default=DEFAULT_AUTH_SCOPE,
    )
    resolved_auth_basic = _resolve_text(
        value=auth_basic,
        env_names=("ORACLE_OPENAI_AUTH_BASIC", "OPENAI_AUTH_BASIC"),
        default=DEFAULT_AUTH_BASIC,
    )

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": resolved_auth_basic,
    }
    data = {
        "scope": resolved_auth_scope,
        "grant_type": "password",
        "username": resolved_username,
        "password": resolved_password,
    }

    try:
        response = requests.post(
            resolved_auth_url,
            headers=headers,
            data=data,
            proxies={"http": "", "https": ""},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Oracle OpenAI auth request failed: {exc}") from exc

    if response.status_code != 200:
        try:
            error_payload: Dict[str, Any] | str = response.json()
        except ValueError:
            error_payload = response.text
        raise RuntimeError(
            f"Oracle OpenAI auth response {response.status_code}: {error_payload}"
        )

    try:
        return str(response.json()["access_token"])
    except Exception as exc:
        raise ValueError("Oracle OpenAI account authentication failed.") from exc


def create_oracle_llm(model: str, base_url: Optional[str] = None, **kwargs):
    from langchain_openai import AzureChatOpenAI

    resolved_base_url = _resolve_text(
        value=base_url,
        env_names=("ORACLE_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        default=DEFAULT_BASE_URL,
    )

    class OracleAzureChatOpenAI(AzureChatOpenAI):
        """AzureChatOpenAI whose structured output defaults to function calling."""

        def with_structured_output(self, schema, *, method=None, **structured_kwargs):
            if method is None:
                method = "function_calling"
            return super().with_structured_output(schema, method=method, **structured_kwargs)

    llm_kwargs = {
        "model": model,
        "azure_deployment": MODEL_NAME_TO_DEPLOYMENT_NAME.get(model, model),
        "azure_endpoint": resolved_base_url,
        "openai_api_version": _resolve_text(
            value=kwargs.pop("api_version", None),
            env_names=("ORACLE_OPENAI_API_VERSION", "OPENAI_API_VERSION"),
            default=DEFAULT_API_VERSION,
        ),
        "azure_ad_token_provider": kwargs.pop("azure_ad_token_provider", None)
        or OracleBearerTokenProvider(
            username=kwargs.pop("username", None),
            password=kwargs.pop("password", None),
            auth_url=kwargs.pop("auth_url", None),
            auth_scope=kwargs.pop("auth_scope", None),
            auth_basic=kwargs.pop("auth_basic", None),
        ),
        **kwargs,
    }
    return OracleAzureChatOpenAI(**llm_kwargs)


def _resolve_text(*, value: Optional[str], env_names: tuple, default: str) -> str:
    if value is not None and value.strip():
        return value.strip()
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return default


def _resolve_credential(*, value: Optional[str], env_names: tuple) -> str:
    if value is not None and value.strip():
        return value.strip()
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    raise ValueError(f"Missing credential; checked env vars: {', '.join(env_names)}")
