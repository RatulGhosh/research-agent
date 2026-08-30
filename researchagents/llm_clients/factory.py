from typing import Optional

# Providers that use the OpenAI-compatible chat completions API
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek", "qwen", "glm", "ollama", "openrouter",
)


def create_llm(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
):
    """Create a LangChain chat model for the specified provider.

    Provider modules are imported lazily so that importing this factory does
    not pull in heavy LLM SDKs or fail when their API keys are absent.
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from langchain_openai import ChatOpenAI

        llm_kwargs = {"model": model, **kwargs}
        if base_url:
            llm_kwargs["base_url"] = base_url
        return ChatOpenAI(**llm_kwargs)

    if provider_lower == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm_kwargs = {"model": model, **kwargs}
        if base_url:
            llm_kwargs["base_url"] = base_url
        return ChatAnthropic(**llm_kwargs)

    if provider_lower == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)

    if provider_lower == "oracle":
        from .oracle_openai_client import create_oracle_llm

        return create_oracle_llm(model=model, base_url=base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
