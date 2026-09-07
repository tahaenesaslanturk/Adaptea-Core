from pathlib import Path

import pytest

from adaptea.config import Config, load_config
from adaptea.inference import (
    InferenceBackend,
    InferenceBackendError,
    create_inference_backend,
    inference_connection,
)
from adaptea.llamacpp.client import LlamaCppClient
from adaptea.lmstudio.client import LMStudioClient, LMStudioError
from adaptea.ollama.client import OllamaClient
from adaptea.planner.opencode import opencode_config
from adaptea.vllm.client import VLLMClient


@pytest.mark.asyncio
async def test_legacy_config_defaults_to_lmstudio_backend(tmp_path: Path) -> None:
    (tmp_path / "adaptea.toml").write_text(
        '[lmstudio]\nbase_url = "http://legacy.test:4321/"\nmodel = "coder"\n',
        encoding="utf-8",
    )

    config = load_config(tmp_path)
    connection = inference_connection(config)

    assert config.inference.backend == "lmstudio"
    assert connection.kind == "lmstudio"
    assert connection.openai_base_url == "http://legacy.test:4321/v1"
    backend = create_inference_backend(config)
    assert isinstance(backend, LMStudioClient)
    await backend.close()


@pytest.mark.asyncio
async def test_lmstudio_backend_exposes_typed_connection_and_compatible_error() -> None:
    config = Config(
        inference={"backend": "lmstudio"},
        lmstudio={"base_url": "http://localhost:1234/", "api_token": "secret"},
    )

    connection = inference_connection(config)
    backend = create_inference_backend(config)

    assert connection.display_name == "LM Studio (Adaptea)"
    assert connection.openai_base_url == "http://localhost:1234/v1"
    assert connection.api_token == "secret"
    assert connection.api_token_environment == "LM_API_TOKEN"
    assert isinstance(backend, InferenceBackend)
    assert issubclass(LMStudioError, InferenceBackendError)
    await backend.close()


@pytest.mark.asyncio
async def test_ollama_backend_is_explicit_and_does_not_change_legacy_defaults() -> None:
    config = Config(inference={"backend": "ollama"})

    connection = inference_connection(config)
    backend = create_inference_backend(config)

    assert connection.kind == "ollama"
    assert connection.openai_base_url == "http://127.0.0.1:11434/v1"
    assert connection.fallback_api_key == "ollama"
    assert isinstance(backend, OllamaClient)
    assert backend.capabilities.safe_default_parallel_limit == 1
    assert backend.capabilities.live_pressure_metrics is False
    await backend.close()


@pytest.mark.asyncio
async def test_llamacpp_backend_connection_and_factory() -> None:
    config = Config(
        inference={"backend": "llamacpp"},
        llamacpp={"base_url": "http://127.0.0.1:8080", "api_token": "llama-token"},
    )

    connection = inference_connection(config)
    backend = create_inference_backend(config)

    assert connection.kind == "llamacpp"
    assert connection.display_name == "llama.cpp (Adaptea)"
    assert connection.openai_base_url == "http://127.0.0.1:8080/v1"
    assert connection.api_token == "llama-token"
    assert isinstance(backend, LlamaCppClient)
    assert backend.capabilities.live_pressure_metrics is True
    await backend.close()


@pytest.mark.asyncio
async def test_vllm_backend_connection_and_factory() -> None:
    config = Config(
        inference={"backend": "vllm"},
        vllm={"base_url": "http://127.0.0.1:8000", "api_token": "vllm-token"},
    )

    connection = inference_connection(config)
    backend = create_inference_backend(config)

    assert connection.kind == "vllm"
    assert connection.display_name == "vLLM (Adaptea)"
    assert connection.openai_base_url == "http://127.0.0.1:8000/v1"
    assert connection.api_token == "vllm-token"
    assert isinstance(backend, VLLMClient)
    assert backend.capabilities.live_pressure_metrics is True
    await backend.close()


def test_opencode_config_uses_selected_backend_connection_without_shape_change() -> None:
    config = Config(lmstudio={"base_url": "http://local.test:9999/", "api_token": "secret"})

    generated = opencode_config(config, "coder")
    provider = generated["provider"]["adaptea"]

    assert provider["name"] == "LM Studio (Adaptea)"
    assert provider["options"] == {
        "baseURL": "http://local.test:9999/v1",
        "apiKey": "{env:LM_API_TOKEN}",
    }


def test_opencode_config_supports_ollama_without_an_api_key_environment() -> None:
    config = Config(
        inference={"backend": "ollama"},
        ollama={"base_url": "http://ollama.test:11434"},
    )

    provider = opencode_config(config, "qwen3-coder")["provider"]["adaptea"]

    assert provider["name"] == "Ollama (Adaptea)"
    assert provider["options"] == {
        "baseURL": "http://ollama.test:11434/v1",
        "apiKey": "ollama",
    }


def test_opencode_config_supports_llamacpp_and_vllm() -> None:
    config_llama = Config(
        inference={"backend": "llamacpp"},
        llamacpp={"base_url": "http://127.0.0.1:8080"},
    )
    provider_llama = opencode_config(config_llama, "qwen2.5-coder")["provider"]["adaptea"]
    assert provider_llama["name"] == "llama.cpp (Adaptea)"
    assert provider_llama["options"] == {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "llamacpp",
    }

    config_vllm = Config(
        inference={"backend": "vllm"},
        vllm={"base_url": "http://127.0.0.1:8000"},
    )
    provider_vllm = opencode_config(config_vllm, "qwen2.5-coder")["provider"]["adaptea"]
    assert provider_vllm["name"] == "vLLM (Adaptea)"
    assert provider_vllm["options"] == {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "vllm",
    }
