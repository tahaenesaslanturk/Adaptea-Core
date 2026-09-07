from __future__ import annotations

from pathlib import Path

from adaptea.config import load_config
from adaptea.setup.configuration import (
    merge_adaptea_config,
    merge_inference_model,
    merge_inference_selection,
    merge_opencode_config,
    merge_opencode_models_config,
    read_json_config,
    strip_jsonc,
)


def test_inference_model_can_be_replaced_and_cleared_without_losing_settings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adaptea.toml"
    path.write_text(
        '[lmstudio]\nbase_url = "http://local.test:1234"\nmodel = "old"\n',
        encoding="utf-8",
    )

    merge_inference_model(path, "lmstudio", "new")
    assert load_config(tmp_path).lmstudio.model == "new"

    merge_inference_model(path, "lmstudio", None)
    config = load_config(tmp_path)
    assert config.lmstudio.model is None
    assert config.lmstudio.base_url == "http://local.test:1234"


def test_jsonc_parser_preserves_strings_and_removes_comments() -> None:
    text = r"""{
      // comment
      "url": "https://example.test/a//b",
      "literal": ",}",
      "nested": {"value": 1,}, /* comment */
    }"""
    assert read_json_config_from_text(text) == {
        "url": "https://example.test/a//b",
        "literal": ",}",
        "nested": {"value": 1},
    }


def read_json_config_from_text(text: str) -> dict[str, object]:
    import json

    value = json.loads(strip_jsonc(text))
    assert isinstance(value, dict)
    return value


def test_opencode_config_merge_preserves_user_settings_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    path.write_text(
        '{"theme":"custom","provider":{"other":{"models":{"x":{}}},'
        '"lmstudio":{"custom":"keep","models":{"old":{}}}}}',
        encoding="utf-8",
    )
    written, backup = merge_opencode_config(
        tmp_path, "/usr/local/bin/opencode", "coder", "http://127.0.0.1:1234"
    )
    assert written == path
    assert backup is not None and backup.exists()
    result = read_json_config(path)
    assert result["theme"] == "custom"
    providers = result["provider"]
    assert isinstance(providers, dict)
    assert "other" in providers
    lmstudio = providers["lmstudio"]
    assert isinstance(lmstudio, dict)
    assert lmstudio["custom"] == "keep"
    assert set(lmstudio["models"]) == {"old", "coder"}
    assert lmstudio["options"] == {"baseURL": "http://127.0.0.1:1234/v1"}


def test_opencode2_uses_v2_project_schema(tmp_path: Path) -> None:
    path, backup = merge_opencode_config(
        tmp_path, r"C:\Tools\opencode2.exe", "coder", "http://127.0.0.1:1234"
    )
    assert backup is None
    result = read_json_config(path)
    providers = result["providers"]
    assert isinstance(providers, dict)
    provider = providers["lmstudio"]
    assert provider["package"] == "@opencode-ai/ai/providers/openai-compatible"
    assert provider["settings"]["baseURL"].endswith("/v1")


def test_opencode_fleet_merge_registers_every_selected_model(tmp_path: Path) -> None:
    path, _backup = merge_opencode_models_config(
        tmp_path,
        "/usr/local/bin/opencode",
        ["publisher/strong", "publisher/fast"],
        "http://127.0.0.1:1234",
    )
    providers = read_json_config(path)["provider"]
    assert isinstance(providers, dict)
    assert set(providers["lmstudio"]["models"]) == {
        "publisher/strong",
        "publisher/fast",
    }


def test_opencode_merge_uses_existing_project_jsonc(tmp_path: Path) -> None:
    path = tmp_path / "opencode.jsonc"
    path.write_text('{// user setting\n"theme":"custom",}', encoding="utf-8")
    written, backup = merge_opencode_config(
        tmp_path, "/usr/local/bin/opencode", "coder", "http://127.0.0.1:1234"
    )
    assert written == path
    assert backup is not None
    assert read_json_config(path)["theme"] == "custom"


def test_adaptea_toml_merge_preserves_unrelated_sections(tmp_path: Path) -> None:
    path = tmp_path / "adaptea.toml"
    path.write_text(
        '[custom]\nowner = "alice"\n\n[lmstudio]\nbase_url = "http://old"\nextra = 7\n',
        encoding="utf-8",
    )
    backup = merge_adaptea_config(
        path,
        base_url="http://127.0.0.1:1234",
        model="coder",
        lms_executable=r"C:\LM Studio\lms.exe",
        opencode_executable=r"C:\Tools\opencode.exe",
        max_agents=6,
    )
    assert backup is not None and backup.exists()
    text = path.read_text(encoding="utf-8")
    assert '[custom]\nowner = "alice"' in text
    assert "extra = 7" in text
    assert 'model = "coder"' in text
    assert 'executable = "C:\\\\Tools\\\\opencode.exe"' in text
    assert "max_agents = 6" in text
    assert 'default_scheduler = "adaptive"' in text


def test_ollama_setup_writes_backend_specific_configs_without_removing_lmstudio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adaptea.toml"
    path.write_text('[lmstudio]\nmodel = "legacy-model"\n', encoding="utf-8")

    merge_adaptea_config(
        path,
        base_url="http://127.0.0.1:11434",
        model="qwen3-coder",
        lms_executable="lms",
        opencode_executable="opencode",
        max_agents=8,
        backend="ollama",
        ollama_executable="/usr/local/bin/ollama",
    )
    opencode_path, _ = merge_opencode_config(
        tmp_path,
        "opencode",
        "qwen3-coder",
        "http://127.0.0.1:11434",
        backend="ollama",
    )

    config = load_config(tmp_path)
    providers = read_json_config(opencode_path)["provider"]
    assert isinstance(providers, dict)
    assert config.inference.backend == "ollama"
    assert config.ollama.model == "qwen3-coder"
    assert config.ollama.executable == "/usr/local/bin/ollama"
    assert config.lmstudio.model == "legacy-model"
    assert providers["ollama"]["options"] == {
        "baseURL": "http://127.0.0.1:11434/v1",
        "apiKey": "ollama",
    }


def test_backend_selection_merge_preserves_provider_settings(tmp_path: Path) -> None:
    path = tmp_path / "adaptea.toml"
    path.write_text('[custom]\nowner = "ada"\n', encoding="utf-8")

    backup = merge_inference_selection(path, "ollama")

    assert backup is not None and backup.exists()
    assert '[custom]\nowner = "ada"' in path.read_text(encoding="utf-8")
    assert load_config(tmp_path).inference.backend == "ollama"
