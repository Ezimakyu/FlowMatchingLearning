from pathlib import Path

from backend.app.config import load_env_file


def test_load_env_file_parses_assignments(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "OPENAI_API_KEY=sk-test",
                "export OPENAI_REASONING_MODEL=gpt-4.1-mini",
                "ACTIAN_VECTORAI_ADDR='localhost:50051'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_MODEL", raising=False)
    monkeypatch.delenv("ACTIAN_VECTORAI_ADDR", raising=False)

    loaded = load_env_file(env_file)
    assert loaded["OPENAI_API_KEY"] == "sk-test"
    assert loaded["OPENAI_REASONING_MODEL"] == "gpt-4.1-mini"
    assert loaded["ACTIAN_VECTORAI_ADDR"] == "localhost:50051"


def test_load_env_file_does_not_override_by_default(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    loaded = load_env_file(env_file, override=False)
    assert "OPENAI_API_KEY" not in loaded


def test_load_env_file_overrides_when_requested(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    loaded = load_env_file(env_file, override=True)
    assert loaded["OPENAI_API_KEY"] == "sk-from-file"
