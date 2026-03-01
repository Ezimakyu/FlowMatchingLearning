from argparse import Namespace

from backend.tools.preflight import check_openai_api_key, collect_checks


def test_check_openai_api_key_ok(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    result = check_openai_api_key()
    assert result.ok is True


def test_check_openai_api_key_typo_hint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_API_KEY", "sk-misnamed")
    result = check_openai_api_key()
    assert result.ok is False
    assert "did you mean OPENAI_API_KEY" in result.details


def test_collect_checks_phase_b_includes_openai_env_check(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    args = Namespace(phase="phase_b", actian_addr="localhost:50051", skip_actian=True)
    checks = collect_checks(args)
    names = [item.name for item in checks]
    assert "import_openai" in names
    assert "env_OPENAI_API_KEY" in names
