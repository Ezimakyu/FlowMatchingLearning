# Testing Guide

All tests are centralized under `tests/` and use `pytest`.

## Install test dependencies

```bash
pip install -e ".[dev]"
```

## Run all tests

```bash
pytest
```

## Run only unit tests

```bash
pytest tests/unit
```

## Notes

- Unit tests avoid calling external services (Modal, Actian, OpenAI).
- External integrations should be covered with targeted integration tests as the pipeline grows.

## Optional preflight checks

Use the project preflight helper before running pipeline scripts:

```bash
python backend/tools/preflight.py --phase all
```
