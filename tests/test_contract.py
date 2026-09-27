from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_enforces_evidence_and_embargo() -> None:
    source = (ROOT / "analyze.py").read_text(encoding="utf-8")
    assert "Do not use memory" in source
    assert "after the cutoff date" in source
    assert "copied verbatim" in source
    assert "must support the label/value/ranking" in source


def test_container_declares_interface_and_verb() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'qfbench2.interface_version="2.0"' in dockerfile
    assert 'ENTRYPOINT ["python", "/app/analyze.py"]' in dockerfile


def test_submission_uses_current_house_category() -> None:
    descriptor = (ROOT / "submission.example.json").read_text(encoding="utf-8")
    assert '"category": "api"' in descriptor
    assert "byo-small" not in descriptor
    assert "byo-large" not in descriptor


def test_wrapper_omits_null_optional_label() -> None:
    source = (ROOT / "analyze.py").read_text(encoding="utf-8")
    assert 'result.prediction.get("label") is None' in source
    assert 'result.prediction.pop("label", None)' in source
