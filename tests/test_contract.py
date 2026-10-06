from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_enforces_evidence_and_embargo() -> None:
    source = (ROOT / "analyze.py").read_text(encoding="utf-8")
    assert "Do not use memory" in source
    assert "after the cutoff date" in source
    assert "copied verbatim" in source
    assert "Quote relevant observed facts" in source


def test_container_declares_interface_and_verb() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'qfbench2.interface_version="2.0"' in dockerfile
    assert 'ENTRYPOINT ["python", "/app/analyze.py"]' in dockerfile


def test_submission_uses_current_house_category() -> None:
    descriptor = (ROOT / "submission.example.json").read_text(encoding="utf-8")
    assert '"category": "api"' in descriptor
    assert "byo-small" not in descriptor
    assert "byo-large" not in descriptor


def test_runtime_omits_optional_label_and_rank() -> None:
    import analyze  # Establish the pinned upstream import path.
    from runtime import normalize_prediction
    from baselines.strong_rag_baseline.indexer import Chunk

    chunk = Chunk("doc", "2024-01-01", 0, 8, "Evidence")
    for family in ("regression", "ranking"):
        result = normalize_prediction(
            {"label": None, "rank": -3, "point_forecast": 1,
             "interval": {"lo": 0, "hi": 2},
             "evidence": [{"doc_id": "doc", "quote": "Evidence", "claim": "Context"}]},
            {"target": {"type": family}, "entities": [{"entity_id": "one"}]}, {"entity_id": "one"}, [chunk])
        assert "label" not in result
        assert "rank" not in result


def test_image_copies_every_local_module_the_entrypoint_imports() -> None:
    """Local tests import from the source tree; the image has only what the Dockerfile copies."""
    import ast
    import re

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = {name for line in dockerfile.splitlines() if line.startswith("COPY ") and "--from=" not in line
              for name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.py\b", line)}
    local = {path.stem for path in ROOT.glob("*.py")}
    needed, queue = set(), ["analyze"]
    while queue:
        module = queue.pop()
        if module in needed:
            continue
        needed.add(module)
        for node in ast.walk(ast.parse((ROOT / f"{module}.py").read_text(encoding="utf-8"))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            queue += [name.split(".")[0] for name in names if name.split(".")[0] in local]
    assert needed - copied == set(), f"imported but not copied into the image: {sorted(needed - copied)}"
