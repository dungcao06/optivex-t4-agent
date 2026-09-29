"""Check emitted prompt compatibility, not model reasoning or forecast quality."""
import json

from test_http_runtime import EXAMPLE, assert_answer, assert_request_contract, run_agent


def test_quantity_period_check_reaches_model_with_original_inputs(tmp_path):
    task = json.loads((EXAMPLE / "task.json").read_text())
    answer, requests = run_agent(tmp_path, EXAMPLE)

    assert len(requests) == 1
    assert_request_contract(requests)
    assert_answer(answer, task, EXAMPLE)
    prompt = requests[0]["payload"]["messages"][-1]["content"]
    assert "TASK: " + task["prompt"] in prompt
    assert "CUTOFF DATE: " + task["cutoff_date"] in prompt
    assert "RESOLUTION DATE: " + task["resolution_date"] in prompt
    assert "TASK FAMILY: " + task["family"] in prompt
    target = prompt.split("FULL TARGET SPECIFICATION: ", 1)[1].split("\n", 1)[0]
    assert json.loads(target) == task["target"]
    for key, value in task["entities"][0].items():
        if key != "corpus_ref":  # The upstream prompt uses this only for retrieval.
            assert f"  {key}: {value}\n" in prompt
    for path in (EXAMPLE / "corpus").glob("*.json"):
        document = json.loads(path.read_text())
        assert f"doc_id={document['doc_id']} (doc_date={document['doc_date']})" in prompt
        assert document["text"] in prompt
    assert set(answer["entity_predictions"][0]) == {
        "entity_id", "label", "point_forecast", "interval", "claims"
    }

    # This catches a missing/truncated check at the actual model-request boundary.
    emitted = " ".join(prompt.split())
    assert "distinguish each evidence observation period and publication date from the requested forecast period" in emitted
    assert "point_forecast and interval bounds describe the requested quantity, units, and denominator" in emitted
    assert "For growth, change, or ratios, use only compatible supplied quantities and the required comparison period" in emitted
    assert "do not substitute a historical level for a future change" in emitted
    assert "Keep these checks internal and return the existing JSON shape only." in emitted
