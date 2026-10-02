"""Keep an entity's selected anchors without padding prompts with peer tables."""
import json

import pytest

from test_retrieval import retrieval_module, write_doc


def evidence(tmp_path, labels, *, entity="A", target="revision"):
    texts = {
        "own": "Revision history: July 100, August 103, September 102.",
        "peer": "Revision history: July 9000, August 9100, September 9200.",
        "shared": "Revision methodology: initial estimates can change by 2 units.",
        "unknown": "Revision schedule: releases occur every 30 days.",
    }
    for name, text in texts.items():
        write_doc(tmp_path, name, text)
    if labels is not None:
        (tmp_path / "manifest.json").write_text(json.dumps({"files": [
            {"path": f"corpus/{name}.json", **label}
            for name, label in labels.items()
        ]}))
    mod = retrieval_module()
    corpus = mod.build_index(tmp_path)
    selected = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": target}}, {"entity_id": entity})
    return selected, corpus


def test_explicit_peer_table_removed_without_losing_own_shared_or_unknown(tmp_path):
    """A high-overlap peer series must not accompany the selected own anchor."""
    selected, corpus = evidence(tmp_path, {
        "own": {"entity_ids": ["A", "C"]},
        "peer": {"entity_ids": ["B"]},
        "shared": {"shared": True, "entity_ids": ["B"]},
    })
    assert {c.doc_id for c in selected} == {"own", "shared", "unknown"}
    for c in selected:
        assert c.text == corpus.doc_texts[c.doc_id][c.span_start:c.span_end]


@pytest.mark.parametrize("labels", [None, {}, {
    "own": {"entity_ids": ["B"]}, "peer": {"entity_ids": ["C"]},
    "shared": {"shared": True},
}])
def test_without_selected_own_anchor_existing_context_remains_usable(tmp_path, labels):
    """Incomplete metadata or missing own evidence must not empty retrieval."""
    selected, _ = evidence(tmp_path, labels)
    assert {c.doc_id for c in selected} == {"own", "peer", "shared", "unknown"}


def test_unlabelled_document_is_not_assumed_to_belong_to_a_peer(tmp_path):
    selected, _ = evidence(tmp_path, {
        "own": {"entity_ids": ["A"]}, "peer": {}, "shared": {"shared": True},
    })
    assert {c.doc_id for c in selected} == {"own", "peer", "shared", "unknown"}


def test_peer_removal_does_not_backfill_with_weaker_shared_prose(tmp_path):
    """Filtering after selection preserves own spans and avoids shared filler."""
    mod = retrieval_module()
    for name, text in {
        "own": "Revision revision anchor 100.",
        "peer": "Revision revision anchor 9000.",
        "filler": "Revision methodology and administrative publication procedures.",
    }.items():
        write_doc(tmp_path, name, text)
    task = {"target": {"name": "revision anchor"}}
    entity = {"entity_id": "A"}
    before = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(task, entity, top_k=2)
    assert {c.doc_id for c in before} == {"own", "peer"}
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [
        {"path": "corpus/own.json", "entity_ids": ["A"]},
        {"path": "corpus/peer.json", "entity_ids": ["B"]},
        {"path": "corpus/filler.json", "shared": True},
    ]}))
    after = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(task, entity, top_k=2)
    assert after == [c for c in before if c.doc_id == "own"]
