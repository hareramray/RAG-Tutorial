"""Behavior checks for the reusable pipeline, with no paid calls or downloads."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import rag_workshop.core as core
from rag_workshop import (
    Document, TfidfIndex, DenseIndex, chunk_documents, load_documents,
    context_text, retrieval_metrics, reciprocal_rank_fusion, rag_answer, ask_claude,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RAG_OFFLINE", "1")


def test_chunk_coverage_and_overlap():
    words = [f"word{i}" for i in range(31)]
    chunks = chunk_documents([Document("a", " ".join(words), {"visibility": "internal"})], 10, 3)
    assert [c.metadata["word_start"] for c in chunks] == [0, 7, 14, 21]
    assert chunks[-1].metadata["word_end"] == 31
    assert set(" ".join(c.text for c in chunks).split()) == set(words)
    assert all(c.metadata["source_id"] == "a" for c in chunks)
    assert all(c.metadata["visibility"] == "internal" for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1), (2, 3)])
def test_bad_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([Document("a", "test")], size, overlap)


def test_chunk_ids_change_on_content_edit():
    old = chunk_documents([Document("a", "old policy")])[0]
    new = chunk_documents([Document("a", "new policy")])[0]
    assert old.id != new.id
    assert old.metadata["source_id"] == new.metadata["source_id"]


def test_filter_applies_before_top_k():
    docs = [Document("secret", "refund refund exception", {"visibility": "restricted"}),
            Document("public", "refund standard policy", {"visibility": "public"})]
    for index in [TfidfIndex(docs), DenseIndex(docs)]:
        hits = index.search("refund exception", k=1, filters={"visibility": "public"})
        assert [x["id"] for x in hits] == ["public"]
        assert index.search("refund", filters={"visibility": "absent"}) == []


def test_dense_offline_is_explicit_lexical():
    index = DenseIndex(load_documents())
    assert "OFFLINE" in index.backend and "TF-IDF" in index.backend
    assert index.embedder is None
    np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), 1, atol=1e-5)


def test_rrf_deduplicates_and_combines_ranks():
    a = {"id": "a", "text": "a", "metadata": {}, "score": 999}
    b = {"id": "b", "text": "b", "metadata": {}, "score": 0.001}
    result = reciprocal_rank_fusion([a, b], [b, a], k=2)
    assert {h["id"] for h in result} == {"a", "b"}
    assert result[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([a, a], k=2)[0]["score"] == pytest.approx(1 / 61)


def test_context_has_ids_and_hard_character_budget():
    hits = TfidfIndex(chunk_documents(load_documents())).search("refund", 3)
    assert f"[{hits[0]['id']}]" in context_text(hits)
    assert len(context_text(hits, max_chars=700)) <= 700
    assert context_text(hits, max_chars=1) == ""


def test_no_evidence_abstains_without_network(monkeypatch):
    monkeypatch.setenv("RAG_OFFLINE", "0")
    monkeypatch.setattr(core, "get_client", lambda: pytest.fail("No network should be needed"))
    answer = rag_answer("unknown", [])
    assert answer["stop_reason"] == "no_evidence"
    assert answer["output_tokens"] == 0


def test_offline_claude_never_returns_fake_model_output(monkeypatch):
    monkeypatch.setattr(core, "get_client", lambda: pytest.fail("Offline must not use API"))
    answer = ask_claude("hello")
    assert answer["offline"] is True
    assert answer["text"] == "[OFFLINE] Claude generation skipped."


def test_metrics_deduplicate_sources_and_use_cutoff():
    result = retrieval_metrics(["a", "a", "b", "c"], ["b", "c"], k=2)
    assert result == {"precision_at_k": 0.5, "recall_at_k": 0.5, "reciprocal_rank": 0.5}
    assert retrieval_metrics(["a"], ["a"], k=3)["precision_at_k"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], [], 3)


def test_fixture_eval_ids_exist_and_holdout_is_disjoint():
    ids = {d.id for d in load_documents()}
    questions = json.loads((core.ROOT / "data/eval_questions.json").read_text())
    assert {q["split"] for q in questions} == {"dev", "test"}
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert set(q["relevant_ids"]) <= ids
        assert q["answerable"] == bool(q["relevant_ids"])


def test_custom_document_sources_do_not_collide(tmp_path):
    for directory in [tmp_path / "one", tmp_path / "two"]:
        directory.mkdir()
        (directory / "policy.md").write_text("This is a local policy.")
    docs = load_documents(tmp_path)
    assert len(docs) == 2 and docs[0].id != docs[1].id


def test_cache_invalidates_when_prompt_or_model_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_OFFLINE", "0")
    monkeypatch.setenv("CLAUDE_KEY", "test-not-a-real-key")
    monkeypatch.setenv("RAG_CACHE", "1")
    monkeypatch.setattr(core, "ROOT", tmp_path)
    calls = []

    class FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        @property
        def messages(self):
            return self
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="fixture answer")],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=2), stop_reason="end_turn")

    monkeypatch.setattr(core, "get_client", FakeClient)
    first = ask_claude("first question")
    repeat = ask_claude("first question")
    ask_claude("different question")
    monkeypatch.setenv("CLAUDE_MODEL", "different-model")
    ask_claude("first question")
    assert not first["cached"] and repeat["cached"]
    assert len(calls) == 3
    assert "test-not-a-real-key" not in repr(core.settings())
    assert all("test-not-a-real-key" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_workspace_header_is_optional_and_explicit(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("CLAUDE_WORKSPACE_ID", raising=False)
    assert core.anthropic_headers() == {}
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_example")
    assert core.anthropic_headers() == {"anthropic-workspace-id": "wrkspc_example"}
