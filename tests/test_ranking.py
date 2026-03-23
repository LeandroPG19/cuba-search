from cuba_search.ranking import (
    bm25_score,
    bm25_rank,
    rrf_fuse,
    information_density,
    compute_confidence,
)

def test_bm25_score_basic():
    query_terms = ["apple", "banana"]
    doc_terms = ["apple", "orange", "grape"]
    doc_freq = {"apple": 1, "orange": 1, "grape": 1, "banana": 0}
    total_docs = 1
    avg_doc_len = 3.0

    score = bm25_score(query_terms, doc_terms, doc_freq, total_docs, avg_doc_len)
    assert score > 0.0

def test_bm25_score_no_match():
    query_terms = ["apple"]
    doc_terms = ["orange", "grape"]
    doc_freq = {"apple": 0, "orange": 1, "grape": 1}
    total_docs = 1
    avg_doc_len = 2.0

    score = bm25_score(query_terms, doc_terms, doc_freq, total_docs, avg_doc_len)
    assert score == 0.0

def test_bm25_score_empty_inputs():
    doc_freq = {}
    total_docs = 1
    avg_doc_len = 1.0

    assert bm25_score([], ["apple"], doc_freq, total_docs, avg_doc_len) == 0.0
    assert bm25_score(["apple"], [], doc_freq, total_docs, avg_doc_len) == 0.0

def test_bm25_score_zero_avg_len():
    # Should use max(avg_doc_len, 1.0) and not crash
    query_terms = ["apple"]
    doc_terms = ["apple"]
    doc_freq = {"apple": 1}
    total_docs = 1

    score = bm25_score(query_terms, doc_terms, doc_freq, total_docs, 0.0)
    assert score > 0.0

def test_bm25_rank_sorting():
    query = "apple"
    docs = [
        {"content": "apple apple"},
        {"content": "apple"},
        {"content": "banana"}
    ]
    ranked = bm25_rank(query, docs)
    assert len(ranked) == 3
    assert ranked[0]["bm25_score"] > ranked[1]["bm25_score"]
    assert ranked[2]["bm25_score"] == 0.0

def test_bm25_rank_empty():
    assert bm25_rank("query", []) == []

def test_bm25_rank_missing_key():
    docs = [{"other": "apple"}]
    # Default text_key is "content", should handle missing key as empty string
    ranked = bm25_rank("apple", docs)
    assert ranked[0]["bm25_score"] == 0.0

def test_rrf_fuse_basic():
    rank1 = [{"url": "a"}, {"url": "b"}]
    rank2 = [{"url": "b"}, {"url": "a"}]
    fused = rrf_fuse([rank1, rank2])

    # a: 1/(60+1) + 1/(60+2)
    # b: 1/(60+2) + 1/(60+1)
    # Should be equal
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
    assert "rrf_score" in fused[0]

def test_rrf_fuse_custom_id():
    rank1 = [{"id": "1"}]
    fused = rrf_fuse([rank1], id_key="id")
    assert fused[0]["id"] == "1"

def test_information_density_diverse():
    # 2 distinct words, log2(2) entropy, density 1.0
    assert information_density("apple banana") == 1.0

def test_information_density_repetitive():
    # "test test" -> p(test)=1 -> entropy = 0
    assert information_density("test test") == 0.0

def test_information_density_short():
    assert information_density("apple") == 0.0
    assert information_density("") == 0.0

def test_compute_confidence_levels():
    # content_relevance=1.0, others=1.0 -> score=1.0 -> "high"
    score, level = compute_confidence(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert level == "high"
    assert score == 1.0

    # low score -> "unknown"
    score, level = compute_confidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert level == "unknown"
    assert score == 0.0

def test_compute_confidence_clamping():
    # Over 1.0 should clamp
    score, _ = compute_confidence(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    assert score == 1.0

    # Negative should clamp
    score, _ = compute_confidence(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    assert score == 0.0
