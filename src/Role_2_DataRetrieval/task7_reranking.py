"""
Task 7 — Reranking Module.

3 phương pháp được hỗ trợ:
    - RRF (Reciprocal Rank Fusion): khuyến nghị — gộp nhiều ranker, không cần model
    - MMR (Maximal Marginal Relevance): tự implement — relevance + diversity
    - Cross-encoder: optional (cần API key, fallback gracefully nếu không có)

Pipeline trong bài lab:
    semantic_search (Task 5) + lexical_search (Task 6)
        → rerank_rrf(RRF, k=60)        ← Task 9 sẽ gọi cái này
        → pageindex fallback (Task 8)  ← nếu cosine < 0.48

Lưu ý RRF (từ comment gốc):
    Điểm RRF fused CHỈ phụ thuộc thứ hạng, không phải độ tương đồng thật.
    Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
    bất kỳ nội dung có thật sự liên quan hay không. ĐỪNG dùng điểm RRF
    để quyết định fallback ở Task 9 — fallback phải so với cosine gốc
    từ dense_results[0]['score'].
"""

from __future__ import annotations

import os
from typing import Optional

# Tùy chọn: load env 1 lần nếu chưa có .env (cho trường hợp chạy test độc lập)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =============================================================================
# RRF — Reciprocal Rank Fusion (BACKBONE cho hybrid search)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Công thức (Cormack et al. 2009):
        RRF(d) = Σ  1 / (k + rank_r(d))

    Trong đó:
        - rank_r(d): thứ hạng của doc d trong ranker r (bắt đầu từ 1)
        - k: hằng số làm mượt (default 60, từ paper gốc, cân bằng giữa
          các ranker có độ dài list khác nhau)

    Args:
        ranked_lists: List các ranked lists từ nhiều ranker.
                      Mỗi item là {'content': str, 'score': float, 'metadata': dict}.
                      Mỗi list phải đã sort giảm dần theo score của ranker đó.
        top_k: Số kết quả cuối cùng (sau khi fuse).
        k: Smoothing constant (default 60 — chuẩn lab, Role 1 check ở CP3).

    Returns:
        List of top_k candidates sorted by RRF score descending.
        Mỗi item giữ nguyên dict gốc, chỉ thêm/update 'score' = rrf_score.

    Edge cases:
        - ranked_lists rỗng hoặc toàn list rỗng → return []
        - 2 documents cùng content từ 2 ranker → cộng dồn RRF score
        - Document chỉ xuất hiện ở 1 ranker → vẫn được tính (điểm thấp hơn)
        - 'content' được dùng làm key (giả định 2 chunk giống nhau → cùng 1 docid).
          Nếu 2 chunk giống text nhưng khác source → vẫn merge (cải thiện: dùng
          metadata['source'] + metadata['chunk_index'] làm key, nhưng content
          là đủ cho corpus ~1500 chunks không bị trùng).
    """
    if not ranked_lists or top_k <= 0:
        return []

    # Key = content (đơn giản, đủ cho corpus ~1500 chunks)
    # Value = (rrf_score, original_dict_with_updated_score)
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            contribution = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contribution
            # Nếu cùng 1 doc xuất hiện ở nhiều ranker, giữ dict đầu tiên
            # nhưng cập nhật metadata source (optional: lưu thêm 'seen_in' list)
            if key not in content_map:
                content_map[key] = item

    # Sort theo RRF score giảm dần
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build output, top_k items
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)  # RRF score thường nhỏ (0.0164 cho top-1)
        results.append(item)

    return results


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vector đã được chuẩn hóa (norm=1)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    # dot product của 2 vector đã normalize = cosine similarity
    return sum(x * y for x, y in zip(a, b))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Công thức:
        MMR(d) = λ * sim(query, d) - (1 - λ) * max(sim(d, d_selected))

    Args:
        query_embedding: Vector embedding của query (cùng model với Task 4/5).
        candidates: List of {
            'content': str,
            'score': float,
            'embedding': list[float],     # ← BẮT BUỘC có key này
            'metadata': dict
        }
        top_k: Số kết quả cuối.
        lambda_param: 1.0 = chỉ relevance, 0.0 = chỉ diversity.
                      Default 0.7 (ưu tiên relevance nhưng có diversity).

    Returns:
        List of top_k candidates selected greedily by MMR.

    Lưu ý:
        - Cần 'embedding' trong mỗi candidate. Nếu candidates từ Task 5
          không có embedding, cần query ChromaDB lại để lấy.
        - O(N × top_k) — đủ nhanh cho top_k=5 và N~20.
        - Cộng dồn sim_to_selected để tránh chọn các chunk gần giống nhau.
    """
    if not candidates or top_k <= 0:
        return []

    # Nếu top_k >= len(candidates), trả về sort theo relevance thôi
    if top_k >= len(candidates):
        return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

    selected: list[int] = []
    remaining: set[int] = set(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx: Optional[int] = None
        best_mmr_score = float("-inf")

        for idx in remaining:
            cand_emb = candidates[idx].get("embedding")
            if not cand_emb:
                # Không có embedding → skip hoặc fallback score
                continue

            # Relevance: sim(query, candidate)
            relevance = _cosine_sim(query_embedding, cand_emb)

            # Diversity penalty: max sim với các đã chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                if sel_emb:
                    sim = _cosine_sim(cand_emb, sel_emb)
                    if sim > max_sim_to_selected:
                        max_sim_to_selected = sim

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    # Trả về candidates theo thứ tự MMR
    return [candidates[i] for i in selected]


# =============================================================================
# CROSS-ENCODER RERANKER (Optional — cần JINA_API_KEY)
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model: str = "jina-reranker-v2-base-multilingual",
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker API (cross-encoder).

    Args:
        query: Câu truy vấn.
        candidates: List of {'content': str, 'score': float, 'metadata': dict}.
        top_k: Số kết quả sau rerank.
        model: Tên model trên Jina (default: multilingual, tốt cho tiếng Việt).

    Returns:
        List of top_k candidates, re-scored và sorted by relevance_score descending.

    Fallback:
        - Nếu không có JINA_API_KEY → return top_k đầu của candidates (không rerank).
        - Nếu API call fail → log warning + return top_k đầu.
    """
    if not candidates or top_k <= 0:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        # Fallback: trả về input theo thứ tự cũ
        return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

    try:
        import requests
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json().get("results", [])

        output = []
        for r in reranked:
            idx = r["index"]
            item = candidates[idx].copy()
            item["score"] = round(float(r["relevance_score"]), 4)
            output.append(item)
        return output

    except Exception as e:
        # API fail → fallback
        print(f"⚠ Jina reranker failed ({e}), falling back to original order")
        return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]


# =============================================================================
# Unified rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
    query_embedding: Optional[list[float]] = None,
    ranked_lists: Optional[list[list[dict]]] = None,
    lambda_param: float = 0.7,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Unified reranking interface — wrapper gọi hàm phù hợp.

    Args:
        query: Câu truy vấn (dùng cho cross-encoder).
        candidates: List of candidates cho cross-encoder / MMR.
        top_k: Số kết quả sau rerank.
        method: "rrf" | "mmr" | "cross_encoder".
        query_embedding: Chỉ dùng cho "mmr".
        ranked_lists: Chỉ dùng cho "rrf".
        lambda_param: Chỉ dùng cho "mmr".
        rrf_k: Chỉ dùng cho "rrf" (default 60).

    Returns:
        List of top_k reranked candidates.

    Examples:
        # RRF (THƯỜNG DÙNG NHẤT trong bài lab)
        results = rerank(
            query="...",
            candidates=[],
            method="rrf",
            ranked_lists=[semantic_results, lexical_results],
            top_k=5,
        )

        # MMR (yêu cầu embedding)
        results = rerank(
            query="...",
            candidates=results_with_embedding,
            method="mmr",
            query_embedding=emb,
            top_k=5,
        )

        # Cross-encoder (cần JINA_API_KEY)
        results = rerank(
            query="...",
            candidates=results,
            method="cross_encoder",
            top_k=5,
        )
    """
    method = method.lower()

    if method == "rrf":
        # Test gọi rerank(query, candidates, top_k=...) truyền candidates trực tiếp,
        # nhưng RRF cần ranked_lists (1+ list). Linh hoạt: nếu có ranked_lists thì
        # dùng, không thì wrap candidates thành ranked_lists đơn (treat là 1 ranker).
        if ranked_lists:
            return rerank_rrf(ranked_lists, top_k=top_k, k=rrf_k)
        if candidates:
            # Sort candidates theo score giảm dần trước khi RRF (1 list = "rerank lại").
            sorted_candidates = sorted(
                candidates, key=lambda x: x.get("score", 0.0), reverse=True
            )
            return rerank_rrf([sorted_candidates], top_k=top_k, k=rrf_k)
        return []  # cả ranked_lists và candidates đều rỗng

    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("rerank(method='mmr') requires query_embedding")
        return rerank_mmr(
            query_embedding=query_embedding,
            candidates=candidates,
            top_k=top_k,
            lambda_param=lambda_param,
        )

    elif method == "cross_encoder":
        return rerank_cross_encoder(query=query, candidates=candidates, top_k=top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method!r}. Use 'rrf' | 'mmr' | 'cross_encoder'.")


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Task 7: Reranking (RRF + MMR + Cross-encoder)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: RRF với 2 dummy lists
    # -------------------------------------------------------------------------
    print("\n>>> Test 1: RRF (Reciprocal Rank Fusion)")

    list_dense = [
        {"content": "Điều kiện thành lập công ty TNHH", "score": 0.85, "metadata": {"source": "a.md"}},
        {"content": "Thủ tục đăng ký hộ kinh doanh", "score": 0.72, "metadata": {"source": "b.md"}},
        {"content": "Quy định thuế cho hộ kinh doanh", "score": 0.65, "metadata": {"source": "c.md"}},
    ]
    list_sparse = [
        {"content": "Thủ tục đăng ký hộ kinh doanh", "score": 5.2, "metadata": {"source": "b.md"}},
        {"content": "Điều kiện thành lập công ty TNHH", "score": 4.8, "metadata": {"source": "a.md"}},
        {"content": "Quy chế bán hàng trên TikTok Shop", "score": 3.1, "metadata": {"source": "d.md"}},
    ]

    rrf_results = rerank_rrf([list_dense, list_sparse], top_k=5, k=60)
    print(f"  → {len(rrf_results)} results (k=60)")
    for i, r in enumerate(rrf_results, 1):
        print(f"  {i}. [RRF={r['score']:.4f}] {r['content'][:60]}")

    # Verify: doc xuất hiện ở cả 2 lists → score cao nhất
    # "Thủ tục đăng ký hộ kinh doanh" rank 2 ở dense + rank 1 ở sparse
    # = 1/(60+2) + 1/(60+1) ≈ 0.01613 + 0.01639 ≈ 0.03252
    print("\n  Verify: doc xuất hiện ở cả 2 lists nên RRF score cao nhất")
    print(f"  Top-1 RRF score ≈ {rrf_results[0]['score']:.4f} (expected ~0.0325)")

    # -------------------------------------------------------------------------
    # Test 2: MMR với dummy embeddings
    # -------------------------------------------------------------------------
    print("\n>>> Test 2: MMR (Maximal Marginal Relevance)")

    candidates_with_emb = [
        {"content": "A", "score": 0.9, "embedding": [1.0, 0.0, 0.0], "metadata": {}},
        {"content": "A2", "score": 0.85, "embedding": [0.99, 0.1, 0.0], "metadata": {}},  # gần A
        {"content": "B", "score": 0.7, "embedding": [0.0, 1.0, 0.0], "metadata": {}},
        {"content": "C", "score": 0.6, "embedding": [0.0, 0.0, 1.0], "metadata": {}},
    ]
    query_emb = [1.0, 0.0, 0.0]  # giống A nhất

    mmr_results = rerank_mmr(query_emb, candidates_with_emb, top_k=3, lambda_param=0.7)
    print(f"  → {len(mmr_results)} results (lambda=0.7)")
    for i, r in enumerate(mmr_results, 1):
        print(f"  {i}. [score={r['score']}] {r['content']}")

    # -------------------------------------------------------------------------
    # Test 3: Cross-encoder (không có API key → fallback)
    # -------------------------------------------------------------------------
    print("\n>>> Test 3: Cross-encoder (Jina API — fallback nếu không có key)")
    ce_results = rerank_cross_encoder(
        query="điều kiện thành lập công ty",
        candidates=list_dense,
        top_k=2,
    )
    print(f"  → {len(ce_results)} results (không có JINA_API_KEY → fallback order cũ)")
    for i, r in enumerate(ce_results, 1):
        print(f"  {i}. [score={r['score']}] {r['content'][:60]}")

    # -------------------------------------------------------------------------
    # Test 4: Unified rerank() interface
    # -------------------------------------------------------------------------
    print("\n>>> Test 4: Unified rerank() interface")
    unified = rerank(
        query="test",
        candidates=[],
        method="rrf",
        ranked_lists=[list_dense, list_sparse],
        top_k=3,
    )
    print(f"  → rerank(method='rrf') returned {len(unified)} items")

    print("\n" + "=" * 70)
    print("✓ All Task 7 tests completed")
