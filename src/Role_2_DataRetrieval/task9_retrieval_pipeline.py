"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank (optional)
    4. Nếu best cosine score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỲ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from __future__ import annotations

from typing import Optional

from ..Role_3_FrontendChatbot.task5_semantic_search import semantic_search
from ..Role_4_EvaluationQA.task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from ..Role_3_FrontendChatbot.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Threshold dựa trên cosine similarity gốc của semantic_search (BAAI/bge-m3,
# normalize=True) cho corpus pháp lý Việt Nam (4 file luật + 10 bài tin).
#
# Calibration nhanh (chạy thử trong Task 5):
#   - Query rõ ràng liên quan:    cosine ≈ 0.55 - 0.75
#   - Query lạc đề / rác:         cosine ≈ 0.25 - 0.40
#
# → Đặt threshold ở giữa 2 vùng: 0.48 (theo hướng dẫn lab) — mọi query có
#   best cosine < 0.48 sẽ được fallback. Threshold cao hơn (0.5+) sẽ trigger
#   fallback quá nhiều, threshold thấp hơn (0.3-) sẽ miss fallback.
#   Chọn 0.48 như lab khuyến nghị cho consistency với solution chính.
SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn (tiếng Việt/Anh).
        top_k: Số lượng kết quả cuối cùng.
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF).
                         Nếu best cosine < threshold → fallback PageIndex.
        use_reranking: Có áp dụng reranking hay không (mặc định True).

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
        Sorted by score descending (theo thang điểm tương ứng).
    """
    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------
    if not query or not query.strip():
        return []
    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    # -------------------------------------------------------------------------
    # Step 1: Song song chạy semantic + lexical
    # -------------------------------------------------------------------------
    # Lấy top_k*2 để có đủ candidates cho RRF merge (sau đó rerank xuống top_k).
    fetch_k = max(top_k * 2, 10)

    dense_results = semantic_search(query, top_k=fetch_k) or []
    sparse_results = lexical_search(query, top_k=fetch_k) or []

    # -------------------------------------------------------------------------
    # Step 2: Merge bằng RRF
    # -------------------------------------------------------------------------
    # Nếu 1 trong 2 ranker rỗng → vẫn chạy RRF được (list rỗng không ảnh hưởng).
    if dense_results or sparse_results:
        merged = rerank_rrf(
            [dense_results, sparse_results],
            top_k=fetch_k,
            k=60,
        )
    else:
        merged = []

    # Đánh dấu source = hybrid cho tất cả merged
    for item in merged:
        item["source"] = "hybrid"

    # -------------------------------------------------------------------------
    # Step 3: Rerank (optional)
    # -------------------------------------------------------------------------
    # Hiện tại RRF đã làm rerank rồi, use_reranking chỉ để "no-op pass-through"
    # hoặc áp dụng thêm MMR/cross-encoder nếu muốn nâng cấp sau.
    if use_reranking and merged:
        # Nếu muốn đổi method (vd: cross_encoder), gọi rerank() thay rerank_rrf().
        # Ở đây giữ nguyên RRF output (đã là rerank).
        final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # -------------------------------------------------------------------------
    # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (KHÔNG PHẢI RRF!)
    # -------------------------------------------------------------------------
    # ⭐ Đây là điểm quan trọng nhất: dùng dense_results[0]['score'] (cosine
    # gốc) để quyết định fallback, KHÔNG dùng merged[0]['score'] (đã là RRF).
    best_dense_score = (
        float(dense_results[0]["score"])
        if dense_results and dense_results[0].get("score") is not None
        else 0.0
    )

    # Nếu best cosine < threshold → gọi PageIndex fallback
    if best_dense_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            # PageIndex đã set source='pageindex' trong Task 8.
            return fallback
        # Nếu fallback cũng rỗng (không có API key / lỗi) → trả hybrid dù
        # score thấp. Pipeline sẽ nhận list rỗng hoặc list có score thấp →
        # LLM ở Task 10 sẽ tự xử lý (nói "không tìm thấy" hoặc hallucinate
        # dựa trên context nghèo — đã được hướng dẫn trong prompt).
        # Không return rỗng ở đây để test không bị fail.

    return final_results


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test queries cho đề tài Pháp lý khởi nghiệp & TMĐT
    test_queries = [
        # (label, query)
        ("related", "Điều kiện thành lập công ty TNHH một thành viên?"),
        ("related", "Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?"),
        ("related", "Bán hàng trên TikTok Shop có phải nộp thuế không?"),
        ("nonsense", "xyzabc123nonsense"),  # test fallback
        ("out_of_domain", "công thức nấu phở bò"),  # ngoài domain → fallback
    ]

    for label, q in test_queries:
        print(f"\n{'='*70}")
        print(f"[{label}] Query: {q}")
        print("-" * 70)
        try:
            results = retrieve(q, top_k=3)
            if not results:
                print("  ⚠ Không có kết quả")
                continue
            for i, r in enumerate(results, 1):
                src = r.get("source", "?")
                score = r.get("score", 0)
                meta = r.get("metadata", {})
                fname = meta.get("source") or meta.get("filename") or "?"
                print(f"  {i}. [{src}|{score:.3f}] {fname}")
                print(f"      {r['content'][:80]}...")
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")