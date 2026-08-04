"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Implementation notes:
    - Embedding model: BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - Vector store: ChromaDB (cosine distance, hnsw:space=cosine)
    - Task 4 dùng normalize_embeddings=True → cosine distance trong [0, 2],
      cosine similarity = 1 - distance nằm trong [-1, 1]. Với embeddings đã
      normalize, similarity thực tế nằm trong [0, 1] (vì dot product giữa 2
      vector đã normalize ≤ 1). Score trả về = max(0.0, 1.0 - distance) để
      đảm bảo ≥ 0, phục vụ cho logic fallback ở Task 9 (so sánh với threshold
      0.48 của điểm cosine gốc — xem comment trong task9_retrieval_pipeline.py).
"""

from pathlib import Path

# Cùng đường dẫn với Task 4 — import để dùng chung CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from ..Role_2_DataRetrieval.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


# =============================================================================
# LAZY-LOADED SINGLETONS
# =============================================================================
# Tránh load lại model + ChromaDB client mỗi lần gọi semantic_search()
# (load model ~3-5 giây, mở ChromaDB client ~50ms). Vì app.py có thể gọi
# search() nhiều lần cho các câu hỏi khác nhau trong 1 phiên chat.

_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None
_COLLECTION = None


def _get_embedding_model():
    """Lazy-load SentenceTransformer (load 1 lần, cache lại)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceModel  # tránh import lúc module load
        _EMBEDDING_MODEL = SentenceModel(EMBEDDING_MODEL)
    return _EMBEDDING_MODEL


def _get_collection():
    """Lazy-load ChromaDB collection (mở 1 lần, cache lại)."""
    global _CHROMA_CLIENT, _COLLECTION
    if _COLLECTION is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def _encode_query(query: str) -> list[float]:
    """Embed query string thành vector (cùng model + normalize với Task 4)."""
    model = _get_embedding_model()
    # normalize_embeddings=True: bắt buộc để cosine similarity tính đúng
    # (giống hệt Task 4 lúc index) — nếu không normalize, distance trong
    # ChromaDB sẽ không tương ứng với score ta đang giả định.
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()


# =============================================================================
# PUBLIC API
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine).

    Args:
        query: Câu truy vấn tiếng Việt/Anh bất kỳ.
        top_k: Số lượng kết quả tối đa trả về.

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity, đã chuyển từ distance, nằm trong [0, 1]
            'metadata': dict     # {source, type, chunk_index} từ Task 4
        }
        Sorted by score descending (cao nhất trước).

    Raises:
        RuntimeError: Nếu collection rỗng (chưa chạy Task 4 để index).
    """
    if not query or not query.strip():
        return []

    collection = _get_collection()

    # Guard: nếu collection rỗng → trả [] để pipeline ở Task 9 fallback PageIndex
    # thay vì crash. Test Task 5 sẽ skipTest nếu không có kết quả (xem test_individual.py).
    if collection.count() == 0:
        return []

    # Lấy thêm nhiều hơn top_k để tránh trường hợp vài chunk có score = 0
    # bị ChromaDB trả về cùng với chunk tốt — query n_results=top_k*2 rồi sort lại.
    fetch_k = max(top_k * 2, 10)

    query_embedding = _encode_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance.
        # Với embedding đã normalize (Task 4), similarity ∈ [0, 1].
        # max(0, ...) để chặn numerical noise làm score âm nhỏ.
        score = max(0.0, 1.0 - float(dist))
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta or {},
        })

    # Sort descending theo score (ChromaDB đã sort rồi, nhưng sort lại cho chắc —
    # đặc biệt sau khi clamp về 0).
    output.sort(key=lambda x: x["score"], reverse=True)

    return output[:top_k]


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test thử với vài câu hỏi mẫu (đề tài Pháp lý khởi nghiệp & TMĐT)
    test_queries = [
        "Điều kiện thành lập công ty TNHH một thành viên?",
        "Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?",
        "Bán hàng trên TikTok Shop có phải nộp thuế không?",
        "Quy định về bảo vệ người tiêu dùng trong thương mại điện tử?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("-" * 70)
        try:
            results = semantic_search(q, top_k=3)
            if not results:
                print("  ⚠ Không có kết quả (collection rỗng — chưa chạy Task 4?)")
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "?")
                doc_type = r["metadata"].get("type", "?")
                print(f"  {i}. [{r['score']:.4f}] ({doc_type}) {src}")
                print(f"      {r['content'][:120]}...")
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
