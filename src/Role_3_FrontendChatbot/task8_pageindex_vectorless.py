"""
Task 8 — PageIndex Vectorless RAG (Fallback cho Hybrid Search).

PageIndex cho phép RAG mà KHÔNG cần vector store — dùng structural understanding
của document (cây phân cấp section) thay vì embedding similarity.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code:      https://github.com/VectifyAI/PageIndex

Vai trò trong pipeline (Task 9):
    Hybrid (Dense + BM25 + RRF) → nếu best_cosine_score < 0.48 → fallback PageIndex.
    Lý do: khi query nằm ngoài domain hoặc quá khó, vector search có thể trả về
    noise với score thấp. PageIndex dùng LLM reasoning trên cấu trúc tài liệu
    nên có thể "hiểu" ngữ nghĩa tốt hơn cho một số trường hợp đặc biệt.

Design choice (Role 1 quyết định):
    - Nếu KHÔNG có PAGEINDEX_API_KEY → trả về [] (graceful fallback).
      Pipeline ở Task 9 sẽ tự handle (trả hybrid results dù score thấp, hoặc
      trả "không tìm thấy" để LLM ở Task 10 tự xử lý).
    - Nếu CÓ API key → upload documents 1 lần (cached trên server PageIndex),
      rồi query bằng `submit_query()` + poll `get_retrieval()`.

Lưu ý từ comment gốc:
    API `/retrieval` hiện đã deprecated (vẫn hoạt động). Response có field
    "retrieved_nodes" — mỗi node có "relevant_contents": list[list[{...}]].
    Cần inspect json thật trước khi parse (không đoán schema từ code cũ).
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent.parent / "data" / "standardized"

# Cache mapping: file_stem -> doc_id (để không upload lại)
_DOC_ID_CACHE_FILE = Path(__file__).parent.parent.parent / "pageindex_doc_ids.json"


# =============================================================================
# CLIENT (lazy import để không fail lúc import module nếu chưa cài đặt)
# =============================================================================

def _get_client():
    """Khởi tạo PageIndex client (lazy import)."""
    try:
        from pageindex import PageIndexClient
    except ImportError as e:
        raise ImportError(
            "Chưa cài pageindex. Chạy: pip install pageindex"
        ) from e

    if not PAGEINDEX_API_KEY:
        raise ValueError(
            "Chưa set PAGEINDEX_API_KEY trong .env. "
            "Đăng ký miễn phí tại https://pageindex.ai/"
        )

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# =============================================================================
# UPLOAD
# =============================================================================

def _load_doc_id_cache() -> dict:
    """Đọc cache {file_stem: doc_id} từ file JSON."""
    if _DOC_ID_CACHE_FILE.exists():
        try:
            return json.loads(_DOC_ID_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_doc_id_cache(cache: dict) -> None:
    """Lưu cache {file_stem: doc_id} ra file JSON."""
    _DOC_ID_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _md_to_pdf_simple(md_path: Path, pdf_path: Path) -> bool:
    """
    Convert Markdown đơn giản sang PDF (dùng fpdf2).
    PageIndex chỉ nhận PDF, không nhận .md trực tiếp.
    Chỉ convert text thô, không cần format Markdown (PageIndex tự tách cấu trúc).
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    text = md_path.read_text(encoding="utf-8").strip()
    if not text:
        return False

    # Bỏ Markdown syntax cơ bản để PDF dễ đọc
    # (giữ nguyên text, PageIndex không quan tâm format)
    clean = text.replace("**", "").replace("__", "").replace("`", "")
    # Bỏ ký tự unicode đặc biệt có thể làm fpdf lỗi
    safe_chars = []
    for ch in clean:
        try:
            ch.encode("latin-1")
            safe_chars.append(ch)
        except UnicodeEncodeError:
            safe_chars.append("?")  # thay thế ký tự không encode được
    clean = "".join(safe_chars)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    # Multi-line text
    for line in clean.split("\n"):
        try:
            pdf.multi_cell(0, 5, line)
        except Exception:
            pdf.multi_cell(0, 5, "[line skipped]")

    pdf.output(str(pdf_path))
    return True


def upload_documents() -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex (1 lần duy nhất).

    Returns:
        dict {file_stem: doc_id} — mapping từ tên file sang doc_id trên PageIndex.

    Raises:
        RuntimeError: nếu thiếu PAGEINDEX_API_KEY.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Thiếu PAGEINDEX_API_KEY. Set trong .env hoặc bỏ qua upload "
            "(Task 9 sẽ fallback về hybrid search thuần)."
        )

    client = _get_client()
    cache = _load_doc_id_cache()

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        stem = md_file.stem
        if stem in cache:
            print(f"  ↻ Cache hit: {md_file.name} → {cache[stem]}")
            continue

        # Convert MD → PDF tạm
        tmp_pdf = Path(__file__).parent.parent.parent / "pageindex_pdfs" / f"{stem}.pdf"
        tmp_pdf.parent.mkdir(parents=True, exist_ok=True)

        if not _md_to_pdf_simple(md_file, tmp_pdf):
            print(f"  ⚠ Skip {md_file.name} (convert PDF fail)")
            continue

        try:
            resp = client.submit_document(str(tmp_pdf))
            # Đoán schema: resp có thể là dict hoặc object
            doc_id = (
                resp.get("doc_id")
                if isinstance(resp, dict)
                else getattr(resp, "doc_id", None) or getattr(resp, "id", None)
            )
            if not doc_id:
                print(f"  ⚠ No doc_id in response: {resp}")
                continue

            cache[stem] = doc_id
            print(f"  ✓ Uploaded: {md_file.name} → {doc_id}")
        except Exception as e:
            print(f"  ✗ Upload fail {md_file.name}: {e}")

    _save_doc_id_cache(cache)
    return cache


# =============================================================================
# SEARCH (Fallback retrieval)
# =============================================================================

def _poll_retrieval(client, retrieval_id: str, max_wait: int = 60, interval: int = 2):
    """
    Poll PageIndex cho đến khi retrieval hoàn thành.
    """
    start = time.time()
    while time.time() - start < max_wait:
        try:
            result = client.get_retrieval(retrieval_id)
            # Status có thể ở: result["status"], result["state"], hoặc nested
            status = (
                result.get("status") if isinstance(result, dict)
                else getattr(result, "status", None)
            )
            if status in ("completed", "complete", "done", "success"):
                return result
            if status in ("failed", "error"):
                print(f"  ⚠ Retrieval failed: {result}")
                return result
        except Exception as e:
            print(f"  ⚠ Poll error: {e}")

        time.sleep(interval)

    print(f"  ⚠ Retrieval timeout after {max_wait}s")
    return None


def pageindex_search(query: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt (cosine < 0.48).

    Args:
        query: Câu truy vấn tiếng Việt/Anh.
        top_k: Số lượng kết quả tối đa.
        doc_ids: Danh sách doc_id để search trên đó. Nếu None → dùng tất cả
                 doc_id đã cache trong pageindex_doc_ids.json.

    Returns:
        List of {
            'content': str,
            'score': float,             # PageIndex không trả score thật →
                                        # tự gán theo rank (1.0 cho top-1, giảm dần)
            'metadata': dict,
            'source': 'pageindex'       # Đánh dấu nguồn retrieval (cho Task 9)
        }

        Nếu thiếu API key / lỗi / không có docs → trả về [] (Task 9 tự handle).
    """
    # -------------------------------------------------------------------------
    # Guard: thiếu API key → fallback gracefully
    # -------------------------------------------------------------------------
    if not PAGEINDEX_API_KEY:
        return []

    # -------------------------------------------------------------------------
    # Lấy doc_ids từ cache nếu không truyền vào
    # -------------------------------------------------------------------------
    if doc_ids is None:
        cache = _load_doc_id_cache()
        doc_ids = list(cache.values())

    if not doc_ids:
        # Chưa upload document nào → trả []
        return []

    # -------------------------------------------------------------------------
    # Query PageIndex API
    # -------------------------------------------------------------------------
    try:
        client = _get_client()
        results: list[dict] = []

        # Submit query cho từng doc (PageIndex API thường query 1 doc / request)
        # Limit số doc để tránh quá tải (top 5 docs là đủ cho fallback)
        for doc_id in doc_ids[:5]:
            try:
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = (
                    resp.get("retrieval_id") if isinstance(resp, dict)
                    else getattr(resp, "retrieval_id", None) or getattr(resp, "id", None)
                )

                if not retrieval_id:
                    continue

                # Poll cho đến khi xong
                retrieval = _poll_retrieval(client, retrieval_id, max_wait=30, interval=2)
                if not retrieval:
                    continue

                # -----------------------------------------------------------------
                # Parse retrieved_nodes
                # -----------------------------------------------------------------
                # Schema (theo comment gốc):
                #   retrieval["retrieved_nodes"] = [
                #       {
                #           "relevant_contents": [
                #               [ {section_title, relevant_content}, ... ],
                #               [ ... ]
                #           ]
                #       },
                #       ...
                #   ]
                #
                # Lưu ý: mỗi node chứa LIST các group, mỗi group là LIST các item.
                # Cần in json.dumps(retrieval, indent=2) trong test thật để confirm schema
                # trước khi parse production code.
                # -----------------------------------------------------------------
                retrieved_nodes = (
                    retrieval.get("retrieved_nodes", [])
                    if isinstance(retrieval, dict)
                    else getattr(retrieval, "retrieved_nodes", [])
                )

                rank = 0
                for node in retrieved_nodes:
                    rel_contents = (
                        node.get("relevant_contents", [])
                        if isinstance(node, dict)
                        else getattr(node, "relevant_contents", [])
                    )
                    for group in rel_contents:
                        # group có thể là list[dict] hoặc list[object]
                        items = group if isinstance(group, list) else [group]
                        for item in items:
                            content = (
                                item.get("relevant_content")
                                if isinstance(item, dict)
                                else getattr(item, "relevant_content", "")
                            )
                            section = (
                                item.get("section_title")
                                if isinstance(item, dict)
                                else getattr(item, "section_title", None)
                            )

                            if not content:
                                continue

                            rank += 1
                            # PageIndex không trả score trực tiếp →
                            # tự gán score theo rank (heuristic: càng cao càng tốt)
                            # Score range [0, 1] tương tự Task 5 để Task 9 dễ xử lý.
                            score = max(0.0, 1.0 - (rank - 1) * 0.1)

                            results.append({
                                "content": content,
                                "score": round(score, 4),
                                "metadata": {
                                    "section": section,
                                    "doc_id": doc_id,
                                    "rank": rank,
                                },
                                "source": "pageindex",
                            })

                            if len(results) >= top_k:
                                break
                        if len(results) >= top_k:
                            break
                    if len(results) >= top_k:
                        break

                if len(results) >= top_k:
                    break

            except Exception as e:
                print(f"  ⚠ PageIndex query fail for doc {doc_id}: {e}")
                continue

        return results[:top_k]

    except Exception as e:
        print(f"⚠ PageIndex search error: {e}")
        return []


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Task 8: PageIndex Vectorless RAG (Fallback)")
    print("=" * 70)

    if not PAGEINDEX_API_KEY:
        print("\n⚠ PAGEINDEX_API_KEY chưa được set trong .env")
        print("  → pageindex_search() sẽ trả về [] (graceful fallback)")
        print("  → Đăng ký tại https://pageindex.ai/ để dùng thật")
        print()

        # Test fallback: phải trả về [] không crash
        print(">>> Test fallback (no API key):")
        result = pageindex_search("điều kiện thành lập công ty", top_k=3)
        print(f"  → Returned {len(result)} results (expected: 0)")
        assert result == [], "Fallback phải trả về []"
        print("  ✓ Fallback hoạt động đúng")

    else:
        print("\n✓ PAGEINDEX_API_KEY đã có sẵn")
        print()

        # Test 1: Upload documents (1 lần)
        print(">>> Test 1: Upload documents")
        try:
            cache = upload_documents()
            print(f"  → Uploaded {len(cache)} documents")
        except Exception as e:
            print(f"  ✗ Upload fail: {e}")
            print("  → Skip test 2/3")

        # Test 2: Query thật
        if _DOC_ID_CACHE_FILE.exists():
            print("\n>>> Test 2: Real query")
            test_queries = [
                "Điều kiện thành lập công ty TNHH một thành viên?",
                "Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?",
            ]
            for q in test_queries:
                print(f"\nQ: {q}")
                print("-" * 70)
                results = pageindex_search(q, top_k=3)
                if not results:
                    print("  ⚠ Không có kết quả")
                for r in results:
                    print(f"  [{r['score']:.3f}] {r['content'][:120]}...")
                    print(f"    (section: {r['metadata'].get('section', '?')})")

    print("\n" + "=" * 70)
    print("✓ Task 8 completed")
