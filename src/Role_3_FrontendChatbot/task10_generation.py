"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from ..Role_2_DataRetrieval.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây "lost in the middle"
# (LLM có context window giới hạn; quá nhiều chunks dễ bị phân tâm)
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse để không lặp từ, nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo (low temp → output tập trung vào evidence)
TEMPERATURE = 0.3

# Chọn LLM model: ưu tiên model ":free" của OpenRouter (không tốn credit).
# Fallback chain (chọn theo thứ tự): openrouter free → openrouter paid → openai
# - "openai/gpt-4o-mini" — rẻ, nhanh, tốt cho tiếng Việt
# - "meta-llama/llama-3.3-70b-instruct:free" — miễn phí, multilingual tốt
# - "qwen/qwen-2.5-72b-instruct:free" — miễn phí, top cho tiếng Việt
LLM_MODEL = "openai/gpt-4o-mini"
LLM_MODEL_FALLBACK = "meta-llama/llama-3.3-70b-instruct:free"


# =============================================================================
# SYSTEM PROMPT (Đề tài: Trợ Lý Pháp Lý Khởi Nghiệp & TMĐT)
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý tư vấn về khởi nghiệp và thương mại điện tử tại Việt Nam.

Lĩnh vực tư vấn:
- Thành lập doanh nghiệp (Công ty TNHH, Công ty cổ phần, Hộ kinh doanh cá thể)
- Đăng ký kinh doanh trên sàn TMĐT (Shopee, TikTok Shop, Lazada, Tiki)
- Nghĩa vụ thuế (thuế TNCN, thuế GTGT, thuế hộ kinh doanh)
- Quyền và nghĩa vụ người bán online

Quy tắc BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt, KHÔNG suy luận ngoài
2. Mỗi khẳng định phải có trích dẫn ngay sau, định dạng: [tên_file, loại_tài_liệu]
   Ví dụ: "Theo quy định [luat-doanh-nghiep-2020.pdf, legal], công ty TNHH một thành viên do một cá nhân làm chủ sở hữu..."
3. Nếu context KHÔNG đủ thông tin để trả lời → trả lời CHÍNH XÁC:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có. Bạn nên tham khảo ý kiến luật sư hoặc cơ quan có thẩm quyền."
4. Trả lời bằng TIẾNG VIỆT, ngắn gọn (3-5 đoạn), có cấu trúc rõ ràng:
   - Tóm tắt ngắn ở đầu
   - Chi tiết có citation
   - Lưu ý quan trọng (nếu có)
5. Trích dẫn điều khoản cụ thể khi có thể (ví dụ: "Điều 78, Khoản 2 Luật Doanh nghiệp 2020")
6. TUYỆT ĐỐI KHÔNG:
   - Bịa số hiệu văn bản, điều khoản
   - Trả lời ngoài phạm vi pháp lý (y tế, tài chính cá nhân, v.v.)
   - Sử dụng kiến thức nền ngoài context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA
    (Liu et al. 2023 — "Lost in the Middle").

    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score descending): [1, 2, 3, 4, 5, 6, 7]
    Output order:                     [1, 3, 5, 7, 6, 4, 2]
    (best first, second-best, third-best, ... last-but-one, ..., worst last)

    Pattern cụ thể: lấy chẵn ở đầu, lẻ ở cuối (reversed).
    - front = chunks[::2]  → indices 0, 2, 4, ... → đặt ở đầu
    - back  = chunks[1::2] → indices 1, 3, 5, ... → đặt ở cuối (reversed)

    Args:
        chunks: List sorted by score descending (from retrieval).

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        # Với 1-2 chunks thì không cần reorder, LLM xử lý tốt
        return chunks

    front = chunks[::2]    # index 0, 2, 4 → quan trọng nhất ở đầu
    back = chunks[1::2]    # index 1, 3, ... → kém quan trọng ở cuối (reversed)

    # back[::-1] đảo ngược: chunk tốt nhất trong back lên sát front (vẫn ở "vùng đầu"),
    # chunk tệ nhất xuống cuối (vùng "cuối" — LLM cũng nhớ tốt ở cuối)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string. Mỗi chunk cách nhau bằng "---".
        Format mỗi chunk:
            [Document N | Source: filename.pdf | Type: legal/news]
            <content>
    """
    if not chunks:
        return "(Không có tài liệu tham khảo)"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        # Lấy source name — ưu tiên 'source' (Task 4: filename), fallback 'filename' (Task 6: full path)
        source_raw = meta.get("source") or meta.get("filename") or f"Source {i}"

        # Nếu là full path (Task 6), chỉ lấy tên file để citation gọn
        if "\\" in str(source_raw) or "/" in str(source_raw):
            source_name = os.path.basename(str(source_raw))
        else:
            source_name = str(source_raw)

        doc_type = meta.get("type", "unknown")
        score = chunk.get("score", 0.0)

        context_parts.append(
            f"[Document {i} | Source: {source_name} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LLM CLIENT (lazy + fallback chain)
# =============================================================================

_LLM_CLIENT = None
_LLM_PROVIDER = None  # "openrouter" | "openai"


def _get_llm_client():
    """
    Lazy-init OpenAI client (compatible với OpenRouter).
    Thử OpenRouter trước, fallback OpenAI nếu OpenRouter key không có.

    Trả về tuple (client, provider) để _call_llm biết cách normalize model name:
        - OpenRouter: giữ nguyên `openai/gpt-4o-mini`, `meta-llama/...`
        - OpenAI direct: bỏ prefix → `gpt-4o-mini`, `meta-llama/...` (vẫn có prefix nếu user cố ý)

    Detection placeholder key robust hơn: check cả endswith("xxxxxxx"),
    startswith("sk-or-v1-xxx"), và string chỉ chứa "x".
    """
    global _LLM_CLIENT, _LLM_PROVIDER
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT, _LLM_PROVIDER

    from openai import OpenAI

    # Ưu tiên OpenRouter (nhiều model free) — chỉ dùng nếu key thật
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    is_placeholder = (
        not openrouter_key
        or openrouter_key.endswith("xxxxxxx")
        or openrouter_key.endswith("xxxxxxxx")
        or openrouter_key == "sk-or-v1-..."
        or "xxx" in openrouter_key[-15:]  # nhiều x ở cuối → placeholder
    )

    if not is_placeholder and openrouter_key.startswith("sk-or-"):
        _LLM_CLIENT = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        _LLM_PROVIDER = "openrouter"
        return _LLM_CLIENT, _LLM_PROVIDER

    # Fallback: OpenAI direct (model name KHÔNG có prefix `provider/`)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key.startswith("sk-"):
        _LLM_CLIENT = OpenAI(api_key=openai_key)
        _LLM_PROVIDER = "openai"
        return _LLM_CLIENT, _LLM_PROVIDER

    raise RuntimeError(
        "Không có OPENROUTER_API_KEY hoặc OPENAI_API_KEY hợp lệ trong .env. "
        "Đăng ký miễn phí tại https://openrouter.ai/ để lấy key, "
        "hoặc điền OPENAI_API_KEY thật vào .env."
    )


def _normalize_model_name(model: str, provider: str) -> str:
    """
    Chuẩn hoá model name theo provider:
        - OpenRouter: giữ nguyên `openai/gpt-4o-mini` (đúng format)
        - OpenAI direct: bỏ prefix `provider/` → `gpt-4o-mini`
        - Fallback model khi dùng OpenAI: dùng `gpt-4o-mini` thay vì `meta-llama/...` (model lạ)

    Nếu model name đã không có prefix (vd: `gpt-4o-mini`), trả về nguyên.
    """
    if provider == "openai":
        # Bỏ prefix "provider/" nếu có
        if "/" in model:
            model_only = model.split("/", 1)[1]
        else:
            model_only = model

        # Fallback chain cho OpenAI direct: model nào có sẵn
        if model_only in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"):
            return model_only
        if "llama" in model_only.lower():
            # meta-llama/... không có trên OpenAI direct → đổi sang gpt-4o-mini
            return "gpt-4o-mini"
        if "qwen" in model_only.lower():
            return "gpt-4o-mini"

        return model_only

    # OpenRouter: giữ nguyên (vd: "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct:free")
    return model


def _call_llm(messages: list[dict], model: str | None = None) -> str:
    """
    Call LLM với fallback chain:
        1. model truyền vào (default: LLM_MODEL)
        2. LLM_MODEL_FALLBACK (model free)
        3. Trả về thông báo lỗi (nếu tất cả fail)

    Tự động normalize model name theo provider (OpenRouter vs OpenAI direct).
    """
    client, provider = _get_llm_client()
    primary_model = model or LLM_MODEL

    # Thử model chính trước
    for raw_model in [primary_model, LLM_MODEL_FALLBACK]:
        try:
            actual_model = _normalize_model_name(raw_model, provider)
            response = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=800,  # giới hạn để response không quá dài
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            print(f"  ⚠ Model {raw_model} (→ {actual_model if 'actual_model' in dir() else raw_model}) failed: {e}")
            continue

    # Tất cả fail
    return (
        "Xin lỗi, hiện không thể kết nối với mô hình ngôn ngữ để sinh câu trả lời. "
        "Vui lòng thử lại sau hoặc kiểm tra API key."
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9 — hybrid + fallback)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM (OpenRouter / OpenAI)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user.
        top_k: Số chunks retrieve (default 5).

    Returns:
        {
            'answer': str,           # Câu trả lời có citation từ LLM
            'sources': list[dict],   # Các chunks đã dùng để trả lời
            'retrieval_source': str  # 'hybrid' | 'pageindex' | 'none'
        }
    """
    # -------------------------------------------------------------------------
    # Step 1: Retrieve
    # -------------------------------------------------------------------------
    chunks = retrieve(query, top_k=top_k) or []

    # Nếu không có chunks nào → trả về thông báo không tìm thấy
    if not chunks:
        return {
            "answer": (
                "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
                "Bạn nên tham khảo ý kiến luật sư hoặc cơ quan có thẩm quyền."
            ),
            "sources": [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")

    # -------------------------------------------------------------------------
    # Step 2: Reorder (tránh lost in the middle)
    # -------------------------------------------------------------------------
    reordered = reorder_for_llm(chunks)

    # -------------------------------------------------------------------------
    # Step 3: Format context
    # -------------------------------------------------------------------------
    context = format_context(reordered)

    # -------------------------------------------------------------------------
    # Step 4: Build prompt
    # -------------------------------------------------------------------------
    user_message = (
        f"CONTEXT (tài liệu tham khảo được trích xuất từ cơ sở tri thức):\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\n"
        f"Hãy trả lời câu hỏi trên dựa trên CONTEXT, kèm trích dẫn cho mỗi khẳng định."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # -------------------------------------------------------------------------
    # Step 5: Call LLM
    # -------------------------------------------------------------------------
    answer = _call_llm(messages)

    # -------------------------------------------------------------------------
    # Step 6: Return
    # -------------------------------------------------------------------------
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test queries cho đề tài Trợ lý Pháp lý Khởi nghiệp & TMĐT
    test_queries = [
        "Bán hàng online trên TikTok Shop đạt doanh thu bao nhiêu thì phải nộp thuế TNCN?",
        "Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?",
        "Điều kiện thành lập công ty TNHH một thành viên là gì?",
        "Người bán trên Shopee có phải đăng ký kinh doanh không?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        try:
            result = generate_with_citation(q)
            print(f"\nA: {result['answer']}")
            print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
            for i, s in enumerate(result["sources"][:3], 1):
                src = s.get("metadata", {}).get("source", "?")
                src_name = os.path.basename(str(src)) if "\\" in str(src) or "/" in str(src) else src
                print(f"  {i}. {src_name} (score={s.get('score', 0):.3f})")
        except Exception as e:
            print(f"❌ Lỗi: {e}")