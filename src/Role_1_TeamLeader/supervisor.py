"""
Supervisor Pattern — Multi-Agent Routing cho RAG Pipeline (Role 1 — bonus).

Pattern:
    User Query
        ↓
    [Supervisor]  ← LLM routing (phân loại intent)
        ↓ (route đến 1 trong N workers)
    ┌─ Hybrid RAG Worker (Task 9)        ← câu hỏi pháp lý cụ thể
    ├─ PageIndex Worker (Task 8)         ← fallback cho query obscurities
    ├─ General Chat Worker               ← chitchat, hỏi ngoài domain
    └─ Critic Worker                     ← verify quality, đánh giá output
        ↓
    Final Answer + Reasoning Trace

Concept (LangGraph-style):
    Supervisor không tự trả lời — nó chỉ quyết định WORKER nào phù hợp,
    rồi gọi worker đó, rồi (tùy chọn) gọi Critic để verify trước khi trả về.

Tại sao pattern này hữu ích so với Task 9 thuần:
    1. **Giảm chi phí LLM**: không phải query nào cũng cần dense+BM25+RRF+LLM.
       Ví dụ: "Xin chào" → General Chat, không cần retrieval.
    2. **Specialization**: Hybrid RAG tối ưu cho pháp lý, nhưng có thể có
       worker cho các lĩnh vực khác (tin tức, hỏi đáp chung).
    3. **Observability**: trace rõ ràng — biết query nào đi qua worker nào,
       fallback khi nào, retry khi nào.
    4. **Extensibility**: thêm worker mới dễ dàng (chỉ cần đăng ký).

Trade-offs:
    - Thêm 1 lần LLM call cho routing (tốn ~1-2s + chi phí token)
    - Phức tạp hơn Task 9 đơn lẻ
    - Cần prompt engineering cho router tốt

Lưu ý implementation:
    - Supervisor dùng CÙNG LLM với Task 10 (OpenAI client + fallback chain).
    - Routing prompt đơn giản → chỉ trả về 1 keyword (hybrid/pageindex/chat).
    - Nếu LLM fail → fallback về "hybrid" (an toàn nhất).
"""

from __future__ import annotations

import re
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Các loại worker có thể route đến
WorkerType = Literal["hybrid", "pageindex", "chat", "retry"]

# Confidence threshold để dùng hybrid search (cosine gốc)
HYBRID_CONFIDENCE_THRESHOLD = 0.48

# Số lần retry tối đa nếu Critic không pass
MAX_RETRIES = 2

# Ngưỡng Critic: output phải dài > MIN_ANSWER_CHARS và có >= 1 citation
MIN_ANSWER_CHARS = 50
MIN_CITATIONS = 1


# =============================================================================
# ROUTER (LLM-based intent classification)
# =============================================================================

ROUTER_PROMPT = """Bạn là bộ định tuyến (router) cho hệ thống RAG tư vấn pháp lý về khởi nghiệp và thương mại điện tử tại Việt Nam.

Nhiệm vụ: Phân loại câu hỏi của người dùng vào ĐÚNG 1 trong 4 loại sau:

1. **hybrid** — Câu hỏi pháp lý CỤ THỂ, có thể trả lời từ văn bản luật / nghị định trong cơ sở tri thức.
   Ví dụ: "Điều kiện thành lập công ty TNHH?", "Thuế TNCN cho người bán TikTok Shop?", "Hồ sơ đăng ký hộ kinh doanh cá thể?"

2. **pageindex** — Câu hỏi cần tra cứu CHI TIẾT trong cấu trúc văn bản pháp lý (số điều, khoản cụ thể).
   Ví dụ: "Điều 78 Luật Doanh nghiệp 2020 quy định gì?", "Khoản 3 Điều 5 Nghị định 52 có nội dung gì?"

3. **chat** — CÂU CHITCHAT, hỏi ngoài domain pháp lý khởi nghiệp/TMĐT, hoặc yêu cầu đơn giản.
   Ví dụ: "Xin chào", "Bạn là ai?", "Cảm ơn", "2+2 bằng mấy?", "Thời tiết hôm nay?"

4. **retry** — KHÔNG BAO GIỜ trả về loại này. Chỉ dùng nội bộ.

QUY TẮC:
- Nếu KHÔNG CHẮC → mặc định "hybrid" (an toàn nhất, pipeline sẽ fallback PageIndex nếu score thấp)
- CHỈ trả lời 1 keyword duy nhất: hybrid / pageindex / chat

Câu hỏi: {query}

Loại:"""


def _get_llm_client():
    """Lazy-init LLM client (tái sử dụng từ Task 10 — fallback chain OpenRouter → OpenAI)."""
    try:
        from ..Role_3_FrontendChatbot.task10_generation import _get_llm_client
        return _get_llm_client()
    except Exception as e:
        raise RuntimeError(f"LLM client không khả dụng: {e}")


def _call_llm_simple(messages: list[dict], max_tokens: int = 10) -> str:
    """LLM call đơn giản — trả về text ngắn (dùng cho routing)."""
    try:
        client, provider = _get_llm_client()
        from ..Role_3_FrontendChatbot.task10_generation import _normalize_model_name
        model = _normalize_model_name("openai/gpt-4o-mini", provider)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # routing cần deterministic
            top_p=1.0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        # LLM fail → return empty, supervisor sẽ fallback
        return ""


def _route(query: str) -> WorkerType:
    """
    LLM-based router: phân loại query → loại worker phù hợp.

    Args:
        query: Câu hỏi của user.

    Returns:
        'hybrid' | 'pageindex' | 'chat'
        Mặc định 'hybrid' nếu LLM fail hoặc trả về kết quả không hợp lệ.
    """
    if not query or not query.strip():
        return "chat"

    messages = [
        {"role": "system", "content": "Bạn là bộ router. Chỉ trả lời 1 keyword: hybrid, pageindex, hoặc chat."},
        {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
    ]

    raw = _call_llm_simple(messages, max_tokens=10).lower()

    # Parse keyword
    if "chat" in raw and "hybrid" not in raw and "pageindex" not in raw:
        return "chat"
    if "pageindex" in raw:
        return "pageindex"
    if "hybrid" in raw:
        return "hybrid"

    # Fallback: dùng heuristic nhanh (không cần LLM)
    return _heuristic_route(query)


def _heuristic_route(query: str) -> WorkerType:
    """Fallback routing không cần LLM — dựa trên keyword matching."""
    q = query.strip().lower()

    # Chitchat patterns
    chat_patterns = [
        r"^(xin chào|hello|hi|chào|hey)\b",
        r"\b(cảm ơn|thank|cám ơn)",
        r"^(bạn là ai|là gì vậy|gì đó)",
        r"\b(thời tiết|weather|2\+2)",
        r"^(tạm biệt|bye|goodbye)",
    ]
    for pat in chat_patterns:
        if re.search(pat, q):
            return "chat"

    # Nếu có số điều/khoản cụ thể → pageindex
    if re.search(r"(điều|khoản)\s+\d+|article\s+\d+", q):
        return "pageindex"

    # Mặc định: hybrid
    return "hybrid"


# =============================================================================
# WORKERS
# =============================================================================

def hybrid_rag_worker(query: str, top_k: int = 5) -> dict:
    """
    Worker 1: Hybrid RAG — dùng Task 9 (hybrid search + reranking + fallback).

    Args:
        query: Câu hỏi của user.
        top_k: Số chunks tối đa.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str,
            'worker': 'hybrid',
            'reasoning': str  # explain why this worker was chosen
        }
    """
    from ..Role_3_FrontendChatbot.task10_generation import generate_with_citation

    result = generate_with_citation(query, top_k=top_k)
    result["worker"] = "hybrid"
    result["reasoning"] = (
        f"Query được route đến hybrid RAG. "
        f"Retrieval source: {result.get('retrieval_source', 'unknown')}. "
        f"Pipeline: Semantic (BGE-M3) + BM25 → RRF → LLM Generation."
    )
    return result


def pageindex_worker(query: str, top_k: int = 5) -> dict:
    """
    Worker 2: PageIndex only — bỏ qua hybrid, dùng PageIndex làm primary.

    Hữu ích khi user hỏi về số điều/khoản cụ thể cần tra cứu structured.

    Args:
        query: Câu hỏi của user.
        top_k: Số chunks tối đa.

    Returns:
        dict giống hybrid_rag_worker.
    """
    from ..Role_3_FrontendChatbot.task8_pageindex_vectorless import pageindex_search
    from ..Role_3_FrontendChatbot.task10_generation import (
        SYSTEM_PROMPT, _call_llm, _normalize_model_name, _get_llm_client
    )

    # Bước 1: PageIndex search
    chunks = pageindex_search(query, top_k=top_k) or []

    if not chunks:
        # PageIndex không có key / lỗi → fallback hybrid
        return hybrid_rag_worker(query, top_k)

    # Bước 2: Format + generate
    from ..Role_3_FrontendChatbot.task10_generation import (
        reorder_for_llm, format_context
    )
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = (
        f"CONTEXT (từ PageIndex — vectorless RAG):\n\n{context}\n\n"
        f"---\n\nCâu hỏi: {query}"
    )

    answer = _call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": "pageindex",
        "worker": "pageindex",
        "reasoning": "Query có tham chiếu điều/khoản cụ thể → dùng PageIndex làm primary (vectorless).",
    }


def chat_worker(query: str, top_k: int = 5) -> dict:
    """
    Worker 3: General Chat — không dùng retrieval, chỉ gọi LLM trực tiếp.

    Hữu ích cho câu chitchat, hỏi ngoài domain, hoặc khi user chào hỏi.

    Args:
        query: Câu hỏi của user.
        top_k: Không dùng (giữ interface thống nhất).

    Returns:
        dict giống hybrid_rag_worker nhưng sources rỗng.
    """
    from ..Role_3_FrontendChatbot.task10_generation import _call_llm

    SYSTEM_PROMPT_CHAT = """Bạn là trợ lý pháp lý tư vấn về khởi nghiệp và thương mại điện tử tại Việt Nam.

Nếu câu hỏi liên quan đến pháp lý khởi nghiệp/TMĐT → gợi ý user hỏi cụ thể hơn để bạn tra cứu.
Nếu câu chitchat → trả lời thân thiện, ngắn gọn, giới thiệu bạn có thể giúp gì.
Nếu câu ngoài domain (toán, thời tiết, v.v.) → trả lời lịch sự rằng đó không phải chuyên môn của bạn.

Trả lời bằng tiếng Việt, dưới 100 từ."""

    answer = _call_llm([
        {"role": "system", "content": SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": query},
    ])

    return {
        "answer": answer,
        "sources": [],
        "retrieval_source": "none",
        "worker": "chat",
        "reasoning": "Câu chitchat / ngoài domain → không cần retrieval, gọi LLM trực tiếp.",
    }


# =============================================================================
# CRITIC
# =============================================================================

CITATION_PATTERN = re.compile(r"\[[\w\.\-]+,(legal|news|unknown)\]")


def critic_worker(output: dict) -> dict:
    """
    Worker 4: Critic — kiểm tra chất lượng output.

    Đánh giá 3 tiêu chí:
        1. Length: answer có quá ngắn không (< MIN_ANSWER_CHARS)?
        2. Citations: có citation [file, type] không?
        3. Source coherence: nếu có sources, answer có đề cập content không?

    Args:
        output: dict từ 1 trong 3 worker trên.

    Returns:
        {
            'passes': bool,
            'score': float  # 0-1, quality score
            'issues': list[str],
            'suggested_action': 'accept' | 'retry_hybrid' | 'retry_pageindex' | 'fallback'
        }
    """
    answer = output.get("answer", "")
    sources = output.get("sources", [])

    issues = []

    # Tiêu chí 1: Length
    if len(answer.strip()) < MIN_ANSWER_CHARS:
        issues.append(f"Answer too short ({len(answer)} chars)")

    # Tiêu chí 2: Citations (chỉ áp dụng nếu có sources)
    if sources:
        citations = CITATION_PATTERN.findall(answer)
        if len(citations) < MIN_CITATIONS:
            issues.append(f"Missing citations (found {len(citations)})")

    # Tiêu chí 3: Source coherence — check có ý nào liên quan đến nội dung sources
    if sources and sources[0].get("content"):
        first_chunk = sources[0]["content"][:200].lower()
        # Lấy 5 từ quan trọng từ chunk đầu (≥5 chars)
        keywords = [w for w in re.findall(r"\b\w{5,}\b", first_chunk) if w.isalpha()][:5]
        answer_lower = answer.lower()
        if keywords and not any(kw in answer_lower for kw in keywords):
            issues.append("Answer seems unrelated to retrieved sources")

    # Tính quality score (1.0 trừ đi phạt)
    score = max(0.0, 1.0 - 0.3 * len(issues))

    # Quyết định action
    if not issues:
        action = "accept"
    elif score >= 0.4:
        action = "retry_hybrid"  # thử lại với hybrid
    else:
        action = "fallback"  # quá tệ → dùng fallback message

    return {
        "passes": not issues,
        "score": round(score, 2),
        "issues": issues,
        "suggested_action": action,
    }


# =============================================================================
# SUPERVISOR (main orchestrator)
# =============================================================================

# Public re-export giữ contract cũ của task10
def generate_with_citation(query: str, top_k: int = 5) -> dict:
    """
    Supervisor orchestrator — entry point thay thế cho Task 10.

    Flow:
        Query → Router → Worker → Critic → Final Answer

    Args:
        query: Câu hỏi của user.
        top_k: Số chunks tối đa (chỉ áp dụng cho hybrid/pageindex).

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str,
            'worker': str,                # 'hybrid' | 'pageindex' | 'chat'
            'reasoning': str,             # từ worker
            'critic': dict | None,        # từ critic
            'supervisor_trace': list[str] # log các bước đã đi qua
        }
    """
    trace = []

    # -------------------------------------------------------------------------
    # Step 1: Routing
    # -------------------------------------------------------------------------
    route = _route(query)
    trace.append(f"[Supervisor] Route query → '{route}'")

    # -------------------------------------------------------------------------
    # Step 2: Dispatch to worker
    # -------------------------------------------------------------------------
    try:
        if route == "hybrid":
            output = hybrid_rag_worker(query, top_k)
        elif route == "pageindex":
            output = pageindex_worker(query, top_k)
        elif route == "chat":
            output = chat_worker(query, top_k)
        else:
            output = hybrid_rag_worker(query, top_k)

        trace.append(f"[Worker] {output['worker']} → retrieved {len(output.get('sources', []))} sources")
    except Exception as e:
        trace.append(f"[Worker] {route} failed: {e}")
        output = {
            "answer": f"⚠️ Xin lỗi, có lỗi khi xử lý câu hỏi: {e}",
            "sources": [],
            "retrieval_source": "none",
            "worker": "error",
            "reasoning": f"Worker {route} failed: {e}",
        }

    # -------------------------------------------------------------------------
    # Step 3: Critic (optional — skip cho chat worker để tiết kiệm)
    # -------------------------------------------------------------------------
    if output.get("worker") not in ("chat", "error"):
        critic_result = critic_worker(output)
        trace.append(
            f"[Critic] score={critic_result['score']}, "
            f"issues={len(critic_result['issues'])}, "
            f"action={critic_result['suggested_action']}"
        )

        # Retry nếu cần
        attempts = 0
        while (
            critic_result["suggested_action"] == "retry_hybrid"
            and attempts < MAX_RETRIES
            and output.get("worker") != "hybrid"
        ):
            attempts += 1
            trace.append(f"[Supervisor] Retry attempt {attempts} with hybrid")
            output = hybrid_rag_worker(query, top_k)
            critic_result = critic_worker(output)

        if critic_result["suggested_action"] == "fallback":
            trace.append("[Supervisor] Critic fallback → trả message không thể xác minh")
            output["answer"] = (
                "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
                "Bạn nên tham khảo ý kiến luật sư hoặc cơ quan có thẩm quyền."
            )
    else:
        critic_result = None

    # -------------------------------------------------------------------------
    # Step 4: Package result
    # -------------------------------------------------------------------------
    output["critic"] = critic_result
    output["supervisor_trace"] = trace

    return output


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        # (label, query)
        ("hybrid", "Bán hàng trên TikTok Shop có phải nộp thuế không?"),
        ("hybrid", "Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?"),
        ("pageindex", "Điều 78 Luật Doanh nghiệp 2020 quy định gì?"),
        ("chat", "Xin chào bạn!"),
        ("chat", "Cảm ơn bạn nhiều nhé"),
        ("chat", "2 + 2 bằng mấy?"),
        ("nonsense", "xyzabc123 nonsense query"),
    ]

    for label, q in test_queries:
        print(f"\n{'='*70}")
        print(f"[Expected: {label}] Q: {q}")
        print("-" * 70)
        try:
            result = generate_with_citation(q, top_k=3)
            print(f"Worker: {result.get('worker')}")
            print(f"Reasoning: {result.get('reasoning', '')[:100]}")
            print(f"Trace: {' → '.join(result.get('supervisor_trace', []))}")
            print(f"\nAnswer: {result['answer'][:200]}...")
            print(f"\nSources: {len(result.get('sources', []))} chunks | retrieval: {result.get('retrieval_source')}")
            if result.get("critic"):
                c = result["critic"]
                print(f"Critic: score={c['score']}, action={c['suggested_action']}, issues={c['issues']}")
        except Exception as e:
            print(f"❌ Error: {e}")