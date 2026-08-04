"""
Chatbot UI — Streamlit Interface cho RAG Pipeline (Role 3).

Toàn bộ giao diện Streamlit được đóng gói trong module này. File `app.py` ở thư mục
gốc chỉ là entrypoint mỏng gọi `run_app()`, nhờ vậy Role 3 làm việc hoàn toàn trong
`src/Role_3_FrontendChatbot/` mà không đụng vào code của role khác.

Chạy:
    streamlit run app.py

Luồng dữ liệu:
    UI (module này)
      └→ Role_3/task10_generation.generate_with_citation(query, top_k)
           └→ Role_2/task9_retrieval_pipeline.retrieve(...)
                └→ Role_3/task5 (semantic) + Role_4/task6 (lexical)
                   + Role_2/task7 (rerank) + Role_3/task8 (PageIndex fallback)
"""

import inspect
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

PAGE_TITLE = "University Services RAG Chatbot"
PAGE_ICON = "🎓"

USER_AVATAR = "🧑‍🎓"
ASSISTANT_AVATAR = "🎓"

SUGGESTED_QUESTIONS = [
    "Học phí tại RMIT Vietnam là bao nhiêu?",
    "Làm sao để đặt phòng học nhóm ở thư viện?",
    "Điều kiện xin học bổng Academic Achievement?",
    "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
    "Cách đăng ký học phần qua myRMIT?",
]

# Nhãn hiển thị cho trường `retrieval_source` mà Task 9 trả về
SOURCE_LABELS = {
    "hybrid": ("🔀", "Hybrid Search", "Semantic + BM25 → RRF Rerank"),
    "pageindex": ("📑", "PageIndex Fallback", "Điểm cosine dưới ngưỡng → tra cứu theo cấu trúc tài liệu"),
    "none": ("∅", "Không có kết quả", "Pipeline không trả về chunk nào"),
}

# Các task pipeline cần cho UI hoạt động, dùng cho bảng trạng thái ở sidebar
PIPELINE_TASKS = [
    ("Task 5 — Semantic Search", "src.Role_3_FrontendChatbot.task5_semantic_search", "semantic_search"),
    ("Task 6 — Lexical Search", "src.Role_4_EvaluationQA.task6_lexical_search", "lexical_search"),
    ("Task 7 — RRF Rerank", "src.Role_2_DataRetrieval.task7_reranking", "rerank_rrf"),
    ("Task 8 — PageIndex Fallback", "src.Role_3_FrontendChatbot.task8_pageindex_vectorless", "pageindex_search"),
    ("Task 9 — Retrieval Pipeline", "src.Role_2_DataRetrieval.task9_retrieval_pipeline", "retrieve"),
    ("Task 10 — Generation", "src.Role_3_FrontendChatbot.task10_generation", "generate_with_citation"),
]


# =============================================================================
# PIPELINE STATUS — kiểm tra không gây side effect
# =============================================================================

def _is_implemented(module_path: str, func_name: str) -> bool:
    """
    Kiểm tra một hàm đã được implement chưa mà KHÔNG gọi nó.

    Đọc source code của hàm và tìm `raise NotImplementedError`. Cách này an toàn hơn
    việc gọi thử: gọi `retrieve()` sẽ mở ChromaDB, gọi `generate_with_citation()` sẽ
    tốn một request LLM chỉ để biết trạng thái.
    """
    try:
        import importlib

        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return "raise NotImplementedError" not in inspect.getsource(func)
    except Exception:
        return False


def _collect_status() -> list[tuple[str, bool]]:
    """Trả về [(tên task, đã implement chưa)] theo thứ tự trong PIPELINE_TASKS."""
    return [(label, _is_implemented(mod, fn)) for label, mod, fn in PIPELINE_TASKS]


def _available_llm_key() -> str | None:
    """Tên biến môi trường của API key khả dụng đầu tiên, None nếu chưa cấu hình."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        if os.getenv(key):
            return key
    return None


# =============================================================================
# RENDER HELPERS
# =============================================================================

def _render_sources(sources: list[dict], show_scores: bool, key_prefix: str) -> None:
    """
    Hiển thị danh sách chunk đã dùng làm bằng chứng.

    Mỗi chunk là dict {'content', 'score', 'metadata', 'source'} theo contract của
    Task 9. Dùng `.get()` cho mọi trường vì lúc pipeline chưa hoàn thiện, chunk có
    thể thiếu key.
    """
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, chunk in enumerate(sources, 1):
            meta = chunk.get("metadata", {}) or {}
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = chunk.get("score", 0.0)

            st.markdown(f"**[{i}] {source_name}** &nbsp;·&nbsp; `{doc_type}`")

            if show_scores:
                # Điểm cosine nằm trong [0, 1]; điểm RRF rất nhỏ (~0.016) nên clamp
                # lại để thanh progress không bị lỗi khi pipeline trả về thang khác.
                st.progress(min(max(float(score), 0.0), 1.0), text=f"score: {score:.4f}")

            content = chunk.get("content", "")
            preview = content[:400] + ("…" if len(content) > 400 else "")
            st.text(preview)

            if i < len(sources):
                st.divider()


def _render_retrieval_badge(retrieval_source: str) -> None:
    """Hiển thị nhánh retrieval đã chạy (hybrid hay fallback) để tiện demo trước lớp."""
    icon, label, note = SOURCE_LABELS.get(
        retrieval_source, ("🔍", retrieval_source or "unknown", "")
    )
    st.caption(f"{icon} **{label}** — {note}")


def _render_message(msg: dict, show_scores: bool, key_prefix: str) -> None:
    """Vẽ lại một tin nhắn từ lịch sử chat."""
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("retrieval_source"):
                _render_retrieval_badge(msg["retrieval_source"])
            _render_sources(msg.get("sources", []), show_scores, key_prefix)


def _export_markdown(messages: list[dict]) -> str:
    """Kết xuất lịch sử hội thoại ra Markdown để nộp kèm báo cáo nhóm."""
    lines = [
        f"# Lịch sử hội thoại — {PAGE_TITLE}",
        f"_Xuất lúc {datetime.now():%d/%m/%Y %H:%M}_",
        "",
    ]
    for msg in messages:
        who = "🧑‍🎓 **Sinh viên**" if msg["role"] == "user" else "🎓 **Chatbot**"
        lines.append(f"### {who}")
        lines.append(msg["content"])
        for i, chunk in enumerate(msg.get("sources", []), 1):
            meta = chunk.get("metadata", {}) or {}
            lines.append(
                f"> [{i}] {meta.get('source', 'Unknown')} "
                f"(`{meta.get('type', 'unknown')}`, score {chunk.get('score', 0):.4f})"
            )
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# SIDEBAR
# =============================================================================

def _render_sidebar() -> dict:
    """Vẽ sidebar và trả về dict các tham số điều khiển pipeline."""
    with st.sidebar:
        st.title(f"{PAGE_ICON} University Services RAG")
        st.caption(
            "Trợ lý hỏi đáp về dịch vụ và chính sách đại học "
            "(học phí, học bổng, ký túc xá, thư viện)"
        )

        st.divider()

        # --- Trạng thái pipeline -------------------------------------------
        status = _collect_status()
        ready_count = sum(ok for _, ok in status)
        total = len(status)

        with st.expander(
            f"🔌 Trạng thái pipeline ({ready_count}/{total})",
            expanded=ready_count < total,
        ):
            for label, ok in status:
                st.markdown(f"{'✅' if ok else '⬜'} {label}")

            llm_key = _available_llm_key()
            st.markdown(
                f"{'✅' if llm_key else '⬜'} LLM API key"
                + (f" (`{llm_key}`)" if llm_key else " — chưa cấu hình trong `.env`")
            )

            if ready_count < total:
                st.info(
                    "Giao diện đã sẵn sàng. Các task còn trống sẽ được UI báo lỗi "
                    "thân thiện thay vì crash — cứ demo UI trước, nối pipeline sau.",
                    icon="💡",
                )

        st.divider()

        # --- Câu hỏi gợi ý --------------------------------------------------
        st.subheader("💡 Câu hỏi gợi ý")
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if st.button(question, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_query = question

        st.divider()

        # --- Thiết lập retrieval --------------------------------------------
        st.subheader("⚙️ Thiết lập")
        top_k = st.slider(
            "Số chunks retrieval (top_k)",
            min_value=3,
            max_value=10,
            value=5,
            help="Số đoạn văn bản đưa vào context của LLM. Nhiều quá dễ gây "
            "'lost in the middle', ít quá thiếu bằng chứng.",
        )
        show_scores = st.toggle(
            "Hiện điểm số retrieval",
            value=True,
            help="Hiển thị thanh điểm của từng chunk trong phần Nguồn tham khảo.",
        )

        st.divider()

        # --- Quản lý hội thoại ----------------------------------------------
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ Xoá chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.pending_query = None
                st.rerun()
        with col_export:
            st.download_button(
                "⬇️ Tải chat",
                data=_export_markdown(st.session_state.get("messages", [])),
                file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not st.session_state.get("messages"),
            )

        st.divider()
        st.caption("**Kiến trúc hệ thống:**")
        st.caption(
            "Hybrid Retrieval (Semantic + BM25) → RRF Rerank → "
            "PageIndex Fallback → LLM Generation có Citation"
        )

    return {"top_k": top_k, "show_scores": show_scores}


# =============================================================================
# QUERY HANDLING
# =============================================================================

def _answer_query(query: str, top_k: int) -> dict:
    """
    Gọi pipeline RAG và luôn trả về dict đúng shape để UI không phải phòng thủ.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    try:
        from .task10_generation import generate_with_citation

        response = generate_with_citation(query, top_k=top_k)
        return {
            "answer": response.get("answer", "Pipeline không trả về câu trả lời."),
            "sources": response.get("sources", []),
            "retrieval_source": response.get("retrieval_source", ""),
        }

    except NotImplementedError as exc:
        pending = [label for label, ok in _collect_status() if not ok]
        checklist = "\n".join(f"- ⬜ {label}" for label in pending)
        return {
            "answer": (
                "⚠️ **Pipeline chưa sẵn sàng.**\n\n"
                f"Hàm chưa implement: `{exc}`\n\n"
                "Các task còn trống:\n"
                f"{checklist}\n\n"
                "_Giao diện vẫn hoạt động bình thường — hoàn thiện các task trên là "
                "chatbot trả lời được ngay, không cần sửa thêm gì ở UI._"
            ),
            "sources": [],
            "retrieval_source": "",
        }

    except ImportError as exc:
        return {
            "answer": (
                f"❌ **Không import được module pipeline:** `{exc}`\n\n"
                "Kiểm tra lại đường dẫn import giữa các thư mục Role, và chạy app từ "
                "thư mục gốc repo bằng `streamlit run app.py`."
            ),
            "sources": [],
            "retrieval_source": "",
        }

    except Exception as exc:
        hint = ""
        text = str(exc).lower()
        if "429" in text or "rate" in text:
            hint = "\n\n_Gợi ý: OpenRouter free tier giới hạn 50 request/ngày — đổi sang API key khác._"
        elif "api" in text and "key" in text:
            hint = "\n\n_Gợi ý: kiểm tra `OPENROUTER_API_KEY` trong file `.env`._"
        elif "chroma" in text or "collection" in text:
            hint = "\n\n_Gợi ý: chạy `python -m src.Role_2_DataRetrieval.task4_chunking_indexing` để tạo vector store._"

        return {
            "answer": f"❌ **Lỗi khi chạy RAG Pipeline:** `{type(exc).__name__}: {exc}`{hint}",
            "sources": [],
            "retrieval_source": "",
        }


# =============================================================================
# MAIN APP
# =============================================================================

def run_app() -> None:
    """Entrypoint của giao diện — được `app.py` ở thư mục gốc gọi."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Session state -----------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    settings = _render_sidebar()

    # --- Khu vực chat ------------------------------------------------------
    st.title(f"{PAGE_ICON} University Services RAG Chatbot")
    st.caption(
        "Hệ thống hỏi đáp thông tin dịch vụ đại học "
        "(Học phí, Học bổng, Ký túc xá, Thư viện)"
    )

    if not st.session_state.messages:
        st.info(
            "Đặt câu hỏi ở ô bên dưới, hoặc bấm một **câu hỏi gợi ý** ở thanh bên trái "
            "để bắt đầu. Mỗi câu trả lời đều kèm danh sách tài liệu nguồn.",
            icon="👋",
        )

    for i, msg in enumerate(st.session_state.messages):
        _render_message(msg, settings["show_scores"], key_prefix=f"hist_{i}")

    # --- Nhận câu hỏi ------------------------------------------------------
    user_input = st.chat_input("Nhập câu hỏi về chính sách/dịch vụ đại học…")
    query = user_input or st.session_state.pending_query

    if not query:
        return

    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời…"):
            result = _answer_query(query, settings["top_k"])

        st.markdown(result["answer"])
        if result["retrieval_source"]:
            _render_retrieval_badge(result["retrieval_source"])
        _render_sources(result["sources"], settings["show_scores"], key_prefix="live")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "retrieval_source": result["retrieval_source"],
    })
