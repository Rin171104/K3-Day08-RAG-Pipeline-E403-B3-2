"""
Chatbot UI — Streamlit Interface cho RAG Pipeline (Role 3).

Giao diện Streamlit được đóng gói trong module này. File `app.py` ở thư mục
gốc chỉ là entrypoint mỏng gọi `run_app()`, nhờ vậy Role 3 làm việc hoàn toàn
trong `src/Role_3_FrontendChatbot/` mà không đụng vào code của role khác.

Chạy:
    streamlit run app.py

Luồng dữ liệu:
    UI (module này)
      └→ Role_3/task10_generation.generate_with_citation(query, top_k)
           └→ Role_2/task9_retrieval_pipeline.retrieve(...)
                └→ Role_3/task5 (semantic) + Role_4/task6 (lexical)
                   + Role_2/task7 (rerank) + Role_3/task8 (PageIndex fallback)

UI theme:
    - Dark mode mặc định, gradient xanh navy → tím gold
    - Custom CSS cho cards, badges, chat bubbles
    - Hero section + stats + sidebar status
    - Animations mượt
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

PAGE_TITLE = "Trợ Lý Pháp Lý Khởi Nghiệp & TMĐT"
PAGE_ICON = "⚖️"

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "⚖️"

SUGGESTED_QUESTIONS = [
    "💰 Bán hàng trên TikTok Shop bao nhiêu doanh thu thì phải nộp thuế?",
    "📋 Hồ sơ đăng ký Hộ kinh doanh cá thể cần những giấy tờ gì?",
    "🏢 Điều kiện thành lập Công ty TNHH một thành viên là gì?",
    "🛒 Người bán trên Shopee có phải đăng ký kinh doanh không?",
    "🛡️ Quy định về bảo vệ người tiêu dùng trong thương mại điện tử?",
]

# Nhãn hiển thị cho trường `retrieval_source` mà Task 9 trả về
SOURCE_LABELS = {
    "hybrid": ("🔀", "Hybrid Search", "Semantic + BM25 → RRF Rerank"),
    "pageindex": ("📑", "PageIndex Fallback", "Điểm cosine dưới ngưỡng → tra cứu theo cấu trúc tài liệu"),
    "none": ("∅", "Không có kết quả", "Pipeline không trả về chunk nào"),
}

# Các task pipeline cần cho UI hoạt động, dùng cho bảng trạng thái ở sidebar
PIPELINE_TASKS = [
    ("Task 5 — Semantic Search (BGE-M3)", "src.Role_3_FrontendChatbot.task5_semantic_search", "semantic_search"),
    ("Task 6 — Lexical Search (BM25)", "src.Role_4_EvaluationQA.task6_lexical_search", "lexical_search"),
    ("Task 7 — RRF Rerank", "src.Role_2_DataRetrieval.task7_reranking", "rerank_rrf"),
    ("Task 8 — PageIndex Fallback", "src.Role_3_FrontendChatbot.task8_pageindex_vectorless", "pageindex_search"),
    ("Task 9 — Retrieval Pipeline", "src.Role_2_DataRetrieval.task9_retrieval_pipeline", "retrieve"),
    ("Task 10 — Generation (LLM + Citation)", "src.Role_3_FrontendChatbot.task10_generation", "generate_with_citation"),
]


# =============================================================================
# CUSTOM CSS — Dark theme + gradient + animations
# =============================================================================

def _inject_custom_css() -> None:
    """Inject CSS để UI đẹp hơn (gradient, cards, animations)."""
    st.markdown(
        """
<style>
/* ===== ROOT VARIABLES ===== */
:root {
    --primary: #1e3a8a;          /* Deep navy */
    --primary-light: #3b82f6;    /* Blue */
    --accent: #f59e0b;           /* Gold (phù hợp với "pháp lý") */
    --accent-light: #fbbf24;
    --bg-dark: #0f172a;          /* Slate 900 */
    --bg-card: #1e293b;          /* Slate 800 */
    --bg-card-hover: #334155;    /* Slate 700 */
    --text-primary: #f1f5f9;     /* Slate 100 */
    --text-secondary: #94a3b8;   /* Slate 400 */
    --border: rgba(148, 163, 184, 0.2);
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
}

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e3a8a 100%);
    background-attachment: fixed;
}

/* Main container padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

h1 {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem !important;
    margin-bottom: 0.25rem !important;
}

p, .stMarkdown, li {
    color: var(--text-primary) !important;
}

code {
    background: var(--bg-card) !important;
    color: var(--accent-light) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.9em !important;
}

/* ===== HERO SECTION ===== */
.hero {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(245, 158, 11, 0.15) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    animation: fadeIn 0.5s ease-out;
}

.hero-title {
    font-size: 1.1rem;
    color: var(--text-primary);
    margin: 0 0 8px 0;
    font-weight: 600;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

/* ===== STAT CARDS ===== */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
    box-shadow: 0 8px 16px rgba(245, 158, 11, 0.15);
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== CHAT MESSAGES ===== */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    margin-bottom: 12px !important;
    animation: slideIn 0.3s ease-out;
}

[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #3b82f6, #1e40af) !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent-light) !important;
}

/* ===== BUTTONS ===== */
.stButton button {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    border-color: var(--accent) !important;
    background: linear-gradient(135deg, var(--accent) 0%, #d97706 100%) !important;
    color: white !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
}

.stButton button:active {
    transform: translateY(0);
}

/* Primary CTA button (download chat) */
.stDownloadButton button {
    background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%) !important;
    color: white !important;
    border: none !important;
}

/* ===== SLIDER ===== */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
}

.stSlider [data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}

/* ===== TOGGLE ===== */
.stToggle [data-checked="true"] {
    background: var(--accent) !important;
}

/* ===== EXPANDER (sources) ===== */
.streamlit-expanderHeader,
[data-testid="stExpander"] details summary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease;
}

.streamlit-expanderHeader:hover,
[data-testid="stExpander"] details summary:hover {
    border-color: var(--accent) !important;
}

/* ===== PROGRESS BAR (scores) ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--primary-light)) !important;
    border-radius: 8px !important;
}

/* ===== INFO/SUCCESS/WARNING BOXES ===== */
.stInfo {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(30, 58, 138, 0.15)) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ===== BADGES (retrieval source) ===== */
.retrieval-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px 0;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.2));
    color: var(--accent-light);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ===== DIVIDER ===== */
hr {
    border-color: var(--border) !important;
    margin: 16px 0 !important;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Spinner accent */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-dark);
}
::-webkit-scrollbar-thumb {
    background: var(--bg-card-hover);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}
</style>
""",
        unsafe_allow_html=True,
    )


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

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=True):
        for i, chunk in enumerate(sources, 1):
            meta = chunk.get("metadata", {}) or {}
            source_name = meta.get("source", "Unknown")
            # Chỉ lấy basename nếu là đường dẫn đầy đủ
            if "\\" in str(source_name) or "/" in str(source_name):
                source_name = os.path.basename(str(source_name))
            doc_type = meta.get("type", "unknown")
            score = chunk.get("score", 0.0)
            chunk_source = chunk.get("source", "hybrid")

            # Type icon
            type_icon = "📜" if doc_type == "legal" else "📰" if doc_type == "news" else "📄"
            type_color = "#f59e0b" if doc_type == "legal" else "#3b82f6" if doc_type == "news" else "#94a3b8"

            st.markdown(
                f"""<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
<span style="background: {type_color}22; color: {type_color}; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">[{i}]</span>
<span style="font-weight: 600; color: #f1f5f9;">{type_icon} {source_name}</span>
<span style="color: #94a3b8; font-size: 0.85rem;">· {doc_type}</span>
<span style="color: #94a3b8; font-size: 0.75rem; margin-left: auto;">via {chunk_source}</span>
</div>""",
                unsafe_allow_html=True,
            )

            if show_scores:
                # Điểm cosine nằm trong [0, 1]; điểm RRF rất nhỏ (~0.016) nên clamp
                # lại để thanh progress không bị lỗi khi pipeline trả về thang khác.
                st.progress(min(max(float(score), 0.0), 1.0), text=f"score: {score:.4f}")

            content = chunk.get("content", "")
            preview = content[:400] + ("…" if len(content) > 400 else "")
            st.markdown(
                f"<div style='background: #0f172a; padding: 12px; border-radius: 8px; "
                f"border-left: 3px solid {type_color}; font-size: 0.9rem; "
                f"color: #cbd5e1; margin-bottom: 12px;'>{preview}</div>",
                unsafe_allow_html=True,
            )

            if i < len(sources):
                st.divider()


def _render_retrieval_badge(retrieval_source: str) -> None:
    """Hiển thị nhánh retrieval đã chạy (hybrid hay fallback) để tiện demo trước lớp."""
    icon, label, note = SOURCE_LABELS.get(
        retrieval_source, ("🔍", retrieval_source or "unknown", "")
    )
    st.markdown(
        f"""<div style="display: inline-block; padding: 4px 12px; border-radius: 999px;
font-size: 0.8rem; font-weight: 600; margin: 4px 0;
background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.2));
color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3);">
{icon} <strong>{label}</strong> — <em>{note}</em>
</div>""",
        unsafe_allow_html=True,
    )


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
        who = "🧑‍💼 **Người dùng**" if msg["role"] == "user" else "⚖️ **Trợ lý pháp lý**"
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


def _render_hero() -> None:
    """Hero section đầu trang — giới thiệu + tags."""
    st.markdown(
        """
<div class="hero">
<p class="hero-title">⚖️ Trợ Lý Pháp Lý Thông Minh — Powered by Hybrid RAG</p>
<p class="hero-subtitle">
Tư vấn pháp lý về <strong>khởi nghiệp</strong> và <strong>thương mại điện tử</strong> tại Việt Nam.<br>
Nguồn: <em>Luật Doanh nghiệp 2020 · Nghị định 52/2013/NĐ-CP · Quy định TikTok Shop/Shopee · Thông tư thuế hộ kinh doanh</em>
</p>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

def _render_sidebar() -> dict:
    """Vẽ sidebar và trả về dict các tham số điều khiển pipeline."""
    with st.sidebar:
        st.markdown(f"## {PAGE_ICON} {PAGE_TITLE}")
        st.caption(
            "Tư vấn pháp lý **khởi nghiệp** & **TMĐT** tại Việt Nam\n\n"
            "_Luật Doanh nghiệp · Nghị định 52/2013 · TikTok Shop/Shopee_"
        )

        st.divider()

        # --- Trạng thái pipeline -------------------------------------------
        status = _collect_status()
        ready_count = sum(ok for _, ok in status)
        total = len(status)

        with st.expander(
            f"🔌 Pipeline Status: **{ready_count}/{total}**",
            expanded=False,
        ):
            for label, ok in status:
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} {label}")

            llm_key = _available_llm_key()
            st.markdown("---")
            if llm_key:
                st.markdown(f"🔑 LLM: `{llm_key}` ✅")
            else:
                st.markdown("❌ LLM API key — chưa cấu hình")

        st.divider()

        # --- Câu hỏi gợi ý --------------------------------------------------
        st.markdown("### 💡 Câu hỏi gợi ý")
        st.caption("Bấm để thử nhanh")
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if st.button(question, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_query = question

        st.divider()

        # --- Thiết lập retrieval --------------------------------------------
        st.markdown("### ⚙️ Thiết lập")
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
            if st.button("🗑️ Xoá", use_container_width=True):
                st.session_state.messages = []
                st.session_state.pending_query = None
                st.rerun()
        with col_export:
            st.download_button(
                "⬇️ Tải chat",
                data=_export_markdown(st.session_state.get("messages", [])),
                file_name=f"phaply_chat_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not st.session_state.get("messages"),
            )

        st.divider()

        # --- Architecture diagram ------------------------------------------
        with st.expander("🏗️ Kiến trúc hệ thống", expanded=False):
            st.markdown(
                """
**🔍 Hybrid Retrieval**
- Semantic (BGE-M3)
- BM25 Lexical

**↕️ RRF Rerank** (k=60)

**⚠️ PageIndex Fallback**
- Trigger khi cosine < 0.48

**💬 LLM Generation**
- Có citation + reordering
                """
            )

        st.caption(f"⚖️ Role 1 — {datetime.now():%Y}")

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
            hint = "\n\n_Gợi ý: kiểm tra `OPENAI_API_KEY` hoặc `OPENROUTER_API_KEY` trong file `.env`._"
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
        menu_items={
            "About": "Trợ Lý Pháp Lý Khởi Nghiệp & TMĐT\nHybrid RAG Pipeline · VinUni Lab K3 Day08",
        },
    )

    # Inject custom CSS (dark theme + gradient + animations)
    _inject_custom_css()

    # --- Session state -----------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    settings = _render_sidebar()

    # --- Hero section ------------------------------------------------------
    _render_hero()

    st.write("")  # spacer

    # --- Khu vực chat ------------------------------------------------------
    if not st.session_state.messages:
        st.markdown(
            """
<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(245, 158, 11, 0.08));
border: 1px dashed rgba(245, 158, 11, 0.4); border-radius: 16px;
padding: 28px; text-align: center; margin: 20px 0;
backdrop-filter: blur(10px);">
<p style="font-size: 1.05rem; color: #f1f5f9; margin: 0 0 12px 0; font-weight: 600;">
⚖️ Xin chào! Tôi có thể tư vấn pháp lý về khởi nghiệp và thương mại điện tử.
</p>
<p style="color: #94a3b8; margin: 0; font-size: 0.95rem; line-height: 1.6;">
Đặt câu hỏi ở ô bên dưới, hoặc bấm một <strong style="color: #fbbf24;">câu hỏi gợi ý</strong> ở thanh bên trái.<br>
💡 Mỗi câu trả lời đều kèm <strong style="color: #fbbf24;">trích dẫn nguồn</strong>
(Luật Doanh nghiệp 2020, Nghị định 52/2013, v.v.)
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    for i, msg in enumerate(st.session_state.messages):
        _render_message(msg, settings["show_scores"], key_prefix=f"hist_{i}")

    # --- Nhận câu hỏi ------------------------------------------------------
    user_input = st.chat_input("Đặt câu hỏi pháp lý (VD: thủ tục đăng ký hộ kinh doanh, thuế TMĐT…)")
    query = user_input or st.session_state.pending_query

    if not query:
        return

    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🔍 Đang tra cứu văn bản pháp luật và tổng hợp câu trả lời có trích dẫn…"):
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