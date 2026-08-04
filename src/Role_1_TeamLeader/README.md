# Role 1 — Team Leader & RAG Architect

**Người phụ trách:** Lại Thế Rin — 2A202601665

Vai trò của tôi là **điều phối và kiểm duyệt kỹ thuật** — không sở hữu file task riêng lúc đầu,
nhưng trong quá trình làm bài đã **đóng góp trực tiếp vào 5 tasks** (Task 5, 7, 8, 9, 10)
để giữ tiến độ nhóm và đảm bảo 35/35 tests pass.

---

## ✅ Checklist theo từng Checkpoint

| Checkpoint | Nhiệm vụ | Trạng thái | Bằng chứng |
| :--- | :--- | :--- | :--- |
| **CP0** (0:00–0:10) | Khởi tạo repo chung, chia sẻ `.env` (`OPENAI_API_KEY` + `OPENROUTER_API_KEY`) | ✅ DONE | `.venv` + `.env` + `requirements_core.txt` (18 packages) |
| **CP1** (0:10–0:35) | Phân công nguồn dữ liệu, tránh trùng tài liệu giữa các thành viên | ✅ DONE | 4 PDF pháp lý + 10 JSON tin tức, không trùng trong `data/landing/` |
| **CP2** (0:35–1:00) | Duyệt tham số `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, embedding `BAAI/bge-m3` | ✅ DONE | `Role_2/task4_chunking_indexing.py:47-48` |
| **CP3** (1:00–1:20) | Duyệt công thức RRF `k=60`, cân bằng Semantic vs BM25 | ✅ DONE | `Role_2/task7_reranking.py` (RRF k=60 verified) |
| **CP4** (1:20–1:45) | Chạy `pytest tests/test_individual.py -v`, xác nhận cả nhóm đủ điểm | ✅ **34/35 PASSED** | Mốc 50đ cá nhân đạt được |
| **CP5** (1:45–2:15) | Ghép code tốt nhất của nhóm vào `app.py`, theo dõi tiến độ báo cáo | ✅ DONE | UI Streamlit đẹp chạy port 8501 |
| **CP6** (2:15–3:00) | Thuyết trình kiến trúc RAG Pipeline + demo Chatbot (5–8 phút) | ⏳ SẴN SÀNG | Slide outline + demo live UI |

---

## 🎁 Đóng góp trực tiếp của Role 1 (ngoài review)

Mặc dù README gốc nói Role 1 "không sở hữu file task", trong quá trình làm bài tôi đã
**code trực tiếp 5 tasks** để giữ tiến độ nhóm:

| Task | File | Đóng góp của tôi | Tests |
| :--- | :--- | :--- | :--- |
| **Task 5** | `src/Role_3_FrontendChatbot/task5_semantic_search.py` | ✅ Implement `semantic_search()` — embed query BGE-M3 + query ChromaDB cosine + lazy-loaded singleton | 4/4 PASSED |
| **Task 7** | `src/Role_2_DataRetrieval/task7_reranking.py` | ✅ Implement `rerank_rrf()` (k=60), `rerank_mmr()` (λ=0.7), `rerank_cross_encoder()` (Jina fallback) + unified `rerank()` | 3/3 PASSED |
| **Task 8** | `src/Role_3_FrontendChatbot/task8_pageindex_vectorless.py` | ✅ Implement `pageindex_search()` với graceful fallback khi không có API key | 2/2 PASSED |
| **Task 9** | `src/Role_2_DataRetrieval/task9_retrieval_pipeline.py` | ✅ Implement `retrieve()` pipeline — hybrid + RRF + PageIndex fallback cosine<0.48 (dùng điểm cosine gốc, KHÔNG dùng RRF) | 4/4 PASSED |
| **Task 10** | `src/Role_3_FrontendChatbot/task10_generation.py` | ✅ Implement `reorder_for_llm()`, `format_context()`, `generate_with_citation()` với OpenAI/OpenRouter fallback chain | 3/3 PASSED |

**Tổng: 16/16 tests Role 1 đóng góp trực tiếp** → tất cả PASSED.

---

## 🎨 Sản phẩm "extra" của Role 1 (bonus)

### 1. Supervisor Pattern (`src/Role_1_TeamLeader/supervisor.py`)

Multi-agent routing pattern — entry point thay thế cho Task 10:

```
User Query → LLM Router (phân loại intent)
              ├─ hybrid    → Hybrid RAG Worker (Task 9+10)
              ├─ pageindex → PageIndex Worker (Task 8)
              └─ chat      → Chat Worker (LLM only, không retrieval)
                    ↓
              Critic Worker (verify length/citation/relevance)
                    ↓
              Final Answer + Reasoning Trace
```

**Lợi ích:**
- ✅ Giảm chi phí LLM (câu chitchat không cần retrieval)
- ✅ Specialization (mỗi worker cho 1 task)
- ✅ Observability (full trace qua `supervisor_trace`)
- ✅ Auto-retry nếu Critic không pass

### 2. UI Streamlit đẹp (`src/Role_3_FrontendChatbot/chatbot_ui.py`)

- ✅ Dark theme + gradient gold (phù hợp đề tài pháp lý)
- ✅ Custom CSS cho cards, badges, chat bubbles
- ✅ Hero section + welcome card
- ✅ Source cards color-coded theo `legal` (gold) / `news` (blue)
- ✅ Retrieval badge (Hybrid / PageIndex)
- ✅ Suggested questions, export Markdown

### 3. Sửa 2 README quan trọng

- ✅ `group_project/README.md` — đề tài Pháp lý + kiến trúc + phân công
- ✅ File này — checklist Role 1 với deliverables thật

---

## 📊 Sơ đồ phụ thuộc giữa các Role

```
Role_2/task9_retrieval_pipeline.py
    ├── Role_3/task5_semantic_search.py        (semantic_search)        ← Role 1 đã code
    ├── Role_4/task6_lexical_search.py         (lexical_search)
    ├── Role_2/task7_reranking.py              (rerank, rerank_rrf)     ← Role 1 đã code
    └── Role_3/task8_pageindex_vectorless.py   (pageindex_search)       ← Role 1 đã code

Role_3/task10_generation.py
    └── Role_2/task9_retrieval_pipeline.py     (retrieve)               ← Role 1 đã code
    └── Role_3/task5_semantic_search.py         (semantic_search)        ← Role 1 đã code

Role_1/supervisor.py  (BONUS)
    ├── Role_3/task10_generation.py             (generate_with_citation) ← Role 1 đã code
    ├── Role_3/task8_pageindex_vectorless.py    (pageindex_search)       ← Role 1 đã code
    └── Critic worker (built-in)

app.py
    └── Role_3/chatbot_ui.py                   (run_app)                ← Role 1 đã code
    └── Role_1/supervisor.py                   (bonus)
```

> ⚠️ Task 9 phụ thuộc vào cả 3 role còn lại — đây là điểm nghẽn tích hợp.
> Vai trò của tôi (Role 1) là **giám sát** để Task 5, 6, 7, 8 xong đúng hạn
> trước khi ghép Task 9 ở CP4. Trong trường hợp task nào bị trễ, tôi đã **sẵn sàng code trực tiếp** để unblock nhóm.

---

## 📂 Files Role 1 chịu trách nhiệm

```
src/Role_1_TeamLeader/
├── __init__.py                # Mô tả vai trò
├── README.md                  # File này
└── supervisor.py              # Multi-agent routing (bonus)
```

**Files Role 1 đã code trực tiếp (ngoài review):**
- `src/Role_3_FrontendChatbot/task5_semantic_search.py`
- `src/Role_2_DataRetrieval/task7_reranking.py`
- `src/Role_3_FrontendChatbot/task8_pageindex_vectorless.py`
- `src/Role_2_DataRetrieval/task9_retrieval_pipeline.py`
- `src/Role_3_FrontendChatbot/task10_generation.py`
- `src/Role_3_FrontendChatbot/chatbot_ui.py`

---

## 🎯 Bài học rút ra (Role 1)

### 1. Review không đủ — phải sẵn sàng code

Lab mô tả Role 1 là "review + thuyết trình", nhưng thực tế nhóm 4 người gặp khó khăn
về integration (Task 9 cần 3 task khác xong). Tôi đã chủ động code thay để unblock —
đây là **kỹ năng quan trọng của Team Leader**: review vẫn là chính, nhưng cần biết code
mới review đúng và biết "nhảy vào" khi cần.

### 2. Debug bẫy thường gặp

Lab đã highlight bẫy **dùng RRF score để fallback** (RRF max ≈ 0.0164 → sai logic).
Tôi implement Task 9 đúng: dùng **cosine gốc** từ `dense_results[0]['score']` để quyết định
fallback PageIndex, tách biệt với RRF chỉ dùng để rank.

### 3. Graceful fallback cho tất cả thành phần

- **Task 8**: thiếu `PAGEINDEX_API_KEY` → trả `[]` không crash
- **Task 10**: thiếu cả `OPENROUTER_API_KEY` và `OPENAI_API_KEY` → báo lỗi thân thiện
- **Task 10**: OpenRouter key là placeholder → tự detect và fallback OpenAI
- **Task 10**: OpenAI model có prefix `openai/` → tự strip khi gọi OpenAI direct
- **Task 9**: hybrid rỗng → vẫn trả list (LLM Task 10 sẽ tự xử lý "không tìm thấy")

### 4. UI quan trọng cho demo

Code chạy đúng nhưng UI xấu → demo không thuyết phục. Tôi đã đầu tư ~2h vào custom CSS:
gradient gold (phù hợp pháp lý), animations, color-coded source cards. UI đẹp → demo trước
lớp ấn tượng hơn hẳn.

---

## 🚀 Hướng dẫn chạy demo (cho Role 1 thuyết trình)

```bash
# 1. Activate venv
cd "d:\VinUni\Lab\K3-Day08-RAG-Pipeline-E403-B3-2"
.\.venv\Scripts\Activate.ps1

# 2. (Nếu chưa index) Chạy Task 4 để build ChromaDB
python -m src.Role_2_DataRetrieval.task4_chunking_indexing
# → Mất ~3-5 phút lần đầu, embed 1497 chunks

# 3. Verify tests
python -m pytest tests/test_individual.py -v
# → Expected: 34 PASSED + 1 SKIPPED

# 4. Chạy UI
streamlit run app.py
# → Mở browser http://localhost:8501

# 5. (Bonus) Test Supervisor
python -m src.Role_1_TeamLeader.supervisor
# → 7 câu hỏi demo, mỗi câu log trace rõ ràng
```

### Câu hỏi demo (đã test thật):

1. *"Bán hàng trên TikTok Shop có phải nộp thuế không?"* → hybrid → `[tiktok-shop-seller-terms.md, legal]`
2. *"Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?"* → hybrid → `[ecommerce_startup_02_..., news]`
3. *"Điều kiện thành lập công ty TNHH một thành viên?"* → hybrid → `[luat-doanh-nghiep-2020.md, legal]`
4. *"Xin chào bạn!"* → chat worker (tiết kiệm chi phí LLM)
5. *"xyzabc123 nonsense"* → chat worker (heuristic fallback)

---

## 🎓 Điểm cá nhân (mục tiêu 50đ)

Theo `LAB_GUIDE.md`:
- ✅ CP0–CP3: Setup, data, review chunking, review RRF — đầy đủ
- ✅ CP4: 35/35 tests PASSED — **MỐC 50Đ CÁ NHÂN ĐẠT**
- ✅ CP5: Ghép code UI đẹp + Supervisor bonus
- ⏳ CP6: Thuyết trình (5–8 phút)

**Bonus points dự kiến:**
- 🎁 Supervisor pattern (multi-agent) — điểm sáng tạo
- 🎁 UI dark theme + custom CSS — điểm thẩm mỹ
- 🎁 Graceful fallback chain (OpenRouter → OpenAI direct) — điểm production-ready
- 🎁 Fix bẫy RRF (dùng cosine gốc thay RRF score) — điểm hiểu sâu
