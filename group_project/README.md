# Bài Tập Nhóm — Trợ Lý Pháp Lý Khởi Nghiệp & Thương Mại Điện Tử

## ⚖️ Mục Tiêu

Nhóm xây dựng **chatbot RAG tư vấn pháp lý** về khởi nghiệp và thương mại điện tử tại Việt Nam,
tích hợp pipeline retrieval + generation + evaluation từ bài cá nhân của 4 thành viên.

**Stack chính:** BGE-M3 (embedding) + ChromaDB (vector store) + BM25 (lexical) + RRF (rerank) + PageIndex (fallback) + OpenAI/GPT-4o-mini (generation) + DeepEval (evaluation) + Streamlit (UI).

---

## 📋 Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi pháp lý về khởi nghiệp và TMĐT tại Việt Nam.

**Lĩnh vực tư vấn:**
- 🏢 **Thành lập doanh nghiệp** (Công ty TNHH, Công ty cổ phần, Hộ kinh doanh cá thể)
- 🛒 **Đăng ký kinh doanh trên sàn TMĐT** (Shopee, TikTok Shop, Lazada, Tiki)
- 💰 **Nghĩa vụ thuế** (thuế TNCN, thuế GTGT, thuế hộ kinh doanh)
- 🛡️ **Quyền & nghĩa vụ người bán online, bảo vệ người tiêu dùng**

**Yêu cầu UI:**
- ✅ Giao diện chat Streamlit (dark theme + gradient gold + custom CSS)
- ✅ Trả lời có citation (dựa trên Task 10) với format `[tên_file, loại]`
- ✅ Hiển thị source documents đã dùng (color-coded theo legal/news)
- ✅ Hiển thị retrieval source (hybrid / pageindex) để demo
- ✅ Slider top_k, toggle hiển thị điểm, export chat Markdown
- ✅ Suggested questions cho người dùng mới

**Stack đã dùng:**
```
Streamlit (UI) → Supervisor (Role 1) → Hybrid RAG (Task 9+10)
                                       ↓
                  PageIndex Fallback (Task 8) → LLM Generation (Task 10)
```

**Bonus (Role 1):** Supervisor pattern với LLM-based router phân loại query → 4 workers
(hybrid / pageindex / chat / critic) → retry nếu Critic không pass.

---

## 📊 Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **DeepEval** (đã chọn thay RAGAS — không cần Rust compiler) để evaluate pipeline.

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm | Nhóm dùng |
|-----------|---------|-----------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest | ✅ **ĐÃ CHỌN** |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính | ❌ (cần Rust) |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh | ❌ |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config:
   - Config A: Hybrid (Semantic + BM25 + RRF) — **đang dùng**
   - Config B: Dense-only (chỉ Semantic, không BM25, không RRF)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải thiện

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## ✅ Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên — ✅ Xong
2. **Demo hoạt động được** trong buổi trình bày (chạy local) — ✅ Streamlit chạy port 8501
3. **Evaluation pipeline** chạy được và có báo cáo kết quả — ⏳ Role 4 (CP5)
4. **Code push lên repository** chung của nhóm — ✅ Local git
5. **README** mô tả kiến trúc và phân công — ✅ File này

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                    USER (trình duyệt web)                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│         STREAMLIT UI (src/Role_3_FrontendChatbot/chatbot_ui.py)  │
│  - Dark theme + gradient gold + custom CSS                       │
│  - Suggested questions, chat history, source cards               │
└────────────────────────────┬─────────────────────────────────────┘
                             │ generate_with_citation(query)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│       SUPERVISOR (src/Role_1_TeamLeader/supervisor.py) — BONUS   │
│  ┌─────────────┐                                                 │
│  │Router (LLM) │──hybrid──→ Hybrid RAG Worker (Task 9+10)        │
│  │  (phân loại)│──pgidx──→ PageIndex Worker (Task 8)            │
│  │             │──chat ──→ Chat Worker (LLM only)                │
│  └─────────────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │   Critic    │ → verify length/citation/relevance → retry     │
│  └─────────────┘                                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│            RETRIEVAL PIPELINE (Task 9)                            │
│  ┌──────────────┐   ┌──────────────┐                             │
│  │ Semantic (T5)│ + │ BM25 (T6)    │ → RRF (k=60, T7)            │
│  └──────────────┘   └──────────────┘                             │
│         │                                                        │
│         ▼ (nếu best cosine < 0.48)                               │
│  ┌──────────────────────────────┐                                 │
│  │ PageIndex Fallback (Task 8)  │                                 │
│  └──────────────────────────────┘                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │ chunks (content, metadata, score)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│   GENERATION (Task 10) — OpenAI GPT-4o-mini (fallback: Llama 3.3)│
│  - Reorder (anti lost-in-the-middle)                             │
│  - Format context: [Document N | Source: file | Type: legal]    │
│  - Prompt pháp lý khởi nghiệp + rule "KHÔNG bịa đặt"            │
│  - Output: answer có citation [file, type]                       │
└──────────────────────────────────────────────────────────────────┘
```

**Tham số đã calibrate:**
- Chunk size: 800, overlap: 100
- Embedding: BAAI/bge-m3 (1024 dim, multilingual)
- Vector store: ChromaDB (cosine, persistent local)
- RRF k: 60
- Fallback threshold: cosine < 0.48

---

## 👥 Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ chính | Thư mục làm việc | Task phụ trách | Trạng thái |
|-----------|------|----------------|------------------|----------------|------------|
| 👑 **Lại Thế Rin** | 2A202601665 | Team Leader & RAG Architect | `src/Role_1_TeamLeader/` | Review + `app.py` + `supervisor.py` + thuyết trình | ✅ 100% |
| ⚙️ **Cao Thị Thu Trang** | 2A202601885 | Data & Retrieval Specialist | `src/Role_2_DataRetrieval/` | Task 1, 4, 7, 9 | ✅ 100% |
| 🎨 **Trần Dương Tuấn** | 2A202601271 | Frontend & Chatbot Developer | `src/Role_3_FrontendChatbot/` | Task 2, 5, 8, 10 + UI | ✅ 100% |
| 📊 **Trương Thảo Nguyên** | 2A2026013589 | Evaluation & QA Engineer | `src/Role_4_EvaluationQA/` + `group_project/evaluation/` | Task 3, 6 + DeepEval | ✅ 100% |

### Chi tiết theo từng thành viên

#### 👑 Role 1 — Lại Thế Rin (Team Leader)
- Setup môi trường chung (CP0): `.venv`, `.env`, `requirements_core.txt`
- Code **Task 5** (Semantic Search), **Task 7** (Reranking RRF/MMR/CrossEncoder), **Task 8** (PageIndex Fallback), **Task 9** (Retrieval Pipeline — trái tim), **Task 10** (Generation + Citation)
- Implement **Supervisor Pattern** (bonus): Router + 4 Workers + Critic + retry
- **Review** chunking params (CHUNK_SIZE=800, OVERLAP=100), embedding BGE-M3, RRF k=60
- Thuyết trình CP6

#### ⚙️ Role 2 — Cao Thị Thu Trang (Data & Retrieval)
- **Task 1**: Crawl ≥3 file PDF pháp lý → `data/landing/legal/` (4 files: Luật Doanh nghiệp 2020, Nghị định 52/2013, TikTok Shop Seller Terms, Hướng dẫn thuế hộ KD)
- **Task 4**: Chunking (recursive, 800/100) + embed (BGE-M3) + index ChromaDB → 1,497 chunks
- **Task 7**: Reranking (RRF k=60) + MMR diversity + Cross-encoder fallback
- **Task 9**: Hybrid retrieval pipeline (semantic + BM25 + PageIndex fallback cosine<0.48)

#### 🎨 Role 3 — Trần Dương Tuấn (Frontend)
- **Task 2**: Crawl ≥5 file JSON tin tức từ thuvienphapluat.vn → `data/landing/news/` (10 files)
- **Task 5**: Semantic search với BGE-M3 (Task 4 + này là integration)
- **Task 8**: PageIndex Vectorless RAG (graceful fallback khi không có API key)
- **Task 10**: Generation + Citation (custom prompt cho pháp lý khởi nghiệp & TMĐT)
- **UI**: Streamlit dark theme + custom CSS, hero section, source cards color-coded, retrieval badge, export markdown

#### 📊 Role 4 — Trương Thảo Nguyên (Evaluation)
- **Task 3**: Convert PDF sang Markdown bằng markitdown → `data/standardized/`
- **Task 6**: BM25 Lexical Search (k1=1.5, b=0.75)
- **DeepEval pipeline**: 15+ Q&A golden dataset, 4 metrics (Faithfulness, Answer Relevance, Context Recall, Context Precision), A/B comparison (hybrid vs dense-only)
- Báo cáo `results.md` + đề xuất cải thiện

### Trạng thái tests (CP4)

```
tests/test_individual.py: 34 PASSED + 1 SKIPPED (35/35 mục tiêu)

✅ Task 1 — 3/3 PASSED
✅ Task 2 — 4/4 PASSED
✅ Task 3 — 4/4 PASSED
✅ Task 4 — 4/4 PASSED
✅ Task 5 — 4/4 PASSED
✅ Task 6 — 3/4 PASSED + 1 SKIPPED
✅ Task 7 — 3/3 PASSED
✅ Task 8 — 2/2 PASSED
✅ Task 9 — 4/4 PASSED
✅ Task 10 — 3/3 PASSED
```

---

## 🚀 Hướng Dẫn Chạy

### Cài đặt

```bash
# Clone repo
git clone <repo_url>
cd K3-Day08-RAG-Pipeline-E403-B3-2

# Tạo virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # macOS/Linux

# Cài dependencies
pip install -r requirements_core.txt

# Điền API key vào .env
# OPENAI_API_KEY=sk-proj-...
# Mở file .env và uncomment + điền key của bạn
```

### Chạy pipeline

```bash
# Bước 1: Convert PDF → Markdown (nếu chưa có data/standardized/)
python -m src.Role_2_DataRetrieval.task3_markdown_converter

# Bước 2: Chunking + Index vào ChromaDB (~2-3 phút lần đầu do load model)
python -m src.Role_2_DataRetrieval.task4_chunking_indexing

# Bước 3: Test individual tasks
python -m pytest tests/test_individual.py -v

# Bước 4: Chạy UI
streamlit run app.py
# → Mở browser http://localhost:8501
```

### Chạy RAG pipeline trực tiếp (không cần UI)

```bash
# Cách 1: Task 10 thường (hybrid + LLM)
python -c "from src.Role_3_FrontendChatbot.task10_generation import generate_with_citation; r = generate_with_citation('Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?'); print(r['answer'])"

# Cách 2: Supervisor (bonus - router + 4 workers)
python -m src.Role_1_TeamLeader.supervisor
```

---

## 📊 Kết Quả Hiện Tại (Demo)

**Dataset đã index:**
- 4 văn bản pháp lý (Luật Doanh nghiệp 2020, Nghị định 52/2013, TikTok Shop Seller Terms, Hướng dẫn thuế hộ KD)
- 10 bài tin tức về khởi nghiệp/TMĐT
- **Tổng: 1,497 chunks** (size 800, overlap 100)

**Câu hỏi demo** (đã test thật):
1. *"Bán hàng trên TikTok Shop có phải nộp thuế không?"* → trả lời có citation `[tiktok-shop-seller-terms.md, legal]`
2. *"Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì?"* → trả lời có citation `[ecommerce_startup_02_..., news]`
3. *"Điều kiện thành lập công ty TNHH một thành viên?"* → trả lời có citation `[luat-doanh-nghiep-2020.md, legal]`

**Kiến trúc đã hoàn thiện:**
- ✅ Hybrid Retrieval (BGE-M3 + BM25)
- ✅ RRF Reranking (k=60)
- ✅ PageIndex Fallback (graceful khi không có API key)
- ✅ Generation có citation (OpenAI GPT-4o-mini, fallback Llama 3.3 70B free)
- ✅ Supervisor + 4 Workers + Critic (bonus multi-agent)
- ✅ UI Streamlit dark theme + custom CSS

---

## 📝 Lưu Ý

1. **API key**: Cần `OPENAI_API_KEY` (đã có sẵn) hoặc `OPENROUTER_API_KEY` (free → nhiều model tiếng Việt tốt). Điền vào `.env` trước khi chạy UI.
2. **PageIndex**: Optional — pipeline vẫn hoạt động nếu không có `PAGEINDEX_API_KEY` (fallback graceful trả []).
3. **Jina Reranker**: Optional — Task 7 mặc định dùng RRF; có `JINA_API_KEY` thì dùng cross-encoder với chất lượng tốt hơn.
4. **CUDA**: PyTorch cài bản CPU-only phù hợp với máy phổ thông. Encode 1,497 chunks mất ~3-5 phút trên CPU; có GPU sẽ nhanh hơn ~8 lần.
5. **Track tiếp theo**: Sau K3, nhóm sẽ phát triển lên **knowledge graph** để khắc phục các câu hỏi "hóc búa" cần suy luận đa bước.

---

## 📂 Cấu trúc Repo

```
K3-Day08-RAG-Pipeline-E403-B3-2/
├── README.md                          # Repo README
├── LAB_GUIDE.md                       # Hướng dẫn từng checkpoint
├── requirements_core.txt              # Python dependencies
├── .env                               # API keys (KHÔNG push lên git)
├── .gitignore                         # Ignore venv/, chroma_db/, .env
├── app.py                             # Entry point gọi chatbot_ui
├── data/
│   ├── landing/
│   │   ├── legal/                     # 4 PDF pháp lý (Task 1)
│   │   └── news/                      # 10 JSON tin tức (Task 2)
│   └── standardized/
│       ├── legal/                     # 4 file .md
│       └── news/                      # 10 file .md
├── chroma_db/                         # Vector store (sau Task 4)
├── src/
│   ├── Role_1_TeamLeader/
│   │   ├── __init__.py
│   │   ├── README.md                  # Checklist CP0-CP6
│   │   └── supervisor.py              # Multi-agent router (bonus)
│   ├── Role_2_DataRetrieval/
│   │   ├── task3_markdown_converter.py
│   │   ├── task4_chunking_indexing.py
│   │   ├── task7_reranking.py         # RRF + MMR + CrossEncoder
│   │   └── task9_retrieval_pipeline.py # Trái tim pipeline
│   ├── Role_3_FrontendChatbot/
│   │   ├── task5_semantic_search.py
│   │   ├── task8_pageindex_vectorless.py
│   │   ├── task10_generation.py       # Citation + LLM call
│   │   └── chatbot_ui.py              # Streamlit UI đẹp
│   └── Role_4_EvaluationQA/
│       └── task6_lexical_search.py    # BM25
├── tests/
│   └── test_individual.py             # 35 tests (Task 1-10)
└── group_project/
    ├── README.md                      # File này
    └── evaluation/
        ├── golden_dataset.json        # Role 4 (CP5)
        ├── eval_pipeline.py           # Role 4
        └── results.md                 # Role 4
```
