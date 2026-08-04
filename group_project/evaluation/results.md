# RAG Evaluation Results

## Framework sử dụng

> **Offline metrics (deterministic)** — không dùng LLM judge để tránh rate-limit OpenRouter.
> Implementation: [`eval_pipeline.py`](eval_pipeline.py).
>
> Lý do: OpenRouter key placeholder (`sk-or-v1-xxx...`) và quota free bị giới hạn 50 req/ngày,
> RAGAS/DeepEval cần 4-8 LLM calls / câu hỏi → 15 câu × 4 metric × 2 config = 120+ LLM calls,
> vượt quota. Các metrics ở đây dùng:
> - **Token overlap (Jaccard + Containment + Bigram)** cho Answer Relevance
> - **Claim-level token support** cho Faithfulness (tách answer thành câu, check từng câu)
> - **Claim-level coverage** cho Context Recall (tách ground_truth, check từng claim)
> - **Heuristic relevance threshold** (overlap ≥ 10%) cho Context Precision
>
> Có thể nâng cấp lên RAGAS/DeepEval khi có key paid.

---

## Overall Scores

| Metric | Config A (hybrid + rerank + fallback) | Config B (dense-only) | Δ (A − B) |
|--------|----------------------:|---------------------:|----------:|
| Faithfulness | 0.9593 | 0.9732 | -0.0139 |
| Answer Relevance | 0.4296 | 0.4420 | -0.0124 |
| Context Recall | 1.0000 | 0.8802 | +0.1198 |
| Context Precision | 0.9700 | 0.9400 | +0.0300 |
| **Average** | 0.8397 | 0.8088 | +0.0309 |

**Số câu hỏi golden dataset:** 20
**Số câu không retrieve được context:** A = 0 | B = 0

---

## A/B Comparison Analysis

**Config A — Hybrid + RRF Rerank + PageIndex Fallback:**
> Pipeline: `semantic_search + lexical_search → RRF rerank (k=60) → PageIndex fallback nếu max cosine < 0.48`.
> Đây là cấu hình mặc định của `task9_retrieval_pipeline.retrieve()`.
> Ưu điểm: tận dụng cả semantic similarity (BAAI/bge-m3) và BM25 keyword matching, có fallback
> khi retrieval yếu.

**Config B — Dense-only:**
> Pipeline: chỉ dùng `semantic_search()` (BAAI/bge-m3, top-k=5), KHÔNG rerank, KHÔNG fallback.
> Đây là baseline tối thiểu — semantic search thuần.

**Kết luận:**
> **Winner: Config A** (Δ_avg = +0.0309).
>
> Hybrid search + rerank + fallback THẮNG. Bằng chứng: context_recall và context_precision đều cao hơn dense-only, cho thấy việc kết hợp BM25 lexical search giúp retrieve được các chunk chứa từ khoá pháp lý cụ thể (số hiệu nghị định, tên văn bản) mà semantic search đơn thuần bỏ sót.

---

## Worst Performers (Bottom 3 trên Config A)

| # | Question (rút gọn) | Faithfulness | Relevance | Recall | Precision | Avg | Root Cause |
|---|---------------------|-------------:|----------:|-------:|----------:|----:|------------|
| 1 | Bán hàng trên TikTok Shop có phải nộp thuế không? Ai chịu tr... | 0.958 | 0.231 | 1.000 | 0.400 | 0.647 | answer miss keyword từ question |
| 2 | Công ty cổ phần khác công ty TNHH ở điểm nào cơ bản? | 0.904 | 0.232 | 1.000 | 1.000 | 0.784 | answer miss keyword từ question |
| 3 | Điều 78 Luật Doanh nghiệp 2020 quy định về vấn đề gì? | 0.921 | 0.323 | 1.000 | 1.000 | 0.811 | câu hỏi có terminology chuyên ngành khó match |

---

## Recommendations

### Cải tiến 1
**Hybrid search cải thiện context recall đáng kể** (Δ=+0.1198). Đây là bằng chứng rõ ràng về giá trị của việc kết hợp BM25 + semantic search.

### Cải tiến 2
**Context precision không có nhiều khác biệt** giữa 2 configs. Có thể do nhiều chunk retrieved chứa từ khoá liên quan (BM25 kéo về các đoạn có từ khoá văn bản pháp lý chung như 'hộ kinh doanh', 'thuế'). Cần đánh giá thêm: phân biệt giữa 'relevant về từ khoá' và 'relevant về nội dung trả lời'.

### Cải tiến 3
**Cải thiện evaluation methodology**: Hiện tại dùng offline metrics (token overlap). Khi có API key paid, nâng cấp lên RAGAS với LLM-judge để đo faithfulness chi tiết hơn (tách claim-level thay vì token-level).


---

## Chi tiết từng câu hỏi

| # | Question | F | AR | CR | CP | #Ctx | Source |
|--:|----------|--:|---:|---:|---:|----:|--------|
| 1 | Điều kiện thành lập công ty TNHH một thành viên theo Luật Do... | 0.887 | 0.423 | 1.000 | 1.000 | 5 | hybrid |
| 2 | Hồ sơ đăng ký hộ kinh doanh cá thể cần những giấy tờ gì? | 0.923 | 0.529 | 1.000 | 1.000 | 5 | hybrid |
| 3 | Bán hàng trên TikTok Shop có phải nộp thuế không? Ai chịu tr... | 0.958 | 0.231 | 1.000 | 0.400 | 5 | hybrid |
| 4 | Ngưỡng doanh thu nào quyết định hộ kinh doanh phải nộp thuế ... | 0.997 | 0.369 | 1.000 | 1.000 | 5 | hybrid |
| 5 | Thuế suất thuế TNCN cho hộ kinh doanh có doanh thu năm trên ... | 0.993 | 0.447 | 1.000 | 1.000 | 5 | hybrid |
| 6 | Công ty cổ phần khác công ty TNHH ở điểm nào cơ bản? | 0.904 | 0.232 | 1.000 | 1.000 | 5 | hybrid |
| 7 | Quyền và trách nhiệm của hộ kinh doanh, cá nhân kinh doanh t... | 0.915 | 0.550 | 1.000 | 1.000 | 5 | hybrid |
| 8 | Doanh nghiệp nhỏ và vua được hưởng chính sách ưu đãi gì? | 1.000 | 0.474 | 1.000 | 1.000 | 5 | hybrid |
| 9 | Kinh doanh online trên các sàn thương mại điện tử có cần đăn... | 0.942 | 0.550 | 1.000 | 1.000 | 5 | hybrid |
| 10 | Điều kiện hoạt động thương mại điện tử tại Việt Nam theo Ngh... | 0.915 | 0.414 | 1.000 | 1.000 | 5 | hybrid |
| 11 | Tiêu chí xác định doanh nghiệp nhỏ và vua mới nhất là gì? | 0.998 | 0.538 | 1.000 | 1.000 | 5 | hybrid |
| 12 | Doanh nghiệp khởi nghiệp sáng tạo là gì? Có chính sách hỗ tr... | 1.000 | 0.544 | 1.000 | 1.000 | 5 | hybrid |
| 13 | Khi nào hộ kinh doanh phải thực hiện quyết toán thuế TNCN th... | 1.000 | 0.466 | 1.000 | 1.000 | 5 | hybrid |
| 14 | Các khoản chi nào được trừ khi xác định thu nhập chịu thuế T... | 1.000 | 0.367 | 1.000 | 1.000 | 5 | hybrid |
| 15 | Cá nhân không cư trú kinh doanh trên nền tảng số có phải thô... | 0.938 | 0.453 | 1.000 | 1.000 | 5 | hybrid |
| 16 | Bán hàng trên Shopee/Lazada/Tiki có phải đăng ký hộ kinh doa... | 0.999 | 0.326 | 1.000 | 1.000 | 5 | hybrid |
| 17 | Công ty TNHH hai thành viên trở lên có thể chuyển đổi thành ... | 0.900 | 0.487 | 1.000 | 1.000 | 5 | hybrid |
| 18 | Tỷ lệ % thuế GTGT cho hộ kinh doanh phân phối, cung cấp hàng... | 1.000 | 0.486 | 1.000 | 1.000 | 5 | hybrid |
| 19 | Hộ kinh doanh có được hoàn thuế không? Điều kiện để được hoà... | 0.996 | 0.382 | 1.000 | 1.000 | 5 | hybrid |
| 20 | Điều 78 Luật Doanh nghiệp 2020 quy định về vấn đề gì? | 0.921 | 0.323 | 1.000 | 1.000 | 5 | hybrid |
