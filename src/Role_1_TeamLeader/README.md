# Role 1 — Team Leader & RAG Architect

**Người phụ trách:** Lại Thế Rin — 2A202601665

Thư mục này không chứa file task vì Role 1 là vai trò **điều phối và kiểm duyệt kỹ thuật**,
không sở hữu Task nào trong `src/`. Sản phẩm bàn giao của Role 1 là các quyết định kiến trúc
và kết quả review dưới đây.

## Checklist theo từng Checkpoint

| Checkpoint | Nhiệm vụ | Bằng chứng hoàn thành |
| :--- | :--- | :--- |
| **CP0** (0:00–0:10) | Khởi tạo repo chung, chia sẻ `.env` (`OPENROUTER_API_KEY`) cho cả nhóm | Cả 4 thành viên clone & import được `chromadb`, `sentence_transformers` |
| **CP1** (0:10–0:35) | Phân công nguồn dữ liệu, tránh trùng tài liệu giữa các thành viên | Danh sách URL/tài liệu đã chia, không có file trùng trong `data/landing/` |
| **CP2** (0:35–1:00) | Duyệt tham số `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, embedding `BAAI/bge-m3` | Xác nhận trong `Role_2_DataRetrieval/task4_chunking_indexing.py` |
| **CP3** (1:00–1:20) | Duyệt công thức RRF `k=60`, cân bằng Semantic vs BM25 | Xác nhận trong `Role_2_DataRetrieval/task7_reranking.py` |
| **CP4** (1:20–1:45) | Chạy `pytest tests/test_individual.py -v`, xác nhận cả nhóm đủ điểm | Ảnh chụp kết quả pytest |
| **CP5** (1:45–2:15) | Ghép code tốt nhất của nhóm vào `app.py`, theo dõi tiến độ báo cáo | `app.py` chạy được, `group_project/evaluation/results.md` hoàn thiện |
| **CP6** (2:15–3:00) | Thuyết trình kiến trúc RAG Pipeline + demo Chatbot (5–8 phút) | Slide/demo trước lớp |

## Sơ đồ phụ thuộc giữa các Role

Role 1 cần nắm rõ luồng gọi hàm liên-thư-mục để merge code không bị vỡ:

```
Role_2/task9_retrieval_pipeline.py
    ├── Role_3/task5_semantic_search.py        (semantic_search)
    ├── Role_4/task6_lexical_search.py         (lexical_search)
    ├── Role_2/task7_reranking.py              (rerank, rerank_rrf)
    └── Role_3/task8_pageindex_vectorless.py   (pageindex_search)

Role_3/task10_generation.py
    └── Role_2/task9_retrieval_pipeline.py     (retrieve)

app.py
    └── Role_3/task10_generation.py            (generate_with_citation)
```

> ⚠️ Task 9 phụ thuộc vào cả 3 role còn lại — đây là điểm nghẽn tích hợp. Role 1 cần đảm bảo
> Task 5, 6, 7, 8 xong trước khi Role 2 bắt đầu ghép Task 9 ở CP4.
