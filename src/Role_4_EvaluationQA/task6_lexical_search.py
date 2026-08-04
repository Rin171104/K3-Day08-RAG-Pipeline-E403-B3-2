"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25.

BM25 hoạt động dựa trên:
    - Term Frequency (TF):
        Từ xuất hiện nhiều trong document → điểm cao.
    - Inverse Document Frequency (IDF):
        Từ hiếm → quan trọng hơn.
    - Document length normalization:
        Document dài không bị ưu tiên quá mức.

Formula:

    score(q,d) = Σ IDF(qi) *
        (tf(qi,d) * (k1+1))
        /
        (tf(qi,d) + k1*(1-b+b*|d|/avgdl))

Configuration:
    k1 = 1.5
    b  = 0.75

Output format được thiết kế tương thích với Task 5 Semantic Search:

{
    "content": str,
    "score": float,
    "metadata": dict
}

Pipeline:

    Markdown corpus
          ↓
      load_corpus()
          ↓
      tokenize()
          ↓
      BM25Okapi
          ↓
    lexical_search()
          ↓
      Task 9 Retrieval Pipeline
"""

from pathlib import Path

from rank_bm25 import BM25Okapi


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent

CORPUS_DIR = BASE_DIR / "data" / "standardized"


# =============================================================================
# CORPUS LOADING
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Load toàn bộ Markdown files từ data/standardized/.

    Expected structure:

        data/
        └── standardized/
            ├── legal/
            │   ├── document_1.md
            │   ├── document_2.md
            │   └── ...
            │
            └── news/
                ├── article_1.md
                ├── article_2.md
                └── ...

    Returns:
        List of:

        {
            "content": str,
            "metadata": {
                "source": str,
                "filename": str,
                "category": str
            }
        }
    """

    corpus = []

    if not CORPUS_DIR.exists():
        print(
            f"⚠ Corpus directory không tồn tại: "
            f"{CORPUS_DIR}"
        )
        return corpus

    # rglob giúp scan cả legal/, news/ và các thư mục con.
    markdown_files = sorted(
        CORPUS_DIR.rglob("*.md")
    )

    print(
        f"Loading corpus from: {CORPUS_DIR}"
    )

    for filepath in markdown_files:

        try:

            content = filepath.read_text(
                encoding="utf-8"
            ).strip()

            # Bỏ qua file rỗng.
            if not content:
                continue

            # Category chính là tên thư mục chứa file.
            category = filepath.parent.name

            metadata = {
                "source": str(filepath),
                "filename": filepath.name,
                "category": category,
                "type": category,
            }

            corpus.append(
                {
                    "content": content,
                    "metadata": metadata,
                }
            )

        except Exception as e:

            print(
                f"⚠ Không thể đọc "
                f"{filepath}: {e}"
            )

    return corpus


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize text cho BM25.

    Hiện tại sử dụng whitespace tokenizer.

    Ví dụ:

        "Luật Doanh nghiệp 2020"

    →

        ["luật", "doanh", "nghiệp", "2020"]

    Lưu ý:
        Đây là tokenizer đơn giản.
        Có thể nâng cấp bằng underthesea / pyvi
        nếu muốn cải thiện lexical retrieval tiếng Việt.
    """

    if not text:
        return []

    return text.lower().split()


# =============================================================================
# LOAD CORPUS
# =============================================================================

# Load corpus một lần khi module được import.
#
# Giống pattern lazy/singleton của Task 5:
# không đọc lại toàn bộ Markdown mỗi lần search.

CORPUS: list[dict] = load_corpus()


# =============================================================================
# BUILD BM25 INDEX
# =============================================================================

def build_bm25_index(
    corpus: list[dict],
) -> BM25Okapi | None:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus:
            List of {
                "content": str,
                "metadata": dict
            }

    Returns:
        BM25Okapi index.

        Nếu corpus rỗng → return None.
    """

    if not corpus:

        print(
            "⚠ Không thể build BM25 index: "
            "corpus đang rỗng."
        )

        return None

    # Tokenize toàn bộ documents.
    tokenized_corpus = [
        tokenize(doc["content"])
        for doc in corpus
    ]

    # BM25 configuration.
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=1.5,
        b=0.75,
    )

    return bm25


# =============================================================================
# BM25 SINGLETON
# =============================================================================

# Build index một lần.
#
# Các lần gọi lexical_search() sau đó
# sẽ sử dụng lại index này.

BM25_INDEX = build_bm25_index(CORPUS)


# =============================================================================
# LEXICAL SEARCH
# =============================================================================

def lexical_search(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query:
            Câu truy vấn tiếng Việt/Anh.

        top_k:
            Số lượng kết quả tối đa.

    Returns:
        List of:

        {
            "content": str,
            "score": float,
            "metadata": dict
        }

        Sorted by score descending.

    Example:

        results = lexical_search(
            "tuition fee payment methods",
            top_k=5
        )
    """

    # -------------------------------------------------------------------------
    # VALIDATE QUERY
    # -------------------------------------------------------------------------

    if not query or not query.strip():
        return []

    # -------------------------------------------------------------------------
    # VALIDATE TOP_K
    # -------------------------------------------------------------------------

    if top_k <= 0:
        return []

    # -------------------------------------------------------------------------
    # CHECK CORPUS
    # -------------------------------------------------------------------------

    if not CORPUS:

        print(
            "⚠ Corpus đang rỗng. "
            f"Kiểm tra: {CORPUS_DIR}"
        )

        return []

    # -------------------------------------------------------------------------
    # CHECK INDEX
    # -------------------------------------------------------------------------

    if BM25_INDEX is None:
        return []

    # -------------------------------------------------------------------------
    # TOKENIZE QUERY
    # -------------------------------------------------------------------------

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    # -------------------------------------------------------------------------
    # CALCULATE BM25 SCORES
    # -------------------------------------------------------------------------

    scores = BM25_INDEX.get_scores(
        tokenized_query
    )

    # -------------------------------------------------------------------------
    # RANK DOCUMENTS
    # -------------------------------------------------------------------------

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # BUILD RESULTS
    # -------------------------------------------------------------------------

    results = []

    for idx in ranked_indices:

        score = float(scores[idx])

        # BM25 score <= 0 nghĩa là document
        # không có mức độ match hữu ích.
        if score <= 0:
            continue

        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": round(score, 4),
                "metadata": CORPUS[idx]["metadata"],
            }
        )

        # Đã đủ top_k.
        if len(results) >= top_k:
            break

    return results


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Task 6: Lexical Search (BM25)")
    print("=" * 70)

    print(
        f"\nCorpus directory:"
        f"\n  {CORPUS_DIR}"
    )

    print(
        f"\nDocuments loaded: "
        f"{len(CORPUS)}"
    )

    # -------------------------------------------------------------------------
    # CORPUS STATUS
    # -------------------------------------------------------------------------

    if not CORPUS:

        print(
            "\n⚠ Không có document nào "
            "trong corpus."
        )

        print(
            "\nHãy kiểm tra:"
        )

        print(
            "  data/standardized/legal/"
        )

        print(
            "  data/standardized/news/"
        )

        print(
            "\nNếu chưa có Markdown, "
            "hãy chạy Task 3 trước."
        )

    else:

        # ---------------------------------------------------------------------
        # SHOW CORPUS BREAKDOWN
        # ---------------------------------------------------------------------

        categories = {}

        for doc in CORPUS:

            category = (
                doc["metadata"]
                .get("category", "unknown")
            )

            categories[category] = (
                categories.get(category, 0) + 1
            )

        print("\nCorpus breakdown:")

        for category, count in sorted(
            categories.items()
        ):

            print(
                f"  - {category}: {count} documents"
            )

        # ---------------------------------------------------------------------
        # TEST QUERIES
        # ---------------------------------------------------------------------

        test_queries = [
            "tuition fee payment methods",
            "quyền của doanh nghiệp",
            "nghĩa vụ của doanh nghiệp",
            "điều kiện thành lập công ty",
        ]

        for query in test_queries:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Q: {query}"
            )

            print(
                "-" * 70
            )

            results = lexical_search(
                query,
                top_k=5,
            )

            if not results:

                print(
                    "  ⚠ Không tìm thấy kết quả."
                )

                continue

            for i, result in enumerate(
                results,
                start=1,
            ):

                metadata = (
                    result["metadata"]
                )

                source = metadata.get(
                    "filename",
                    metadata.get(
                        "source",
                        "N/A",
                    ),
                )

                category = metadata.get(
                    "category",
                    "unknown",
                )

                print(
                    f"\n  #{i} "
                    f"[BM25 Score: "
                    f"{result['score']:.4f}]"
                )

                print(
                    f"  Type: {category}"
                )

                print(
                    f"  Source: {source}"
                )

                preview = (
                    result["content"]
                    .replace("\n", " ")
                    .strip()
                )

                print(
                    f"  Content: "
                    f"{preview[:300]}..."
                )