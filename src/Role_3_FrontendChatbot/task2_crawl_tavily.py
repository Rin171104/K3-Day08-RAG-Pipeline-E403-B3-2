"""
Task 2 (biến thể Tavily) — Thu thập bài viết pháp lý về khởi nghiệp trong thương mại điện tử.

Dùng Tavily Search API thay cho Crawl4AI, giới hạn nguồn ở thuvienphapluat.vn.
Lý do chọn Tavily: không cần cài Playwright/Chromium (~400MB), API tự trích xuất
nội dung sạch từ trang, và tự lọc theo domain nên không crawl lan sang trang khác.

Cài đặt:
    Chỉ cần `requests` (đã có trong requirements.txt) — module này gọi thẳng REST API,
    không cần thêm package `tavily-python`.

    Thêm vào .env:
        TAVILY_API_KEY=tvly-...

Chạy:
    python -m src.Role_3_FrontendChatbot.task2_crawl_tavily

Output:
    data/landing/news/ecommerce_startup_XX.json — mỗi bài 1 file, cùng schema với
    task2_crawl_news.py để Task 3 (Role 4) convert sang Markdown được ngay:
        {"url", "title", "date_crawled", "content_markdown", ...}
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "landing" / "news"

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Chỉ lấy kết quả từ thuvienphapluat.vn — Tavily lọc ở phía server nên không tốn
# request cho các domain khác.
INCLUDE_DOMAINS = ["thuvienphapluat.vn"]

# Nhiều truy vấn con thay vì một truy vấn chung: mỗi truy vấn nhắm một khía cạnh
# pháp lý khác nhau của khởi nghiệp TMĐT, cho corpus đa dạng hơn và tăng khả năng
# BM25 (Task 6) khớp được từ khoá chuyên ngành.
SEARCH_QUERIES = [
    "quy định đăng ký kinh doanh thương mại điện tử",
    "điều kiện thành lập website thương mại điện tử bán hàng",
    "nghị định quản lý hoạt động thương mại điện tử",
    "thủ tục đăng ký sàn giao dịch thương mại điện tử với Bộ Công Thương",
    "chính sách hỗ trợ doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo",
    "nghĩa vụ thuế của cá nhân kinh doanh trên sàn thương mại điện tử",
    "bảo vệ quyền lợi người tiêu dùng trong giao dịch thương mại điện tử",
]

RESULTS_PER_QUERY = 3      # Tavily trả tối đa số này cho mỗi truy vấn
MIN_ARTICLES = 5           # Ngưỡng pass của Task 2 theo tests/test_individual.py
MIN_CONTENT_LENGTH = 200   # Bỏ bài quá ngắn (trang lỗi, trang chuyển hướng)
REQUEST_TIMEOUT = 60       # giây
RETRY_DELAY = 3            # giây, chờ giữa hai lần thử lại


# =============================================================================
# HELPERS
# =============================================================================

def _get_api_key() -> str:
    """Đọc TAVILY_API_KEY, báo lỗi rõ ràng nếu thiếu."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu TAVILY_API_KEY trong .env.\n"
            "Lấy key miễn phí tại https://app.tavily.com rồi thêm dòng:\n"
            "    TAVILY_API_KEY=tvly-..."
        )
    return api_key


def _slugify(text: str, max_length: int = 60) -> str:
    """Chuyển tiêu đề tiếng Việt thành slug an toàn cho tên file trên Windows."""
    text = text.lower()
    # Bỏ dấu tiếng Việt bằng bảng thay thế đơn giản (đủ dùng cho tên file)
    replacements = {
        "[àáạảãâầấậẩẫăằắặẳẵ]": "a", "[èéẹẻẽêềếệểễ]": "e", "[ìíịỉĩ]": "i",
        "[òóọỏõôồốộổỗơờớợởỡ]": "o", "[ùúụủũưừứựửữ]": "u", "[ỳýỵỷỹ]": "y",
        "đ": "d",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_length] or "bai-viet"


def _post(url: str, payload: dict, api_key: str) -> dict:
    """Gọi Tavily API, thử lại 1 lần nếu gặp lỗi mạng hoặc rate limit."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in (1, 2):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 429:
                print(f"    ! Rate limit, chờ {RETRY_DELAY}s rồi thử lại…")
                time.sleep(RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f"    ! Lỗi mạng ({exc}), thử lại sau {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)

    return {}


# =============================================================================
# CRAWLING
# =============================================================================

def search_articles(api_key: str) -> list[dict]:
    """
    Chạy toàn bộ SEARCH_QUERIES qua Tavily, gộp kết quả và khử trùng lặp theo URL.

    Returns:
        List of {"url", "title", "content", "score", "query"}
    """
    seen_urls: set[str] = set()
    articles: list[dict] = []

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"[{i}/{len(SEARCH_QUERIES)}] Tìm: {query}")

        payload = {
            "query": query,
            "search_depth": "advanced",   # phân tích sâu hơn, trả nội dung dài hơn
            "include_domains": INCLUDE_DOMAINS,
            "include_raw_content": True,  # lấy full text thay vì chỉ đoạn tóm tắt
            "max_results": RESULTS_PER_QUERY,
        }

        try:
            data = _post(TAVILY_SEARCH_URL, payload, api_key)
        except requests.RequestException as exc:
            print(f"    ✗ Bỏ qua truy vấn này: {exc}")
            continue

        results = data.get("results", [])
        print(f"    → {len(results)} kết quả")

        for item in results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # raw_content đầy đủ hơn content (vốn chỉ là đoạn trích liên quan)
            content = item.get("raw_content") or item.get("content") or ""
            if len(content) < MIN_CONTENT_LENGTH:
                print(f"    - Bỏ (nội dung {len(content)} ký tự, quá ngắn): {url}")
                continue

            articles.append({
                "url": url,
                "title": item.get("title", "Không rõ tiêu đề"),
                "content": content,
                "score": item.get("score", 0.0),
                "query": query,
            })

    return articles


def save_articles(articles: list[dict]) -> list[Path]:
    """Ghi mỗi bài thành 1 file JSON trong data/landing/news/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for i, article in enumerate(articles, 1):
        record = {
            "url": article["url"],
            "title": article["title"],
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": article["content"],
            # Metadata bổ sung — Task 4 dùng làm metadata chunk, Task 10 dùng để cite
            "source": "thuvienphapluat.vn",
            "topic": "khoi-nghiep-thuong-mai-dien-tu",
            "type": "legal_news",
            "search_query": article["query"],
            "relevance_score": article["score"],
        }

        filename = f"ecommerce_startup_{i:02d}_{_slugify(article['title'])}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        saved.append(filepath)
        print(f"  ✓ [{i:02d}] {article['title'][:70]}")
        print(f"       {len(article['content']):,} ký tự → {filename}")

    return saved


def crawl_all() -> list[Path]:
    """Pipeline đầy đủ: search → lọc → lưu JSON."""
    api_key = _get_api_key()

    print("=" * 70)
    print("TASK 2 — Crawl pháp lý khởi nghiệp TMĐT qua Tavily")
    print(f"Nguồn: {', '.join(INCLUDE_DOMAINS)}")
    print(f"Đích:  {DATA_DIR}")
    print("=" * 70)

    articles = search_articles(api_key)

    if not articles:
        print("\n✗ Không thu được bài nào.")
        print("  Kiểm tra: TAVILY_API_KEY còn hạn mức? Kết nối mạng ổn không?")
        return []

    # Ưu tiên bài có điểm liên quan cao nhất
    articles.sort(key=lambda a: a["score"], reverse=True)

    print(f"\nThu được {len(articles)} bài (đã khử trùng lặp). Đang lưu…\n")
    saved = save_articles(articles)

    print(f"\n{'=' * 70}")
    if len(saved) >= MIN_ARTICLES:
        print(f"✓ HOÀN THÀNH — {len(saved)} bài (yêu cầu tối thiểu {MIN_ARTICLES})")
        print("  Bước tiếp theo: Role 4 chạy Task 3 để convert sang Markdown.")
    else:
        print(f"⚠ Mới có {len(saved)}/{MIN_ARTICLES} bài — chưa đạt ngưỡng Task 2.")
        print("  Cách khắc phục: thêm truy vấn vào SEARCH_QUERIES hoặc tăng RESULTS_PER_QUERY.")
    print("=" * 70)

    return saved


if __name__ == "__main__":
    crawl_all()
