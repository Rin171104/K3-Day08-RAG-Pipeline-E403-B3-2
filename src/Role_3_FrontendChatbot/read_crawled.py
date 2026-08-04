"""
Công cụ đọc dữ liệu đã crawl (Task 2).

Nội dung bài nằm trong field `content_markdown` của file JSON, thường dài vài chục
nghìn ký tự trên một dòng nên mở thẳng bằng editor rất khó đọc. Script này liệt kê,
xem nhanh, hoặc xuất ra file .md để đọc bằng Markdown preview.

Chạy:
    python -m src.Role_3_FrontendChatbot.read_crawled              # liệt kê tất cả
    python -m src.Role_3_FrontendChatbot.read_crawled 6            # xem bài số 6
    python -m src.Role_3_FrontendChatbot.read_crawled 6 --full     # xem toàn bộ bài 6
    python -m src.Role_3_FrontendChatbot.read_crawled --export     # xuất tất cả ra .md
"""

import json
import sys
from pathlib import Path

NEWS_DIR = Path(__file__).parent.parent.parent / "data" / "landing" / "news"
EXPORT_DIR = Path(__file__).parent.parent.parent / "data" / "preview"

PREVIEW_CHARS = 1500


def load_articles() -> list[tuple[Path, dict]]:
    """Đọc toàn bộ JSON trong data/landing/news/, bỏ qua file hỏng."""
    articles = []
    for path in sorted(NEWS_DIR.glob("*.json")):
        try:
            # encoding='utf-8' bắt buộc: Windows mặc định cp1252, gặp tiếng Việt sẽ lỗi
            articles.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"⚠ Bỏ qua {path.name}: {exc}")
    return articles


def list_articles() -> None:
    """In bảng tổng quan các bài đã crawl."""
    articles = load_articles()
    if not articles:
        print(f"Chưa có bài nào trong {NEWS_DIR}")
        print("Chạy: python -m src.Role_3_FrontendChatbot.task2_crawl_tavily")
        return

    print(f"{'#':>3}  {'Tiêu đề':<58} {'Ký tự':>9}")
    print("-" * 74)
    for i, (_, doc) in enumerate(articles, 1):
        title = doc.get("title", "(không tiêu đề)")[:56]
        print(f"{i:>3}  {title:<58} {len(doc.get('content_markdown', '')):>9,}")
    print("-" * 74)
    print(f"Tổng: {len(articles)} bài")
    print("\nXem chi tiết: python -m src.Role_3_FrontendChatbot.read_crawled <số>")


def show_article(index: int, full: bool = False) -> None:
    """In nội dung một bài theo số thứ tự trong bảng."""
    articles = load_articles()
    if not 1 <= index <= len(articles):
        print(f"Số bài phải trong khoảng 1–{len(articles)}")
        return

    path, doc = articles[index - 1]
    content = doc.get("content_markdown", "")

    print("=" * 74)
    print(f"[{index}] {doc.get('title', '')}")
    print("=" * 74)
    print(f"URL       : {doc.get('url', '')}")
    print(f"Nguồn     : {doc.get('source', '')}")
    print(f"Truy vấn  : {doc.get('search_query', '')}")
    print(f"Độ dài    : {len(content):,} ký tự")
    print(f"File      : {path.name}")
    print("-" * 74)

    if full or len(content) <= PREVIEW_CHARS:
        print(content)
    else:
        print(content[:PREVIEW_CHARS])
        print(f"\n… còn {len(content) - PREVIEW_CHARS:,} ký tự. Thêm --full để xem hết.")


def export_markdown() -> None:
    """Xuất mỗi bài thành 1 file .md để đọc bằng Markdown preview."""
    articles = load_articles()
    if not articles:
        print("Chưa có bài nào để xuất.")
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for i, (path, doc) in enumerate(articles, 1):
        md = (
            f"# {doc.get('title', '')}\n\n"
            f"- **Nguồn:** {doc.get('source', '')}\n"
            f"- **URL:** {doc.get('url', '')}\n"
            f"- **Crawl lúc:** {doc.get('date_crawled', '')}\n"
            f"- **Truy vấn:** {doc.get('search_query', '')}\n\n"
            "---\n\n"
            f"{doc.get('content_markdown', '')}\n"
        )
        out = EXPORT_DIR / f"{path.stem}.md"
        out.write_text(md, encoding="utf-8")
        print(f"  ✓ [{i:02d}] {out.name}")

    print(f"\nĐã xuất {len(articles)} file vào {EXPORT_DIR}")
    print("Mở file .md trong VS Code rồi bấm Ctrl+Shift+V để xem bản render.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]

    if "--export" in args:
        export_markdown()
    elif args and args[0].isdigit():
        show_article(int(args[0]), full="--full" in args)
    else:
        list_articles()
