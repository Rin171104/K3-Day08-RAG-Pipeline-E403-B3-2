"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    if not legal_dir.exists():
        print(f"  ⚠ Directory không tồn tại: {legal_dir}")
        return

    for filepath in legal_dir.iterdir():
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")

            try:
                # Convert file sang Markdown
                result = md.convert(str(filepath))

                # Tạo đường dẫn output
                output_path = output_dir / f"{filepath.stem}.md"

                # Lưu nội dung Markdown
                output_path.write_text(
                    result.text_content,
                    encoding="utf-8"
                )

                print(f"  ✓ Saved: {output_path}")

            except Exception as e:
                print(f"  ✗ Error: {filepath.name}")
                print(f"    {type(e).__name__}: {e}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  ⚠ Directory không tồn tại: {news_dir}")
        return

    for filepath in news_dir.iterdir():
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")

            try:
                # Đọc JSON
                data = json.loads(
                    filepath.read_text(encoding="utf-8")
                )

                # Tạo đường dẫn output
                output_path = output_dir / f"{filepath.stem}.md"

                # Metadata header
                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
                header += "---\n\n"

                # Nội dung bài viết
                content = data.get("content_markdown", "")

                # Nếu content không phải string thì chuyển thành string
                if not isinstance(content, str):
                    content = str(content)

                # Ghép metadata + content
                markdown_content = header + content

                # Lưu file Markdown
                output_path.write_text(
                    markdown_content,
                    encoding="utf-8"
                )

                print(f"  ✓ Saved: {output_path}")

            except json.JSONDecodeError as e:
                print(f"  ✗ Invalid JSON: {filepath.name}")
                print(f"    {e}")

            except Exception as e:
                print(f"  ✗ Error: {filepath.name}")
                print(f"    {type(e).__name__}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
