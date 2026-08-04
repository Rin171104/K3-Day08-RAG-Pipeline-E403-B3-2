"""
RAG Chatbot — University Services.
Entrypoint Streamlit: kết nối UI (Role 3) với RAG Pipeline (Task 9 + Task 10).

Chạy:
    streamlit run app.py

Toàn bộ giao diện nằm ở `src/Role_3_FrontendChatbot/chatbot_ui.py`. File này chỉ
đưa project root vào sys.path rồi gọi `run_app()`, để mỗi Role làm việc trong thư
mục riêng mà không giẫm chân nhau khi merge.
"""

import sys
from pathlib import Path

# Thêm project root vào sys.path để import được package `src` khi chạy
# `streamlit run app.py` từ thư mục gốc repo.
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.Role_3_FrontendChatbot.chatbot_ui import run_app

run_app()
