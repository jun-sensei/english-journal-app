"""
SQLite の journals データを Supabase に一括移行するスクリプト。
Supabase 側で journals テーブルを作成した後に一度だけ実行する。

実行方法:
  cd english-journal-app
  python3 migrate_to_supabase.py
"""

import sqlite3
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_PATH = os.path.join(os.path.dirname(__file__), "journal.db")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: .env に SUPABASE_URL と SUPABASE_KEY を設定してください")
    exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# 冪等性チェック: Supabase 側にすでにデータがあれば中断
existing = sb.table("journals").select("id", count="exact").execute()
if existing.count and existing.count > 0:
    print(f"Supabase にすでに {existing.count} 件あります。重複防止のため中断します。")
    print("強制実行したい場合はこのチェックを削除してから実行してください。")
    exit(0)

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT filename, created_at, original_text, corrected_text, corrections "
        "FROM journals ORDER BY created_at ASC"
    ).fetchall()

if not rows:
    print("SQLite に移行対象データがありません。")
    exit(0)

print(f"{len(rows)} 件を Supabase に移行します...")

for row in rows:
    sb.table("journals").insert({
        "filename": row["filename"],
        "created_at": row["created_at"],
        "original_text": row["original_text"],
        "corrected_text": row["corrected_text"],
        "corrections": row["corrections"],
    }).execute()

print(f"完了: {len(rows)} 件を移行しました。")
print("Supabase ダッシュボードの journals テーブルで確認してください。")
