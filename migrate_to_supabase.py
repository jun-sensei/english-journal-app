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

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT filename, created_at, original_text, corrected_text, corrections "
        "FROM journals ORDER BY created_at ASC"
    ).fetchall()

if not rows:
    print("SQLite に移行対象データがありません。")
    exit(0)

print(f"SQLite に {len(rows)} 件見つかりました。重複チェックしながら移行します...")

inserted = 0
skipped = 0

for row in rows:
    # (filename, created_at) の組み合わせで重複チェック
    existing = sb.table("journals") \
        .select("id") \
        .eq("filename", row["filename"] or "") \
        .eq("created_at", row["created_at"]) \
        .execute()

    if existing.data:
        print(f"  スキップ（重複）: {row['filename']} / {row['created_at']}")
        skipped += 1
        continue

    sb.table("journals").insert({
        "filename": row["filename"],
        "created_at": row["created_at"],
        "original_text": row["original_text"],
        "corrected_text": row["corrected_text"],
        "corrections": row["corrections"],
    }).execute()
    print(f"  挿入: {row['filename']} / {row['created_at']}")
    inserted += 1

print(f"\n完了: {inserted} 件挿入, {skipped} 件スキップ（重複）")
print("Supabase ダッシュボードの journals テーブルで確認してください。")
