import html
import os
import re
import json
import socket
import sqlite3
from collections import defaultdict
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "journal.db")
MODEL = "gemini-2.5-flash"
MAX_PDF_MB = 10

CATEGORY_JA = {
    # 新7分類
    "Tense/Aspect": "時制・アスペクト",
    "Subject-Verb Agreement": "主語・動詞の一致",
    "Word Form": "語形",
    "Article/Preposition": "冠詞・前置詞",
    "Vocabulary/Collocation": "語彙・コロケーション",
    "Sentence Structure": "文の構造",
    "Other": "その他",
    # 旧分類（後方互換）
    "Grammar": "文法",
    "Vocabulary": "語彙",
    "Tense": "時制",
}

SEVERITY_WEIGHTS = {
    "Tense/Aspect": 5,
    "Subject-Verb Agreement": 5,
    "Article/Preposition": 5,
    "Word Form": 4,
    "Sentence Structure": 4,
    "Vocabulary/Collocation": 3,
    "Other": 2,
    # 旧分類（後方互換）
    "Grammar": 5,
    "Vocabulary": 3,
    "Tense": 4,
}

RATING_COLOR = {"S": "#dc2626", "A": "#f97316", "B": "#eab308", "C": "#22c55e"}
RATING_LABEL = {"S": "S（最重要）", "A": "A（重要）", "B": "B（要改善）", "C": "C（軽微）"}

CATEGORY_BG = {
    "Subject-Verb Agreement": "#fecaca",
    "Grammar": "#fecaca",
    "Tense/Aspect": "#fed7aa",
    "Tense": "#fed7aa",
    "Word Form": "#f5d0fe",
    "Article/Preposition": "#fef08a",
    "Vocabulary/Collocation": "#bfdbfe",
    "Vocabulary": "#bfdbfe",
    "Sentence Structure": "#d9f99d",
    "Other": "#e5e7eb",
}
CATEGORY_TC = {
    "Subject-Verb Agreement": "#991b1b",
    "Grammar": "#991b1b",
    "Tense/Aspect": "#9a3412",
    "Tense": "#9a3412",
    "Word Form": "#7e22ce",
    "Article/Preposition": "#92400e",
    "Vocabulary/Collocation": "#1e40af",
    "Vocabulary": "#1e40af",
    "Sentence Structure": "#3f6212",
    "Other": "#374151",
}

PROMPT = """You are a system with three expert roles for processing handwritten English journal entries.

IMPORTANT: Because the source text is handwritten, do NOT flag or correct punctuation errors (periods, commas, etc.) or capitalization errors (uppercase/lowercase mistakes). Only correct genuine grammar, vocabulary, and expression errors.

Role 1 - Transcriber: Read the handwritten text from the PDF carefully. Use context clues for unclear handwriting and transcribe as accurately as possible.

Role 2 - Proofreader: Correct grammar, vocabulary, and expression errors. Rewrite into natural, journal-appropriate English that a native speaker would write. Do NOT change punctuation or capitalization — leave them exactly as written.

Role 3 - Tutor: For each correction made, provide:
1. The category (choose exactly one):
   - "Tense/Aspect" (wrong tense, incorrect perfect or progressive form)
   - "Subject-Verb Agreement" (third-person singular -s missing, be-verb mismatch, etc.)
   - "Word Form" (wrong part of speech, incorrect verb conjugation such as irregular past tense)
   - "Article/Preposition" (a/an/the errors, wrong or missing prepositions)
   - "Vocabulary/Collocation" (wrong word choice, unnatural word combinations like "make homework")
   - "Sentence Structure" (word order, incomplete sentences, run-on sentences)
   - "Other" (spelling errors only — never punctuation or capitalization)
2. A reason in Japanese explaining the grammatical rule violated, understandable to a Japanese middle or high school student.
3. A one-sentence learning tip in Japanese to help avoid this mistake next time.

Also identify up to 3 phrases the student used correctly and effectively (good grammar, natural expression, or strong vocabulary choice) and note why they are good.

Return ONLY a valid JSON object with no markdown fences or extra text:
{
  "original_text": "<transcribed text exactly as handwritten, preserving all errors>",
  "corrected_text": "<fully corrected, natural English text>",
  "corrections": [
    {
      "original": "<original phrase containing the error>",
      "corrected": "<corrected phrase>",
      "reason": "<explanation of the grammar rule violated, in Japanese>",
      "learning_point": "<one-sentence tip to avoid this mistake, in Japanese>",
      "category": "<Tense/Aspect|Subject-Verb Agreement|Word Form|Article/Preposition|Vocabulary/Collocation|Sentence Structure|Other>"
    }
  ],
  "strengths": [
    {
      "phrase": "<phrase the student used well>",
      "comment": "<why this is good, in Japanese>"
    }
  ]
}"""

TEXT_PROMPT = """You are a proofreader and English tutor for journal entries written by Japanese students.

Role 1 - Proofreader: Correct grammar, vocabulary, and expression errors. Rewrite into natural, journal-appropriate English that a native speaker would write.

Role 2 - Tutor: For each correction made, provide:
1. The category (choose exactly one):
   - "Tense/Aspect" (wrong tense, incorrect perfect or progressive form)
   - "Subject-Verb Agreement" (third-person singular -s missing, be-verb mismatch, etc.)
   - "Word Form" (wrong part of speech, incorrect verb conjugation such as irregular past tense)
   - "Article/Preposition" (a/an/the errors, wrong or missing prepositions)
   - "Vocabulary/Collocation" (wrong word choice, unnatural word combinations like "make homework")
   - "Sentence Structure" (word order, incomplete sentences, run-on sentences)
   - "Other" (spelling errors, punctuation errors)
2. A reason in Japanese explaining the grammatical rule violated, understandable to a Japanese middle or high school student.
3. A one-sentence learning tip in Japanese to help avoid this mistake next time.

Also identify up to 3 phrases the student used correctly and effectively (good grammar, natural expression, or strong vocabulary choice) and note why they are good.

Return ONLY a valid JSON object with no markdown fences or extra text:
{
  "corrected_text": "<fully corrected, natural English text>",
  "corrections": [
    {
      "original": "<original phrase containing the error>",
      "corrected": "<corrected phrase>",
      "reason": "<explanation of the grammar rule violated, in Japanese>",
      "learning_point": "<one-sentence tip to avoid this mistake, in Japanese>",
      "category": "<Tense/Aspect|Subject-Verb Agreement|Word Form|Article/Preposition|Vocabulary/Collocation|Sentence Structure|Other>"
    }
  ],
  "strengths": [
    {
      "phrase": "<phrase the student used well>",
      "comment": "<why this is good, in Japanese>"
    }
  ]
}"""


@st.cache_resource
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("環境変数 GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
        st.stop()
    return genai.Client(api_key=api_key, http_options={"api_version": "v1"})


@st.cache_resource
def _supabase():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key:
        from supabase import create_client
        return create_client(url, key)
    return None


@st.cache_resource
def init_db():
    sb = _supabase()
    if sb:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filename TEXT,
                original_text TEXT,
                corrected_text TEXT,
                corrections TEXT
            )
        """)
        conn.commit()


def save_to_db(filename: str, original_text: str, corrected_text: str, corrections: list):
    sb = _supabase()
    if sb:
        try:
            sb.table("journals").insert({
                "filename": filename,
                "original_text": original_text,
                "corrected_text": corrected_text,
                "corrections": json.dumps(corrections, ensure_ascii=False),
            }).execute()
            return
        except Exception as e:
            st.error(f"Supabase保存エラー: {type(e).__name__}: {e}")
            raise
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO journals (filename, original_text, corrected_text, corrections) VALUES (?,?,?,?)",
            (filename, original_text, corrected_text, json.dumps(corrections, ensure_ascii=False))
        )
        conn.commit()


def load_all_entries():
    sb = _supabase()
    if sb:
        res = sb.table("journals").select(
            "id,created_at,filename,original_text,corrected_text,corrections"
        ).order("created_at", desc=True).execute()
        return res.data
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, filename, original_text, corrected_text, corrections "
            "FROM journals ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def load_corrections_with_meta() -> list[dict]:
    sb = _supabase()
    if sb:
        res = sb.table("journals").select("id,created_at,corrections").order("created_at").execute()
        rows = res.data
    else:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            raw_rows = conn.execute(
                "SELECT id, created_at, corrections FROM journals ORDER BY created_at ASC"
            ).fetchall()
        rows = [dict(row) for row in raw_rows]
    result = []
    for row in rows:
        try:
            for c in json.loads(row["corrections"]):
                result.append({
                    "journal_id": row["id"],
                    "created_at": row["created_at"],
                    "category": c.get("category", "Other"),
                    "original": c.get("original", ""),
                    "corrected": c.get("corrected", ""),
                    "reason": c.get("reason", ""),
                })
        except (json.JSONDecodeError, TypeError):
            continue
    return result


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return ""


def get_public_url() -> str:
    """ngrok が起動中であれば公開URLを返す（ngrok local API経由）"""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=1) as r:
            data = json.loads(r.read())
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return tunnel["public_url"]
    except Exception:
        pass
    return ""


def highlight_corrections(text: str, corrections: list, use_original: bool) -> str:
    """修正箇所をカテゴリ色でハイライトした HTML div を返す"""
    field = "original" if use_original else "corrected"
    result = html.escape(text)
    for c in corrections:
        phrase = c.get(field, "").strip()
        if not phrase:
            continue
        escaped_phrase = html.escape(phrase)
        if escaped_phrase not in result:
            continue
        cat = c.get("category", "Other")
        bg = CATEGORY_BG.get(cat, "#e5e7eb")
        tc = CATEGORY_TC.get(cat, "#374151")
        tag = (
            f'<mark style="background:{bg};color:{tc};padding:1px 4px;'
            f'border-radius:3px;font-weight:600;">{escaped_phrase}</mark>'
        )
        result = result.replace(escaped_phrase, tag, 1)
    return (
        '<div style="font-family:monospace;white-space:pre-wrap;font-size:0.88em;'
        'line-height:1.75;padding:12px;background:#f9fafb;border:1px solid #e5e7eb;'
        f'border-radius:6px;max-height:260px;overflow-y:auto;">{result}</div>'
    )


def extract_json(raw: str) -> str:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        return match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def analyze_journal(pdf_bytes: bytes) -> dict:
    client = get_client()
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    response = client.models.generate_content(
        model=MODEL,
        contents=[pdf_part, PROMPT]
    )
    if not response.text:
        raise ValueError("AIからの応答が空でした。PDFの内容を確認してください。")
    return json.loads(extract_json(response.text))


def analyze_text(text: str) -> dict:
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[f"Please proofread and correct the following English journal entry:\n\n{text}\n\n{TEXT_PROMPT}"]
    )
    if not response.text:
        raise ValueError("AIからの応答が空でした。")
    result = json.loads(extract_json(response.text))
    result["original_text"] = text
    return result


def _show_analysis_result(result: dict, original_label: str, key_prefix: str):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(original_label)
        st.text_area(
            original_label,
            value=result.get("original_text", ""),
            height=350,
            disabled=True,
            key=f"{key_prefix}result_original",
            label_visibility="collapsed"
        )
    with col2:
        st.subheader("添削後（ネイティブ表現）")
        st.text_area(
            "添削後のテキスト",
            value=result.get("corrected_text", ""),
            height=350,
            disabled=True,
            key=f"{key_prefix}result_corrected",
            label_visibility="collapsed"
        )

    corrections = result.get("corrections", [])
    st.subheader(f"添削ポイント（{len(corrections)}件）")
    if corrections:
        df = pd.DataFrame(corrections)
        if "category" in df.columns:
            df["category"] = df["category"].map(CATEGORY_JA).fillna(df["category"])
        rename_map = {
            "original": "元の表現",
            "corrected": "修正後",
            "reason": "理由",
            "learning_point": "学習ポイント",
            "category": "カテゴリ",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        cols_order = [c for c in ["カテゴリ", "元の表現", "修正後", "理由", "学習ポイント"] if c in df.columns]
        st.dataframe(df[cols_order], use_container_width=True, hide_index=True)
    else:
        st.success("修正点はありませんでした！素晴らしい英文です。")

    strengths = result.get("strengths", [])
    if strengths:
        st.subheader("良かった表現")
        for s in strengths:
            phrase = html.escape(s.get("phrase", ""))
            comment = html.escape(s.get("comment", ""))
            st.markdown(
                f'<div style="margin:4px 0;padding:8px 12px;background:#dcfce7;'
                f'border-left:3px solid #22c55e;border-radius:4px;">'
                f'<span style="color:#166534;font-weight:600;">✓ {phrase}</span>'
                f'<br><span style="font-size:0.85em;color:#16a34a;">{comment}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _pdf_section():
    if "pdf_analysis_result" not in st.session_state:
        st.session_state.pdf_analysis_result = None
    if "pdf_analyzing" not in st.session_state:
        st.session_state.pdf_analyzing = False

    uploaded = st.file_uploader(
        "手書き英語ジャーナルのPDFを選択してください",
        type=["pdf"],
        help="スキャンした手書きジャーナルのPDFをアップロードしてください"
    )

    if not uploaded:
        st.session_state.pdf_analysis_result = None
        return

    size_mb = uploaded.size / (1024 * 1024)
    st.write(f"ファイル: `{uploaded.name}` ({size_mb:.1f} MB)")

    if size_mb > MAX_PDF_MB:
        st.warning(f"ファイルサイズが大きすぎます（上限 {MAX_PDF_MB}MB）。圧縮してから再度お試しください。")
        return

    if st.button("添削開始", type="primary", use_container_width=True, disabled=st.session_state.pdf_analyzing, key="pdf_submit"):
        st.session_state.pdf_analyzing = True
        st.session_state.pdf_analysis_result = None
        with st.spinner("Gemini AIが文字起こし・添削中... 少々お待ちください"):
            try:
                result = analyze_journal(uploaded.read())
            except json.JSONDecodeError:
                st.error("レスポンスのパースに失敗しました。もう一度お試しください。")
                st.session_state.pdf_analyzing = False
                return
            except ValueError as e:
                st.error(str(e))
                st.session_state.pdf_analyzing = False
                return
            except Exception as e:
                st.error(f"APIエラー: {e}")
                st.session_state.pdf_analyzing = False
                return

        original_text = result.get("original_text", "")
        corrected_text = result.get("corrected_text", "")
        corrections = result.get("corrections", [])

        if not original_text:
            st.error("文字起こし結果が取得できませんでした。もう一度お試しください。")
            st.session_state.pdf_analyzing = False
            return

        save_to_db(uploaded.name, original_text, corrected_text, corrections)
        st.session_state.pdf_analysis_result = result
        st.session_state.pdf_analyzing = False
        st.toast("添削完了！データベースに保存しました。", icon="✅")

    result = st.session_state.pdf_analysis_result
    if result:
        _show_analysis_result(result, original_label="文字起こし（元の文章）", key_prefix="pdf_")


def _text_section():
    if "text_analysis_result" not in st.session_state:
        st.session_state.text_analysis_result = None
    if "text_analyzing" not in st.session_state:
        st.session_state.text_analyzing = False

    input_text = st.text_area(
        "英語ジャーナルを入力してください",
        height=250,
        placeholder="Today I went to school and...",
        help="添削したい英文を直接入力してください",
        key="text_input_area",
    )

    if st.button("添削開始", type="primary", use_container_width=True, disabled=st.session_state.text_analyzing, key="text_submit"):
        if not input_text.strip():
            st.warning("テキストを入力してください。")
            return
        st.session_state.text_analyzing = True
        st.session_state.text_analysis_result = None
        with st.spinner("Gemini AIが添削中... 少々お待ちください"):
            try:
                result = analyze_text(input_text.strip())
            except json.JSONDecodeError:
                st.error("レスポンスのパースに失敗しました。もう一度お試しください。")
                st.session_state.text_analyzing = False
                return
            except ValueError as e:
                st.error(str(e))
                st.session_state.text_analyzing = False
                return
            except Exception as e:
                st.error(f"APIエラー: {e}")
                st.session_state.text_analyzing = False
                return

        original_text = result.get("original_text", "")
        corrected_text = result.get("corrected_text", "")
        corrections = result.get("corrections", [])
        save_to_db("（直接入力）", original_text, corrected_text, corrections)
        st.session_state.text_analysis_result = result
        st.session_state.text_analyzing = False
        st.toast("添削完了！データベースに保存しました。", icon="✅")

    result = st.session_state.text_analysis_result
    if result:
        _show_analysis_result(result, original_label="入力した文章", key_prefix="text_")


def upload_tab():
    st.header("アップロード・添削")

    input_method = st.radio(
        "入力方法",
        ["PDFをアップロード", "テキストを直接入力"],
        horizontal=True,
    )

    if input_method == "PDFをアップロード":
        _pdf_section()
    else:
        _text_section()


def dashboard_tab():
    st.header("エラー分析ダッシュボード")

    entries = load_all_entries()

    if not entries:
        st.info("まだデータがありません。PDFをアップロードして添削を実行してください。")
        return

    all_corrections = []
    for row in entries:
        try:
            all_corrections.extend(json.loads(row["corrections"]))
        except (json.JSONDecodeError, TypeError):
            continue

    if not all_corrections:
        st.info("添削データがまだありません。")
        return

    df = pd.DataFrame(all_corrections)
    total_entries = len(entries)
    total_corrections = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("添削済みジャーナル数", total_entries)
    col2.metric("総添削件数", total_corrections)
    col3.metric("1件あたりの平均エラー数", f"{total_corrections / total_entries:.1f}")

    st.divider()

    if "category" in df.columns:
        df["category_ja"] = df["category"].map(CATEGORY_JA).fillna(df["category"])
    else:
        df["category_ja"] = "その他"

    category_counts = df["category_ja"].value_counts().reset_index()
    category_counts.columns = ["カテゴリ", "件数"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            category_counts, values="件数", names="カテゴリ",
            title="エラーカテゴリの割合",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            category_counts, x="カテゴリ", y="件数",
            title="カテゴリ別エラー数",
            color="カテゴリ",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("全添削履歴")
    cols_to_show = [c for c in ["category", "original", "corrected", "reason"] if c in df.columns]
    history_df = df[cols_to_show].copy()
    if "category" in history_df.columns:
        history_df["category"] = history_df["category"].map(CATEGORY_JA).fillna(history_df["category"])
    history_df = history_df.rename(columns={
        "category": "カテゴリ",
        "original": "元の表現",
        "corrected": "修正後",
        "reason": "理由"
    })
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    legend_cats = [
        "Subject-Verb Agreement", "Tense/Aspect", "Word Form",
        "Article/Preposition", "Vocabulary/Collocation", "Sentence Structure", "Other"
    ]
    legend_parts = " ".join(
        f'<span style="background:{CATEGORY_BG[cat]};color:{CATEGORY_TC[cat]};'
        f'padding:2px 8px;border-radius:10px;font-size:0.8em;font-weight:600;margin-right:4px;">'
        f'{CATEGORY_JA[cat]}</span>'
        for cat in legend_cats
    )
    st.subheader(f"ジャーナル一覧（全{total_entries}件）")
    st.markdown(f"ハイライト凡例: {legend_parts}", unsafe_allow_html=True)

    for i, row in enumerate(entries[:10]):
        try:
            corrs = json.loads(row["corrections"])
        except (json.JSONDecodeError, TypeError):
            corrs = []
        label = f"{str(row['created_at'])[:16]}  |  {row['filename']}  |  修正 {len(corrs)} 件"
        with st.expander(label):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**元の文章**（修正箇所を強調）")
                st.markdown(
                    highlight_corrections(row["original_text"], corrs, use_original=True),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("**添削後**（修正済み箇所を強調）")
                st.markdown(
                    highlight_corrections(row["corrected_text"], corrs, use_original=False),
                    unsafe_allow_html=True,
                )
            if corrs:
                st.markdown("---")
                st.markdown("**修正一覧**")
                for c in corrs:
                    cat = c.get("category", "Other")
                    bg = CATEGORY_BG.get(cat, "#e5e7eb")
                    tc = CATEGORY_TC.get(cat, "#374151")
                    cat_ja = CATEGORY_JA.get(cat, cat)
                    orig = html.escape(c.get("original", ""))
                    corr_text = html.escape(c.get("corrected", ""))
                    reason = html.escape(c.get("reason", ""))
                    learning_point = html.escape(c.get("learning_point", ""))
                    lp_html = (
                        f'<br><span style="font-size:0.8em;color:#7c3aed;">💡 {learning_point}</span>'
                        if learning_point else ""
                    )
                    st.markdown(
                        f'<div style="margin:4px 0;padding:6px 10px;background:{bg}33;'
                        f'border-left:3px solid {bg};border-radius:4px;">'
                        f'<span style="background:{bg};color:{tc};padding:1px 6px;border-radius:10px;'
                        f'font-size:0.78em;font-weight:600;">{cat_ja}</span>&nbsp;'
                        f'<span style="color:#dc2626;text-decoration:line-through;">{orig}</span>'
                        f' → <span style="color:#16a34a;font-weight:600;">{corr_text}</span>'
                        f'<br><span style="font-size:0.82em;color:#6b7280;">{reason}</span>'
                        f'{lp_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    if total_entries >= 10:
        st.info(f"直近10件を表示中。全{total_entries}件のデータがあります。")


def _importance_rating(score: float) -> str:
    if score >= 3.5:
        return "S"
    if score >= 2.5:
        return "A"
    if score >= 1.5:
        return "B"
    return "C"


def recurrence_tab():
    st.header("再発エラー分析")
    st.caption("複数のジャーナルをまたいで繰り返されたミスを可視化し、是正の優先度を格付けします")

    entries = load_all_entries()
    total_journals = len(entries)

    if total_journals < 2:
        st.info("再発分析には2件以上のジャーナルが必要です。PDFをもう1件アップロードしてください。")
        return

    all_corrections = load_corrections_with_meta()
    if not all_corrections:
        st.info("添削データがありません。")
        return

    cat_journals: dict[str, set] = defaultdict(set)
    cat_counts: dict[str, int] = defaultdict(int)
    for c in all_corrections:
        cat = c["category"]
        cat_journals[cat].add(c["journal_id"])
        cat_counts[cat] += 1

    cat_rows = []
    for cat, journals in cat_journals.items():
        rate = len(journals) / total_journals
        severity = SEVERITY_WEIGHTS.get(cat, 2)
        importance = rate * severity
        cat_rows.append({
            "category": cat,
            "category_ja": CATEGORY_JA.get(cat, cat),
            "total_count": cat_counts[cat],
            "journal_count": len(journals),
            "recurrence_rate": rate,
            "severity": severity,
            "importance": importance,
            "rating": _importance_rating(importance),
        })

    cat_df = pd.DataFrame(cat_rows).sort_values("importance", ascending=False)

    st.subheader("重要度マトリックス（再発率 × 重大性）")
    fig = px.scatter(
        cat_df,
        x="recurrence_rate",
        y="severity",
        size="total_count",
        color="rating",
        text="category_ja",
        title="エラーカテゴリ別 是正重要度マトリックス",
        labels={
            "recurrence_rate": "再発率（ジャーナル全体に占める割合）",
            "severity": "重大性スコア（英語習得への影響度）",
            "total_count": "総修正回数",
            "rating": "格付け",
        },
        color_discrete_map=RATING_COLOR,
        size_max=70,
    )
    fig.update_traces(textposition="top center", marker=dict(opacity=0.85))
    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.15], tickformat=".0%"),
        yaxis=dict(range=[0.5, 6]),
        legend_title_text="格付け",
    )
    fig.add_hline(y=3.5, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("是正優先度ランキング")
    display_df = cat_df[[
        "rating", "category_ja", "total_count", "journal_count", "recurrence_rate", "severity", "importance"
    ]].copy()
    display_df["rating"] = display_df["rating"].map(RATING_LABEL)
    display_df["recurrence_rate"] = display_df["recurrence_rate"].apply(lambda x: f"{x:.0%}")
    display_df["importance"] = display_df["importance"].apply(lambda x: f"{x:.2f}")
    display_df.columns = ["格付け", "カテゴリ", "総修正回数", "出現ジャーナル数", "再発率", "重大性(1-5)", "是正重要度"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("格付け基準について"):
        st.markdown(
            "**是正重要度 = 再発率 × 重大性スコア** で算出します。\n\n"
            "| 格付け | 是正重要度 | 目安 |\n"
            "|--------|-----------|------|\n"
            "| S（最重要）| 3.5以上 | 高頻度かつ意味伝達に直結するミス |\n"
            "| A（重要）  | 2.5〜3.5 | 複数回出現・英語の自然さに影響 |\n"
            "| B（要改善）| 1.5〜2.5 | 散発的だが改善余地あり |\n"
            "| C（軽微）  | 1.5未満  | 出現が少なく影響も限定的 |\n\n"
            "**重大性スコア**は英語試験評価基準を参考に設定:\n"
            "主語・動詞の一致(5) = 時制・アスペクト(5) = 冠詞・前置詞(5) ＞ 語形(4) = 文の構造(4) ＞ 語彙・コロケーション(3) ＞ その他(2)"
        )

    st.divider()

    st.subheader("同一表現の再発検出")
    st.caption("2件以上のジャーナルで同じ（または酷似した）表現が修正されたケースを抽出します")

    phrase_journals: dict[tuple, set] = defaultdict(set)
    phrase_data_map: dict[tuple, dict] = {}
    for c in all_corrections:
        key = (c["category"], c["original"].strip().lower())
        phrase_journals[key].add(c["journal_id"])
        if key not in phrase_data_map:
            phrase_data_map[key] = c

    recurring_rows = []
    for key, journal_set in phrase_journals.items():
        if len(journal_set) < 2:
            continue
        c = phrase_data_map[key]
        rate = len(journal_set) / total_journals
        importance = rate * SEVERITY_WEIGHTS.get(c["category"], 2)
        recurring_rows.append({
            "格付け": RATING_LABEL.get(_importance_rating(importance), ""),
            "カテゴリ": CATEGORY_JA.get(c["category"], c["category"]),
            "元の表現": c["original"],
            "修正後": c["corrected"],
            "理由": c["reason"],
            "再発回数": len(journal_set),
        })

    if recurring_rows:
        recurring_df = pd.DataFrame(recurring_rows).sort_values("再発回数", ascending=False)
        st.dataframe(recurring_df, use_container_width=True, hide_index=True)
    else:
        st.success("同一表現の再発は検出されませんでした。データが増えると自動で追跡されます。")


def main():
    st.set_page_config(
        page_title="英語ジャーナル添削AI",
        page_icon="✏️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Streamlit Community Cloud のシークレットを os.environ にブリッジ（_supabase()より前に実行）
    try:
        for _k, _v in st.secrets.items():
            if _k not in os.environ:
                os.environ[_k] = str(_v)
    except Exception:
        pass

    st.title("英語ジャーナル添削AI")
    st.caption("英語ジャーナルをPDFアップロードまたは直接入力すると、文法添削・エラー分析を自動で行います")

    is_cloud = bool(os.getenv("SUPABASE_URL"))
    if is_cloud:
        public_url = ""
        local_ip = ""
    else:
        public_url = get_public_url()
        local_ip = get_local_ip()

    with st.sidebar:
        if is_cloud:
            st.subheader("デプロイ情報")
            st.success("Streamlit Community Cloudで動作中")
            st.caption("このページのURLをブックマークまたは共有してください")
        else:
            st.subheader("アクセスURL")
            if public_url:
                st.success(f"**どこからでもアクセス可能**\n\n{public_url}")
                st.caption("インターネット経由（ngrok）")
            if local_ip:
                st.info(f"**同じWi-Fi内からのアクセス**\n\nhttp://{local_ip}:8501")
                st.caption("LAN内のみ")
            if not public_url and not local_ip:
                st.warning("URLを取得できませんでした")
            if not public_url:
                st.caption("外部公開するには start.sh でアプリを起動してください")

    init_db()

    tab1, tab2, tab3 = st.tabs(["📄 アップロード・添削", "📊 ダッシュボード", "🔁 再発エラー分析"])
    with tab1:
        upload_tab()
    with tab2:
        dashboard_tab()
    with tab3:
        recurrence_tab()


if __name__ == "__main__":
    main()
