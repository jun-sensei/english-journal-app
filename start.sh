#!/bin/bash
# アプリ起動スクリプト（Streamlit + ngrok 同時起動）
# 初回: ngrok の設定が必要です（README の手順を参照）

set -e
cd "$(dirname "$0")"

# .env から NGROK_DOMAIN を読み込む
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep NGROK_DOMAIN | xargs) 2>/dev/null || true
fi

if [ -z "$NGROK_DOMAIN" ]; then
    echo "❌ .env に NGROK_DOMAIN が設定されていません。"
    echo "   例: NGROK_DOMAIN=your-name.ngrok-free.app"
    exit 1
fi

# 既存プロセスを停止
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "ngrok http" 2>/dev/null || true
sleep 1

echo "🚀 Streamlit を起動中..."
python3 -m streamlit run app.py &
STREAMLIT_PID=$!
sleep 3

echo "🌐 ngrok トンネルを開始中（ドメイン: $NGROK_DOMAIN）..."
ngrok http --domain="$NGROK_DOMAIN" 8501 &
NGROK_PID=$!
sleep 2

echo ""
echo "✅ アプリが起動しました"
echo "   公開URL : https://$NGROK_DOMAIN"
echo "   ローカル: http://localhost:8501"
echo ""
echo "終了するには Ctrl+C を押してください"

cleanup() {
    echo ""
    echo "⏹  停止中..."
    kill "$STREAMLIT_PID" 2>/dev/null || true
    kill "$NGROK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$STREAMLIT_PID"
