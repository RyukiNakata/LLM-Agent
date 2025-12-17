from datetime import datetime
# @tool は削除します
from langchain_openai import ChatOpenAI
from google_calendar_api import get_upcoming_events, add_event
import os

# --- 変更点: @tool を削除し、ダミー引数 query を追加 ---
def get_calendar_events(query: str = "") -> str:
    """
    Google Calendarから直近の予定を取得します。
    引数 query は無視されますが、エージェントの呼び出し互換性のために存在します。
    """
    # 予定を10件取得
    return get_upcoming_events(10)

# --- 変更点: @tool を削除 ---
def create_calendar_event_natural(user_input: str) -> str:
    """
    自然言語から予定を解析し、Googleカレンダーに登録します。
    相対日時（例：「明後日」）は現在日時を基準に解析されます。
    入力例: "明日の14時から会議を追加"
    """
    
    # APIキーの確認（念のため）
    if not os.getenv("OPENAI_API_KEY"):
        return "エラー: OPENAI_API_KEY が設定されていません。"

    # 抽出用のLLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    now = datetime.now()
    
    # プロンプト：現在時刻を伝えて、ISO形式の日時を抽出させる
    prompt = f"""
以下の自然文から予定の「タイトル」「開始日時」「終了日時」を抽出してください。
現在日時は {now.isoformat()} です。相対的な表現（今日、明日、明後日など）はこの日時を基準に計算してください。

出力形式（必ずこの形式で3行のみ出力すること）:
タイトル: [予定のタイトル]
開始: [ISO 8601形式の日時 (例: 2025-05-01T15:00:00+09:00)]
終了: [ISO 8601形式の日時 (例: 2025-05-01T16:00:00+09:00)]

自然文: "{user_input}"
"""

    try:
        response = llm.invoke(prompt)
        lines = response.content.splitlines()

        summary, start_iso, end_iso = "", "", ""
        for line in lines:
            if "タイトル:" in line:
                summary = line.split(":", 1)[1].strip()
            elif "開始:" in line:
                start_iso = line.split(":", 1)[1].strip()
            elif "終了:" in line:
                end_iso = line.split(":", 1)[1].strip()

        if not (summary and start_iso and end_iso):
            return f"エラー: 日時情報の抽出に失敗しました。\nLLMの応答: {response.content}"

        # google_calendar_api の関数を呼び出して登録
        return add_event(summary, start_iso, end_iso)

    except Exception as e:
        return f"エラーが発生しました: {str(e)}"