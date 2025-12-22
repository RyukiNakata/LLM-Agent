import requests
import os
from dotenv import load_dotenv # ★追加: .env読み込み用ライブラリ

# ★追加: .envファイルを読み込む
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ★デバッグ用: 実行時にキーが読み込めているか確認（本番では消してもOK）
if OPENWEATHER_API_KEY:
    print("[Debug] Weather API Key loaded.")
else:
    print("[Debug] Weather API Key is MISSING!")

def get_weather(input_text: str) -> str:
    """天気を取得する関数です。"""

    # 都市名の判定
    if "岡山" in input_text or "Okayama" in input_text:
        city = "Okayama,JP"
    elif "," in input_text:
        city = input_text
    else:
        # デフォルト
        city = "Okayama,JP"

    # APIキーがない場合のエラー処理
    if not OPENWEATHER_API_KEY:
        return "エラー: OpenWeatherMap APIキーが .env から読み込めませんでした。"

    # URL作成
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=ja"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            return f"{city} の天気情報を取得できませんでした（理由: {data.get('message', '不明なエラー')})"

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return f"{city}の現在の天気は「{weather}」、気温は{temp}℃（体感{feels_like}℃）、湿度は{humidity}%です。"

    except Exception as e:
        return f"天気情報の取得中にエラーが発生しました: {str(e)}"