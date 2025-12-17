import requests
import os

# @tool デコレータは削除します
# from langchain_core.tools import tool  <-- これも削除でOK

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(input_text: str) -> str:
    """天気を取得する関数です。"""

    if "岡山" in input_text or "Okayama" in input_text:
        city = "Okayama,JP"
    elif "," in input_text:
        city = input_text
    else:
        # 都市名が曖昧または省略されている場合
        city = "Okayama,JP"

    # APIキーが設定されていない場合のエラーハンドリング
    if not OPENWEATHER_API_KEY:
        return "エラー: OpenWeatherMap APIキーが設定されていません。"

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