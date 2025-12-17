import requests
import json

# ★CO2/湿度と同じAPI GatewayのURLをここに貼ってください
API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

def get_temperature_by_date(query: str = "") -> str:
    """
    M5Stackから送信された現在の気温(℃)を取得します。
    引数 query (日付など) は現状無視して、最新のデータを返します。
    """
    try:
        # APIを叩いてDynamoDBから最新データを取得
        response = requests.get(API_URL, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: データの取得に失敗しました。Status: {response.status_code}"

        data = response.json()

        # データを取り出す (M5Stack側のキー名に合わせて temp や temperature を探します)
        temp = data.get("temperature") or data.get("temp")
        timestamp = data.get("timestamp", "不明な時刻")
        
        # エラーハンドリング
        if temp is None:
            return f"エラー: 気温データが含まれていません。受信データ: {json.dumps(data, ensure_ascii=False)}"

        return f"{timestamp} 時点の気温は {temp} ℃ です。"

    except Exception as e:
        return f"通信エラーが発生しました: {str(e)}"

# テスト用
if __name__ == "__main__":
    print(get_temperature_by_date())