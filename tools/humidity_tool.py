import requests
import json

# ★CO2と同じAPI GatewayのURLをここに貼ってください
API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

def get_humidity_by_date(query: str = "") -> str:
    """
    M5Stackから送信された現在の湿度(%)を取得します。
    引数 query (日付など) は現状無視して、最新のデータを返します。
    """
    try:
        # APIを叩いてDynamoDBから最新データを取得
        response = requests.get(API_URL, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: データの取得に失敗しました。Status: {response.status_code}"

        data = response.json()

        # データを取り出す (M5Stackが送っているキー名に合わせてください: humidity や hum など)
        humidity = data.get("humidity") or data.get("hum")
        timestamp = data.get("timestamp", "不明な時刻")
        
        # エラーハンドリング
        if humidity is None:
            # デバッグ用に受信データを表示
            return f"エラー: 湿度データが含まれていません。受信データ: {json.dumps(data, ensure_ascii=False)}"

        return f"{timestamp} 時点の湿度は {humidity} % です。"

    except Exception as e:
        return f"通信エラーが発生しました: {str(e)}"

# テスト用
if __name__ == "__main__":
    print(get_humidity_by_date())