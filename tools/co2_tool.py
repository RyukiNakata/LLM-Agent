import requests
import json

API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

def get_co2_concentration(query: str = "") -> str:
    """
    M5Stackから送信された現在の二酸化炭素(CO2)濃度を取得します。
    """
    try:
        # APIを叩いてDynamoDBから最新データを取得
        response = requests.get(API_URL, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: データの取得に失敗しました。Status: {response.status_code}"

        data = response.json()

        # データを取り出す
        co2 = data.get("co2")
        timestamp = data.get("timestamp", "不明な時刻")
        
        # エラーハンドリング
        if co2 is None:
            return "エラー: CO2データが含まれていません。"

        return f"{timestamp} 時点の二酸化炭素濃度は {co2} ppm です。"

    except Exception as e:
        return f"通信エラーが発生しました: {str(e)}"

# テスト用
if __name__ == "__main__":
    print(get_co2_concentration())