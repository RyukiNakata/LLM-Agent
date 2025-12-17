import requests
import json

# ★ここに M5Stack のデータ取得用URL (humidity_tool.pyと同じもの) を入れてください
DATA_API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

# ★加湿器を操作するAPIのURL (元のコードにあったものを記載しています。必要なら修正してください)
CONTROL_API_URL = "https://vm70a85bd5.execute-api.ap-northeast-1.amazonaws.com/latest/humidifier-control"

def control_humidifier(query: str = "") -> str:
    """
    現在の湿度を取得し、条件に応じて加湿器をON/OFF制御します。
    引数 query は無視されます。
    """
    try:
        # 1. M5Stack APIから現在の湿度を取得
        response = requests.get(DATA_API_URL, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: 湿度データの取得に失敗しました。Status: {response.status_code}"

        data = response.json()
        
        # M5Stackのデータ形式に合わせて取得 (humidity または hum)
        current_humidity = data.get("humidity") or data.get("hum")
        
        if current_humidity is None:
            return f"エラー: データ内に湿度情報が見つかりません。受信データ: {data}"

        # 数値型に変換（念のため）
        current_humidity = float(current_humidity)

        # 2. 制御ロジック (40%未満でON, 60%以上でOFF)
        command = ""
        if current_humidity < 40:
            command = "加湿器をONにする"
            payload = {"command": "on"} # APIに合わせて変更してください
        elif current_humidity > 60:
            command = "加湿器をOFFにする"
            payload = {"command": "off"}
        else:
            return f"現在の湿度は {current_humidity}% です。適切な範囲内（40-60%）のため操作は行いません。"

        # 3. 制御APIを叩く
        # 実際に制御APIがある場合のみ実行
        ctrl_response = requests.post(CONTROL_API_URL, json=payload, timeout=10)
        
        # 結果の返却
        if ctrl_response.status_code == 200:
            result_msg = ctrl_response.json().get("message", "成功")
            return f"現在の湿度は {current_humidity}% です。判定: {command} → 実行結果: {result_msg}"
        else:
            return f"現在の湿度は {current_humidity}% です。判定: {command} を試みましたが、制御APIでエラーが発生しました (Status: {ctrl_response.status_code})。"

    except Exception as e:
        return f"加湿器の自動制御中にエラーが発生しました: {str(e)}"

if __name__ == "__main__":
    print(control_humidifier())