# tools/humidifier_tool.py
import requests
import json

DATA_API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

def control_humidifier(command: str) -> str:
    """
    加湿器を制御します。
    引数 command: 'on', 'off', 'auto' など
    """
    try:
        # 1. (オプション) 現在の湿度を確認してから動作する
        # 不要ならこのブロックは削除してもOK
        try:
            response = requests.get(DATA_API_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                hum = data.get("humidity") or data.get("hum")
                print(f"[Debug] Current Humidity: {hum}%") # デバッグ表示
        except:
            pass # センサー取得エラーは無視して制御に進む

        # 2. 制御実行（シミュレーション）
        # ※ 本当に動かす場合は、ここに M5StackやSwitchBotへのAPIリクエストを書きます
        
        # AIへの返事を作成
        if command == "on":
            return "【実行完了】加湿器をONにしました。湿度が上がるのを待ちます。"
        elif command == "off":
            return "【実行完了】加湿器をOFFにしました。"
        else:
            return f"【実行完了】加湿器を {command} モードに設定しました。"

    except Exception as e:
        return f"加湿器の操作中にエラーが発生しました: {str(e)}"

# テスト用
if __name__ == "__main__":
    print(control_humidifier("off"))