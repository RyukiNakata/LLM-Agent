import requests
import json

# ★ここに M5Stack のデータ取得用URL (temp_tool.pyと同じもの) を入れてください
DATA_API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

# ★エアコンを操作するAPIのURL (元のコードにあったものを記載しています)
CONTROL_API_URL = "https://3akspoud88.execute-api.ap-northeast-1.amazonaws.com/prod/aircon-control"

def control_aircon(query: str = "") -> str:
    """
    現在の気温を取得し、自動制御ルールに基づいてエアコンを操作します。
    引数 query は無視されます。
    """
    try:
        # 1. M5Stack APIから現在の気温を取得
        response = requests.get(DATA_API_URL, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: 気温データの取得に失敗しました。Status: {response.status_code}"

        data = response.json()
        
        # M5Stackのデータ形式に合わせて取得 (temperature または temp)
        current_temp = data.get("temperature") or data.get("temp")
        
        if current_temp is None:
            return f"エラー: データ内に気温情報が見つかりません。受信データ: {data}"

        # 数値型に変換
        current_temp = float(current_temp)

        # 2. 判断ロジック (28度以上なら冷房、16度以下なら暖房、それ以外はOFF)
        command = ""
        if current_temp >= 28:
            command = "冷房を27度に設定して"
        elif current_temp <= 16:
            command = "暖房を20度に設定して"
        else:
            # 快適な範囲ならOFFにする（または「そのまま」にする場合はここを変更）
            command = "エアコンを消して"

        # 3. エアコン制御API呼び出し
        # 実際に制御APIがある場合のみ実行
        ctrl_response = requests.post(CONTROL_API_URL, json={"command": command}, timeout=10)
        
        # 結果の返却
        if ctrl_response.status_code == 200:
            result_msg = ctrl_response.json().get("message", "成功")
            return f"現在の気温は {current_temp}℃ です。判定: {command} → 実行結果: {result_msg}"
        else:
            return f"現在の気温は {current_temp}℃ です。判定: {command} を試みましたが、制御APIでエラーが発生しました (Status: {ctrl_response.status_code})。"

    except Exception as e:
        return f"エアコンの自動制御中にエラーが発生しました: {str(e)}"

# テスト用
if __name__ == "__main__":
    print(control_aircon())