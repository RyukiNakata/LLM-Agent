# tools/ac_tool.py
import requests
import json

# ★ここに M5Stack のデータ取得用URL (センサー確認用)
DATA_API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

def control_aircon(command: str) -> str:
    """
    エアコンを制御します。
    引数 command:
      - 'on', 'cool': 冷房を入れます
      - 'warm', 'heat': 暖房を入れます
      - 'off': 停止します
      - 'auto': 現在の室温に合わせて自動で判定します
    """
    try:
        # -------------------------------------------------
        # 1. センサーデータ取得 (現状確認のため)
        # -------------------------------------------------
        current_temp = "不明"
        try:
            response = requests.get(DATA_API_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # 温度を取得
                val = data.get("temperature") or data.get("temp")
                if val:
                    current_temp = float(val)
        except:
            pass # センサーエラーでもエアコン操作は続行する

        # -------------------------------------------------
        # 2. コマンドに応じた処理
        # -------------------------------------------------
        msg = ""
        
        # --- 自動モード (従来のロジック) ---
        if command == "auto":
            if isinstance(current_temp, float):
                if current_temp >= 28:
                    msg = f"現在 {current_temp}℃ なので、冷房(27℃)をONにしました。"
                elif current_temp <= 16:
                    msg = f"現在 {current_temp}℃ なので、暖房(20℃)をONにしました。"
                else:
                    msg = f"現在 {current_temp}℃ なので、エアコンはOFFのままにします。"
            else:
                msg = "室温が取得できなかったため、自動判断できませんでした。"

        # --- 手動モード (AIが指示) ---
        elif command in ["on", "cool"]:
            msg = f"エアコン(冷房)をONにしました。(現在の室温: {current_temp}℃)"
        
        elif command in ["warm", "heat"]:
            msg = f"エアコン(暖房)をONにしました。(現在の室温: {current_temp}℃)"
            
        elif command == "off":
            msg = "エアコンをOFFにしました。"
            
        else:
            # 想定外のコマンドが来た場合
            msg = f"エアコンを {command} 設定にしました。"

        # -------------------------------------------------
        # 3. 実行 (現在はシミュレーション)
        # -------------------------------------------------
        # ※ 本当に操作するAPIがあるならここで requests.post などを呼ぶ
        
        return f"【実行完了】{msg}"

    except Exception as e:
        return f"エアコン操作中にエラーが発生しました: {str(e)}"

if __name__ == "__main__":
    # テスト
    print(control_aircon("on"))