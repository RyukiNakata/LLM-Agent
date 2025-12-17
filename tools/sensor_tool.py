import requests
import json
from langchain_core.tools import tool

# ==========================================================
# ★ API URLの設定
# ==========================================================

# 1. 環境データ用 (CO2, 温度, 湿度) - 既存API
CO2_API_URL = "https://h3sit82de1.execute-api.ap-northeast-1.amazonaws.com/latest"

# 2. 生体データ用 (心拍, SpO2) - 新規API
# ※ Step 6で取得したURLに書き換えてください
HEART_API_URL = "https://xxxxxx.execute-api.ap-northeast-1.amazonaws.com/prod/heartrate"

# ==========================================================

def _fetch_and_format(url: str, device_id: str, label: str, mode: str) -> str:
    """
    APIからデータを取得し，モードに合わせて整形して返す関数
    mode: "env" (環境データ) または "vital" (生体データ)
    """
    try:
        # APIリクエスト
        params = {"device_id": device_id}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return f"エラー: {label}の取得に失敗しました (Status: {response.status_code})"

        data = response.json()
        
        # データなし判定
        if "msg" in data and data["msg"] == "No data":
             return f"現在，{label}のデータはありません．"

        # 時刻情報の取得
        time_str = data.get("datetime_str") or str(data.get("timestamp", "不明"))
        
        # --- メッセージ作成 ---
        msg = f"【{time_str} 時点】\n"
        
        # モードごとの表示項目
        if mode == "env":
            # 環境データ (CO2, 温度, 湿度)
            co2 = data.get("co2")
            temp = data.get("temperature")
            hum = data.get("humidity")
            
            if co2 is not None: msg += f"🍃 CO2濃度: {co2} ppm\n"
            if temp is not None: msg += f"🌡️ 気温: {temp} ℃\n"
            if hum is not None: msg += f"💧 湿度: {hum} %"
            
            if co2 is None and temp is None:
                msg += "(有効な環境データが含まれていません)"

        elif mode == "vital":
            # 生体データ (心拍, SpO2)
            bpm = data.get("heart_rate")
            spo2 = data.get("spo2") # もしあれば
            
            if bpm is not None: msg += f"❤️ 心拍数: {bpm} bpm\n"
            if spo2 is not None: msg += f"🩸 SpO2: {spo2} %"
            
            if bpm is None:
                msg += "(有効な心拍データが含まれていません)"

        return msg

    except Exception as e:
        return f"通信エラーが発生しました: {str(e)}"

# =================================================
#  Agent用ツール定義
# =================================================

# --- 1. 環境センサー (CO2, 温度, 湿度) ---

@tool
def get_home_environment(query: str = "") -> str:
    """自宅の環境データ（CO2濃度，温度，湿度）を取得します．部屋が快適か知りたい時に使います．"""
    return _fetch_and_format(CO2_API_URL, "m5_home", "自宅の環境", "env")

@tool
def get_lab_environment(query: str = "") -> str:
    """研究室の環境データ（CO2濃度，温度，湿度）を取得します．研究室の様子を知りたい時に使います．"""
    return _fetch_and_format(CO2_API_URL, "m5_lab", "研究室の環境", "env")

# --- 2. 生体センサー (心拍) ---

@tool
def get_home_heart_rate(query: str = "") -> str:
    """自宅でのユーザーの心拍数を取得します．リラックス状態か確認する時に使います．"""
    return _fetch_and_format(HEART_API_URL, "m5_home", "自宅での心拍数", "vital")

@tool
def get_lab_heart_rate(query: str = "") -> str:
    """研究室にいる人の心拍数を取得します．緊張状態や在室状況を確認する時に使います．"""
    return _fetch_and_format(HEART_API_URL, "m5_lab", "研究室での心拍数", "vital")