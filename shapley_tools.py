from __future__ import annotations

from langchain_core.tools import tool

# ------------------------------------------------------------
# 既存のロジック関数をインポート
# ------------------------------------------------------------
from tools.weather_tool import get_weather
from tools.calendar_tool import create_calendar_event_natural, get_calendar_events
from tools.ac_tool import control_aircon
from tools.humidifier_tool import control_humidifier
from tools.date_tool import get_current_datetime

# ★重要: 新しいセンサーツール（AWS連携版）をインポート
# これらは既に @tool デコレータが付いている想定です
from tools.sensor_tool import (
    get_home_environment,
    get_lab_environment,
    get_home_heart_rate,
    get_lab_heart_rate
)


# ------------------------------------------------------------
# Toolラッパー定義
# ------------------------------------------------------------

@tool
def tool_get_weather(location: str) -> str:
    """天気を取得するツール．location（例：Okayama）を受け取り天気文字列を返す．"""
    return get_weather(location)


@tool
def tool_get_calendar_events(query: str = "") -> str:
    """予定を取得するツール．必要なら検索語を渡す．"""
    return get_calendar_events(query)


@tool
def tool_create_calendar_event_natural(text: str) -> str:
    """自然言語で予定を追加するツール．例：『今日の14時から15時 会議』"""
    return create_calendar_event_natural(text)


@tool
def tool_control_aircon(command: str) -> str:
    """
    エアコンを操作するツール。
    引数 command には以下のいずれかのみを指定する：
    - 'cool': 冷房を入れる場合
    - 'warm': 暖房を入れる場合
    - 'off': 停止する場合
    - 'auto': 温度に合わせて自動で判断してほしい場合
    """
    return control_aircon(command)


@tool
def tool_control_humidifier(command: str) -> str:
    """加湿器を制御するツール．例：『on』／『off』など"""
    return control_humidifier(command)


@tool
def tool_get_current_datetime(_: str = "") -> str:
    """今日の日時を取得するツール．引数不要だがtool都合でダミー引数を許可．"""
    return get_current_datetime()


# ------------------------------------------------------------
# ツールリストの作成
# evaluation.py から import される共通ツールセット
# ------------------------------------------------------------
tools = [
    # 基本ツール
    tool_get_weather,
    tool_get_calendar_events,
    tool_create_calendar_event_natural,
    tool_control_aircon,
    tool_control_humidifier,
    tool_get_current_datetime,
    
    # ★新しいIoTセンサーツール (AWS DynamoDB連携)
    get_home_environment,  # 自宅の気温・湿度・CO2
    get_lab_environment,   # 研究室の環境
    get_home_heart_rate,   # 自宅の心拍数
    get_lab_heart_rate,    # 研究室の心拍数
]