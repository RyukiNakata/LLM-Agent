from __future__ import annotations

from langchain_core.tools import tool

# 既存ツール関数（あなたのプロジェクトのまま）
from tools.weather_tool import get_weather
from tools.calendar_tool import create_calendar_event_natural, get_calendar_events
from tools.temp_tool import get_temperature_by_date
from tools.humidity_tool import get_humidity_by_date
from tools.ac_tool import control_aircon
from tools.humidifier_tool import control_humidifier
from tools.co2_tool import get_co2_concentration
from tools.date_tool import get_current_datetime


@tool
def tool_get_weather(location: str) -> str:
    """天気を取得するツール．location（例：Okayama）を受け取り天気文字列を返す．"""
    return get_weather(location)


@tool
def tool_get_temperature_by_date(date: str) -> str:
    """室内の気温を取得するツール．date（例：2025-05-07）を受け取り気温文字列を返す．"""
    return get_temperature_by_date(date)


@tool
def tool_get_humidity_by_date(date: str) -> str:
    """室内の湿度を取得するツール．date（例：2025-05-07）を受け取り湿度文字列を返す．"""
    return get_humidity_by_date(date)


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
    """エアコンを制御するツール．例：『on 24 cool』など（あなたの実装仕様に合わせる）"""
    return control_aircon(command)


@tool
def tool_control_humidifier(command: str) -> str:
    """加湿器を制御するツール．例：『on』／『off』／『auto』など（あなたの実装仕様に合わせる）"""
    return control_humidifier(command)


@tool
def tool_get_co2_concentration(_: str = "") -> str:
    """二酸化炭素濃度を取得するツール．引数不要だがtool都合でダミー引数を許可．"""
    return get_co2_concentration()


@tool
def tool_get_current_datetime(_: str = "") -> str:
    """今日の日時を取得するツール．引数不要だがtool都合でダミー引数を許可．"""
    return get_current_datetime()


# -----------------------------
# 「今」専用ツール（重要）
# LLMに日付を捏造させないため，内部で今日の日付を決めて *_by_date を呼ぶ．
# -----------------------------

def _extract_yyyy_mm_dd(dt_value) -> str:
    s = str(dt_value).strip()
    # 例："2025-05-07 11:52:20.386382" を想定
    return s.split()[0] if s else s


@tool
def tool_get_temperature_now(_: str = "") -> str:
    """現在の室温を取得する．"""
    dt = get_current_datetime()
    date = _extract_yyyy_mm_dd(dt)
    return get_temperature_by_date(date)


@tool
def tool_get_humidity_now(_: str = "") -> str:
    """現在の湿度を取得する．"""
    dt = get_current_datetime()
    date = _extract_yyyy_mm_dd(dt)
    return get_humidity_by_date(date)


# evaluation.py から import される共通ツールセット
tools = [
    tool_get_weather,
    tool_get_temperature_by_date,
    tool_get_humidity_by_date,
    tool_get_calendar_events,
    tool_create_calendar_event_natural,
    tool_control_aircon,
    tool_control_humidifier,
    tool_get_co2_concentration,
    tool_get_current_datetime,
    tool_get_temperature_now,
    tool_get_humidity_now,
]
