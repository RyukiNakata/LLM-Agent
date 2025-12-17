from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
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


# ------------------------------------------------------------
# 1 Toolラッパ（LangChain v1では @tool で登録するのが安定）
#    既存関数をそのまま呼び出す薄いラッパにする
# ------------------------------------------------------------

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


TOOLS = [
    tool_get_weather,
    tool_get_temperature_by_date,
    tool_get_humidity_by_date,
    tool_get_calendar_events,
    tool_create_calendar_event_natural,
    tool_control_aircon,
    tool_control_humidifier,
    tool_get_co2_concentration,
    tool_get_current_datetime,
]


# ------------------------------------------------------------
# 2 プロンプト（few-shotを system_prompt に埋め込む）
#    v1では「SystemMessageを渡す」より system_prompt 文字列が扱いやすい
# ------------------------------------------------------------

FEW_SHOT = """
以下はユーザーの質問と，ツール利用の例です（内部的にはツールを呼び出して回答する）．

例1：
ユーザー：明日の天気を教えて
行動：tool_get_weather(location="Okayama")
最終回答：明日の岡山は晴れの予報です．

例2：
ユーザー：今日の14時から15時会議を追加して
行動：tool_get_current_datetime()
行動：tool_create_calendar_event_natural(text="今日の14時から15時 会議を追加")
最終回答：14時から15時の会議を追加しました．
""".strip()


def _build_agent(mode: str = "zero"):

    # .env を使う場合（OPENAI_API_KEY）
    load_dotenv()

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1"),
        temperature=0,
    )

    if mode == "few":
        system_prompt = (
            "あなたは家庭内アシスタントです．可能ならツールを使って正確に答えてください．\n\n"
            f"{FEW_SHOT}\n\n"
            "ユーザー入力に対し，必要なツールを呼び出した上で，日本語で簡潔に最終回答のみ返してください．"
        )
    elif mode == "react":
        # ReAct“風”に「段階的に」進める指示はできるが，
        # v1では思考文の出力を強制しないのが安全（モデルに内部推論させる）．
        system_prompt = (
            "あなたは家庭内アシスタントです．ユーザー要求を満たすために必要ならツールを複数回使ってください．"
            "カレンダー操作，環境取得（温湿度CO2），機器制御（エアコン，加湿器）を統合して支援します．"
            "最終回答は日本語で簡潔に，結論と実行内容を含めてください．\n\n"
            f"{FEW_SHOT}"
        )
    else:
        system_prompt = (
            "あなたは家庭内アシスタントです．可能ならツールを使って正確に答えてください．"
            "最終回答は日本語で簡潔に返してください．"
        )

    return create_agent(model=llm, tools=TOOLS, system_prompt=system_prompt)


# ------------------------------------------------------------
# 3 実行関数（zero／few／react）＋ ツール呼び出しログ抽出
# ------------------------------------------------------------

def _extract_tool_logs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    create_agent().invoke() の戻り値 state から，tool_calls を抽出する簡易ログ．
    """
    logs: List[Dict[str, Any]] = []
    msgs = state.get("messages", [])
    for m in msgs:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                logs.append(tc)
    return logs


def run_zero_shot(query: str) -> str:
    agent = _build_agent(mode="zero")
    state = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return state["messages"][-1].content


def run_few_shot(query: str) -> str:
    agent = _build_agent(mode="few")
    state = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return state["messages"][-1].content


def run_react_few_shot(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    agent = _build_agent(mode="react")
    state = agent.invoke({"messages": [{"role": "user", "content": query}]})
    output = state["messages"][-1].content
    logs = _extract_tool_logs(state)
    return output, logs


def run_agent(query: str) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    # デフォルト：react風＋few-shot
    return run_react_few_shot(query)
