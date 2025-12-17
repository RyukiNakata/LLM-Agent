from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class ToolCallSpec:
    name: str
    args: Dict[str, Any]


class PaperWorkflowAgent:
    """
    4コンポーネント（Planning, Reasoning, Action, Reflection）を別LLMで回す評価用エージェント．
    tools は @tool で生成された Tool オブジェクトを想定し，tool.invoke で実行する．
    """

    def __init__(
        self,
        planning_llm: BaseChatModel,
        reasoning_llm: BaseChatModel,
        action_llm: BaseChatModel,
        reflection_llm: BaseChatModel,
        tools: List[Any],
        verbose: bool = False,
    ):
        self.planning_llm = planning_llm
        self.reasoning_llm = reasoning_llm
        self.action_llm = action_llm
        self.reflection_llm = reflection_llm
        self.tools = tools
        self.verbose = verbose

        self.tool_map = {t.name: t for t in tools if getattr(t, "name", None)}

    def _invoke_llm(self, llm: BaseChatModel, system: str, user: str) -> str:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return getattr(resp, "content", str(resp))

    def _parse_tool_calls(self, text: str) -> List[ToolCallSpec]:
        text = (text or "").strip()

        # そのまま JSON の場合
        try:
            obj = json.loads(text)
            return self._obj_to_calls(obj)
        except Exception:
            pass

        # 前後に余計な文がある場合は最初の { ... } を拾う
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                obj = json.loads(candidate)
                return self._obj_to_calls(obj)
            except Exception:
                return []

        return []

    def _obj_to_calls(self, obj: Any) -> List[ToolCallSpec]:
        if not isinstance(obj, dict):
            return []
        calls = obj.get("tool_calls", [])
        if not isinstance(calls, list):
            return []
        out: List[ToolCallSpec] = []
        for c in calls:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            args = c.get("args", {})
            if isinstance(name, str) and isinstance(args, dict):
                out.append(ToolCallSpec(name=name, args=args))
        return out

    def _execute_tool_call(self, call: ToolCallSpec) -> str:
        tool = self.tool_map.get(call.name)
        if tool is None:
            raise ValueError(f"unknown tool: {call.name}")
        if not hasattr(tool, "invoke"):
            raise TypeError(f"tool has no invoke(): {call.name}")

        result = tool.invoke(call.args or {})
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def run(self, query: str) -> str:
        """
        成功時は必ず「タスク成功」を含める．失敗時は必ず「タスク失敗」を含める．
        """

        tool_names = ", ".join(sorted(self.tool_map.keys()))

        # --- Planning ---
        planning_sys = (
            "あなたはタスクの計画担当です．ユーザー要求を達成するために，必要な手順を短く計画してください．"
            "ツールが必要なら，どの情報を取得するかも明示してください．"
        )
        plan = self._invoke_llm(self.planning_llm, planning_sys, f"ユーザー要求：{query}")

        # --- Reasoning（ツール名を厳格に制約する）---
        reasoning_sys = (
            "あなたは推論担当です．ユーザー要求を達成するために，必要ならツールを呼び出します．\n"
            "利用可能なツール名は次だけです．この中からのみ選んでください．\n"
            f"{tool_names}\n\n"
            "出力は次の厳格なJSONだけです．余計な文章は禁止です．\n"
            "{\n"
            '  "tool_calls": [\n'
            '    {"name": "tool_name", "args": {"arg1": "value1"}}\n'
            "  ]\n"
            "}\n\n"
            "制約：\n"
            "1．name は必ず上記の利用可能ツール名のいずれか\n"
            "2．args はそのツールの引数に合わせる（不要なら空dict）\n"
            "3．ツール不要なら tool_calls は空配列\n"
        )

        tool_plan_text = self._invoke_llm(
            self.reasoning_llm,
            reasoning_sys,
            f"計画：\n{plan}\n\nユーザー要求：\n{query}",
        )
        tool_calls = self._parse_tool_calls(tool_plan_text)

        # 不正ツール名が混ざったら，1回だけ自動修復
        invalid = [c for c in tool_calls if c.name not in self.tool_map]
        if invalid:
            repair_sys = (
                "あなたの出力には存在しないツール名が含まれています．"
                "利用可能なツール名の中から選び直して，JSONのみを出力してください．\n"
                f"利用可能：{tool_names}\n"
            )
            tool_plan_text = self._invoke_llm(
                self.reasoning_llm,
                repair_sys + "\n\n" + reasoning_sys,
                f"ユーザー要求：{query}\n\n前回の出力：\n{tool_plan_text}",
            )
            tool_calls = self._parse_tool_calls(tool_plan_text)

        # --- Action（ツール実行）---
        observations: List[str] = []
        try:
            for call in tool_calls:
                obs = self._execute_tool_call(call)
                observations.append(f"{call.name}({call.args}) -> {obs}")
        except Exception as e:
            reflection_sys = (
                "あなたは振り返り担当です．失敗の原因を1文で特定し，最小の修正方針を1つ提案してください．"
                "出力は日本語で簡潔にしてください．"
            )
            reflection = self._invoke_llm(
                self.reflection_llm,
                reflection_sys,
                f"ユーザー要求：{query}\n計画：{plan}\nツール計画出力：{tool_plan_text}\n例外：{repr(e)}",
            )
            return f"タスク失敗．Reflection：{reflection}"

        # --- Answer（観測に基づき回答生成）---
        answer_sys = (
            "あなたは実行結果をまとめてユーザーに回答する担当です．"
            "観測結果に基づいて，日本語で簡潔に結論を述べてください．"
            "数値がある場合は具体的な数値を含めてください．"
            "条件付き依頼（例：1000ppmを超えたら換気）には必ず対応してください．"
        )
        answer_user = (
            f"ユーザー要求：{query}\n\n"
            f"計画：\n{plan}\n\n"
            "観測結果：\n" + "\n".join(observations if observations else ["（ツール呼び出しなし）"])
        )
        answer = self._invoke_llm(self.action_llm, answer_sys, answer_user)

        # --- Reflection（成功時も短く）---
        reflection_sys_ok = (
            "あなたは振り返り担当です．今回の回答がユーザー要求を満たしているかを一言で判定し，"
            "改善点があれば一言で述べてください．"
        )
        reflection_ok = self._invoke_llm(
            self.reflection_llm,
            reflection_sys_ok,
            f"ユーザー要求：{query}\n回答：{answer}\n観測：\n" + "\n".join(observations),
        )

        return f"タスク成功．{answer}\nReflection：{reflection_ok}"
