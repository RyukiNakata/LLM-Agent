from __future__ import annotations

import itertools
import math
import os
import re
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent


load_dotenv()


def build_models() -> Dict[str, object]:
    """
    base：Llama（Ollama）
    target：GPT（OpenAI）
    """
    base_model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    target_model_name = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    return {
        "base": ChatOllama(model=base_model_name),
        "target": ChatOpenAI(
            model=target_model_name,
            temperature=0,
            request_timeout=int(os.environ.get("OPENAI_TIMEOUT", "120")),
        ),
    }


models = build_models()


# evaluation_tasks = [
#     {"query": "今の湿度は．"},
#     {"query": "現在の部屋のCO2濃度を教えて．もし1000ppmを超えていたら換気を促して．"},
# ]
evaluation_tasks = [
    {"query": "今の気温，湿度，CO2濃度を確認して，これから3時間快適に過ごせるよう必要な対処をして．"},
    {"query": "今日は在宅勤務なので，集中できる室内環境か確認して，問題があれば調整して．"},
    {"query": "明日雨が降るなら，午前中の予定をオンライン会議に変更して．"},
    {"query": "次の会議の前に，空気が悪くならないよう部屋の状態を整えて．"},
    {"query": "今の部屋の状態を説明して，改善できる点があれば実行して．"},
    {"query": "今日の予定と天気を考慮して，エアコンを使うべきか判断して．"},
    {"query": "湿度が低く，かつCO2濃度が高い状態なら，適切に対処して．"},
    {"query": "今日の午後に向けて，快適な室内環境になるよう準備して．"},
    {"query": "今の室内環境が健康に悪影響がないかチェックして．"},
    {"query": "来客があるので，その前に部屋を快適な状態に整えて．"},
]


# --- 成功判定（キーワードではなく要件ベースにする）---
def evaluate_success(response: str, task: dict) -> bool:
    if not response:
        return False

    q = task["query"]

    # 湿度：% が含まれていれば成功（例：45.3%）
    if "湿度" in q:
        return bool(re.search(r"(\d+(\.\d+)?)\s*%", response))

    # CO2：ppm が含まれ，換気に言及していれば成功
    if ("CO2" in q) or ("二酸化炭素" in q):
        ok_ppm = bool(re.search(r"(\d+(\.\d+)?)\s*ppm", response, flags=re.IGNORECASE))
        mention_vent = ("換気" in response) or ("窓" in response) or ("空気" in response)
        return ok_ppm and mention_vent

    # その他：最低限「タスク成功」を含むか（保険）
    return "タスク成功" in response


def run_evaluation() -> Dict[FrozenSet[str], float]:
    print("🤖 論文に基づいた4コンポーネントの体系的評価を開始します．")

    components = ["Planning", "Reasoning", "Action", "Reflection"]
    model_choices = ["base", "target"]

    all_combinations = list(itertools.product(model_choices, repeat=len(components)))
    performance_scores: Dict[FrozenSet[str], float] = {}

    for i, combo in enumerate(all_combinations):
        config_map = {
            "planning_llm": models[combo[0]],
            "reasoning_llm": models[combo[1]],
            "action_llm": models[combo[2]],
            "reflection_llm": models[combo[3]],
        }

        coalition = frozenset({components[j] for j, m in enumerate(combo) if m == "target"})
        config_str = f"P:{combo[0]}, R:{combo[1]}, A:{combo[2]}, F:{combo[3]}"
        print(f"\n--- 評価中 ({i+1}/{len(all_combinations)}): [{config_str}] ---")

        agent = PaperWorkflowAgent(**config_map, tools=tools, verbose=False)

        success_count = 0
        for task in evaluation_tasks:
            response = agent.run(task["query"])
            print(f"  - Query: {task['query']} -> Response: {response}")

            if evaluate_success(response, task):
                success_count += 1

        success_rate = (success_count / len(evaluation_tasks)) * 100.0
        performance_scores[coalition] = success_rate
        print(f"--- 結果: 成功率 = {success_rate:.2f}% ---")

    return performance_scores


def calculate_shapley_values(
    performance_scores: Dict[FrozenSet[str], float],
    components: List[str],
) -> Dict[str, float]:
    shapley_values = {comp: 0.0 for comp in components}
    n = len(components)

    for component_i in components:
        other_components = [c for c in components if c != component_i]

        for k in range(len(other_components) + 1):
            for S_tuple in itertools.combinations(other_components, k):
                S = frozenset(S_tuple)
                S_with_i = S.union({component_i})

                v_S = performance_scores.get(S, 0.0)
                v_S_with_i = performance_scores.get(S_with_i, 0.0)

                marginal_contribution = v_S_with_i - v_S
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                shapley_values[component_i] += weight * marginal_contribution

    return shapley_values


if __name__ == "__main__":
    print(f"base（Ollama）モデル：{os.environ.get('OLLAMA_MODEL', 'llama3.1')}")
    print(f"target（OpenAI）モデル：{os.environ.get('OPENAI_MODEL', 'gpt-4.1')}")

    scores = run_evaluation()

    print("\n\n--- 📈 全16組み合わせの性能スコア (v(S)) ---")
    sorted_scores = sorted(scores.items(), key=lambda item: len(item[0]))
    for coalition, score in sorted_scores:
        coalition_name = ", ".join(sorted(list(coalition))) if coalition else "∅（全てベースモデル）"
        print(f"連合 [{coalition_name.ljust(45)}]：成功率 {score:.2f}%")

    components_list = ["Planning", "Reasoning", "Action", "Reflection"]
    shapley_results = calculate_shapley_values(scores, components_list)

    print("\n\n--- 📊 各コンポーネントのシャープレイ値（貢献度） ---")
    print("この値は，各部品を高性能モデルに替えた際の平均的な性能向上率を示します．")
    sorted_shapley = sorted(shapley_results.items(), key=lambda item: item[1], reverse=True)
    for component, value in sorted_shapley:
        print(f"貢献度 [{component.ljust(15)}]：{value:+.2f}")
    print("--------------------------------------------------")
