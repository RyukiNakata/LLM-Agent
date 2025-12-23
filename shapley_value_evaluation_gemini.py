from __future__ import annotations

import itertools
import math
import os
import re
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# ★変更: Google Gemini用ライブラリをインポート
from langchain_google_genai import ChatGoogleGenerativeAI

from shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent


load_dotenv()


def build_models() -> Dict[str, object]:
    """
    base：Llama（Ollama）
    target：Gemini（Google）
    """
    base_model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    # ★変更: デフォルトモデルをGeminiに変更
    target_model_name = os.environ.get("GOOGLE_MODEL", "gemini-flash-latest")

    return {
        "base": ChatOllama(model=base_model_name),
        # ★変更: Geminiの定義
        "target": ChatGoogleGenerativeAI(
            model=target_model_name,
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        ),
    }


models = build_models()


# 元ファイルと同じ評価タスク
evaluation_tasks = [
    {"query": "これから作業をする場所を決めたいので、自宅と研究室の環境（CO2濃度や温度）を比較して、より快適な方を教えて。"},
    {"query": "今の私の心拍数が平常時より高いようなら、リラックスできるようにエアコンを冷房にして室温を少し下げて。"},
    {"query": "次の会議の開始時間を確認して、その時刻までに部屋が快適な温度になるように、今からエアコンを調整しておいて。"},
    {"query": "もし湿度が40%以下で、かつ気温が20度を下回っているなら、風邪予防のために加湿器と暖房を両方ともONにして。"},
    {"query": "外の気温と室内の気温を確認して、もし室内の方が暑ければエアコンを冷房でつけて、逆なら窓を開けるよう（換気）アドバイスして。"},
    {"query": "研究室の心拍データを確認して、誰もいなさそうなら今日は自宅で仕事をするので、自宅の環境を整えて。"},
    {"query": "明日が雨予報なら湿度が上がるはずなので、今のうちに加湿器をOFFにして、カレンダーに「傘を持っていく」と追加して。"},
    {"query": "CO2濃度が1000ppmを超えているなら集中力が下がるので教えて。もし超えていなければ、そのままエアコンで温度だけ維持して。"},
    {"query": "もうすぐ寝るので、部屋が乾燥しすぎていないか確認して。問題なければ加湿器は操作せず、エアコンだけOFFにして。"},
    {"query": "午前中の天気が荒れそうなら、エアコンをつけて暖めて。"},
    {"query": "今、部屋（自宅）のCO2濃度が低く、かつエアコンがついているようなら、無駄なのでOFFにしておいて。"},
    {"query": "最近体調が優れないので、今の部屋の環境（温湿度・CO2）と私の心拍数を見て、健康に悪そうな要因があれば解消して。"},
    {"query": "今の天気、室温、湿度、CO2を総合的に判断して、私が今一番快適に過ごせる設定にエアコンと加湿器を自動でセットして。"},
    {"query": "今日の午後に予定が入っていなければ、15時から1時間「集中作業」という予定を入れて。"},
    {"query": "帰宅したばかりで部屋がすごく暑い気がする。今の温度を確認して、28度以上なら急速冷房ですぐに涼しくして。"}
]


# --- 成功判定（元のロジックを維持）---
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
    print("   Target Model: Google Gemini")

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
            # エラーで止まらないようにtry-exceptを追加しても良いですが、
            # 元コードの振る舞いに合わせてそのまま実行します
            try:
                response = agent.run(task["query"])
            except Exception as e:
                response = f"実行エラー: {e}"
            
            print(f"  - Query: {task['query'][:20]}... -> Response: {response[:50]}...")

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
    print(f"target（Google）モデル：{os.environ.get('GOOGLE_MODEL', 'gemini-1.5-flash')}")

    scores = run_evaluation()

    print("\n\n--- 📈 全16組み合わせの性能スコア (v(S)) ---")
    sorted_scores = sorted(scores.items(), key=lambda item: len(item[0]))
    for coalition, score in sorted_scores:
        coalition_name = ", ".join(sorted(list(coalition))) if coalition else "∅（全てベースモデル）"
        print(f"連合 [{coalition_name.ljust(45)}]：成功率 {score:.2f}%")

    components_list = ["Planning", "Reasoning", "Action", "Reflection"]
    shapley_results = calculate_shapley_values(scores, components_list)

    print("\n\n--- 📊 各コンポーネントのシャープレイ値（貢献度） ---")
    print("この値は，各部品をGeminiに替えた際の平均的な性能向上率を示します．")
    sorted_shapley = sorted(shapley_results.items(), key=lambda item: item[1], reverse=True)
    for component, value in sorted_shapley:
        print(f"貢献度 [{component.ljust(15)}]：{value:+.2f}")
    print("--------------------------------------------------")