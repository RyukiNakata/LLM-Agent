from __future__ import annotations

import itertools
import math
import os
import time  # レート制限対策用
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent


load_dotenv()

# ------------------------------------------------------------------
# 1. エージェント用モデルの定義 (Base: Llama, Target: GPT)
# ------------------------------------------------------------------
def build_agent_models() -> Dict[str, object]:
    """
    base：Llama（Ollama）
    target：GPT（OpenAI）
    """
    base_model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    # ユーザー指定のターゲットモデル（デフォルトは gpt-4o 推奨）
    target_model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")

    return {
        "base": ChatOllama(model=base_model_name),
        "target": ChatOpenAI(
            model=target_model_name,
            temperature=0,
            request_timeout=120,
        ),
    }

agent_models = build_agent_models()

# ------------------------------------------------------------------
# 2. 判定用モデルの定義 (Judge: GPT-4o)
# ------------------------------------------------------------------
# 判定には常に高性能なモデルを使用します
judge_llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ------------------------------------------------------------------
# 3. 評価タスク
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 4. GPTによる成功判定関数（ここを修正しました）
# ------------------------------------------------------------------
def evaluate_success(response: str, task: dict) -> bool:
    if not response:
        return False

    query = task["query"]
    
    # 判定用プロンプト：評価基準を明確に指示
    prompt = f"""
    あなたはIoTエージェントの動作評価者です。
    以下の「ユーザーの要求」に対して、「エージェントの回答」が適切かどうかを厳格に判定してください。

    ### ユーザーの要求
    {query}

    ### エージェントの回答
    {response}

    ### 判定基準
    1. 要求された情報（数値や状態）が含まれているか？（例：「温度を教えて」に対し「25度です」と答えているか）
    2. 条件付きの指示（もし〇〇なら××して）に対し、条件判定を行った形跡があるか？
    3. 実行エラーや「できませんでした」という内容で終わっていないか？
    4. 最終的にユーザーの目的が達成されたか？

    ### 出力形式
    成功の場合は "SUCCESS" 、失敗の場合は "FAILURE" とだけ出力してください。余計な文章は不要です。
    """

    try:
        # GPT-4oに判定させる
        judgment = judge_llm.invoke(prompt).content.strip()
        
        # デバッグ用に判定結果を表示しても良い
        # print(f"    [Judge Result]: {judgment}")

        return "SUCCESS" in judgment
        
    except Exception as e:
        print(f"    [Judge Error] {e}")
        return False


# ------------------------------------------------------------------
# 5. 評価実行ループ
# ------------------------------------------------------------------
def run_evaluation() -> Dict[FrozenSet[str], float]:
    print("🤖 論文に基づいた4コンポーネントの体系的評価を開始します．")
    print("   Target Model: OpenAI GPT")
    print("   Judge Model : OpenAI GPT-4o")

    components = ["Planning", "Reasoning", "Action", "Reflection"]
    model_choices = ["base", "target"]

    all_combinations = list(itertools.product(model_choices, repeat=len(components)))
    performance_scores: Dict[FrozenSet[str], float] = {}

    for i, combo in enumerate(all_combinations):
        config_map = {
            "planning_llm": agent_models[combo[0]],
            "reasoning_llm": agent_models[combo[1]],
            "action_llm": agent_models[combo[2]],
            "reflection_llm": agent_models[combo[3]],
        }

        coalition = frozenset({components[j] for j, m in enumerate(combo) if m == "target"})
        config_str = f"P:{combo[0]}, R:{combo[1]}, A:{combo[2]}, F:{combo[3]}"
        print(f"\n--- 評価中 ({i+1}/{len(all_combinations)}): [{config_str}] ---")

        agent = PaperWorkflowAgent(**config_map, tools=tools, verbose=False)

        success_count = 0
        for task in evaluation_tasks:
            try:
                response = agent.run(task["query"])
            except Exception as e:
                response = f"実行エラー: {e}"
            
            # レスポンスが長い場合は短縮表示
            clean_res = response.replace('\n', ' ')[:60]
            print(f"  - Q: {task['query'][:15]}... -> A: {clean_res}...")

            if evaluate_success(response, task):
                success_count += 1
            
            # APIレート制限対策（必要に応じて調整）
            # time.sleep(1) 

        success_rate = (success_count / len(evaluation_tasks)) * 100.0
        performance_scores[coalition] = success_rate
        print(f"--- 結果: 成功率 = {success_rate:.2f}% ---")

    return performance_scores


# ------------------------------------------------------------------
# 6. シャープレイ値計算 (変更なし)
# ------------------------------------------------------------------
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
    print(f"target（OpenAI）モデル：{os.environ.get('OPENAI_MODEL', 'gpt-4o')}")

    scores = run_evaluation()

    print("\n\n--- 📈 全16組み合わせの性能スコア (v(S)) ---")
    sorted_scores = sorted(scores.items(), key=lambda item: len(item[0]))
    for coalition, score in sorted_scores:
        coalition_name = ", ".join(sorted(list(coalition))) if coalition else "∅（全てベースモデル）"
        print(f"連合 [{coalition_name.ljust(45)}]：成功率 {score:.2f}%")

    components_list = ["Planning", "Reasoning", "Action", "Reflection"]
    shapley_results = calculate_shapley_values(scores, components_list)

    print("\n\n--- 📊 各コンポーネントのシャープレイ値（貢献度） ---")
    print("この値は，各部品を高性能モデル(GPT)に替えた際の平均的な性能向上率を示します．")
    sorted_shapley = sorted(shapley_results.items(), key=lambda item: item[1], reverse=True)
    for component, value in sorted_shapley:
        print(f"貢献度 [{component.ljust(15)}]：{value:+.2f}")
    print("--------------------------------------------------")