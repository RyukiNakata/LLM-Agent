import itertools
import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

# LangChainのモデルクラス
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI  # ★変更点1: Gemini用ライブラリ

# エージェントとツールのインポート
from shapley_decomposed_agent import PaperWorkflowAgent
from shapley_tools import tools  # ツール定義ファイルから

# 環境変数の読み込み
load_dotenv()

# ------------------------------------------------------------------
# 1. モデルの定義
# ------------------------------------------------------------------

# Base Model (軽量・ローカル): Llama 3.1 8B
base_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# Target Model (高性能): Gemini 2.0 Flash (または Pro)
# ★変更点2: Geminiに変更
target_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", # または "gemini-1.5-pro" など
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# コンポーネント名の定義
COMPONENTS = ["Planning", "Reasoning", "Action", "Reflection"]

# ------------------------------------------------------------------
# 2. 評価タスクの定義
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
# 3. エージェント構築・実行関数
# ------------------------------------------------------------------
async def evaluate_combination(combo_indices: List[int], task_id: int, query: str) -> bool:
    """
    指定されたコンポーネントだけをTarget(Gemini)にし，残りをBase(Llama)にする
    """
    
    # デフォルトはBase
    models = {
        "Planning": base_llm,
        "Reasoning": base_llm,
        "Action": base_llm,
        "Reflection": base_llm,
    }

    # 指定されたインデックス（コンポーネント）だけTargetに差し替え
    combo_names = []
    for idx in combo_indices:
        comp_name = COMPONENTS[idx]
        models[comp_name] = target_llm
        combo_names.append(comp_name)
    
    combo_str = ", ".join(combo_names) if combo_names else "∅（全てベースモデル）"
    print(f"\n--- Task {task_id+1} | Combo: [{combo_str}] ---")

    # エージェント構築
    agent = PaperWorkflowAgent(
        planning_llm=models["Planning"],
        reasoning_llm=models["Reasoning"],
        action_llm=models["Action"],
        reflection_llm=models["Reflection"],
        tools=tools,
        verbose=True
    )

    # 実行
    try:
        result = agent.run(query)
        print(f"Result: {result[:100]}...") # ログ省略
        
        # 成功判定
        is_success = "タスク成功" in result
        return is_success
    except Exception as e:
        print(f"Error: {e}")
        return False

# ------------------------------------------------------------------
# 4. メイン処理（全組み合わせ実行 & シャープレイ値計算）
# ------------------------------------------------------------------
async def main():
    print(f"Base Model: {base_llm.model}")
    print(f"Target Model: {target_llm.model} (Gemini)")
    
    # 全組み合わせ (2^4 = 16通り)
    combinations = []
    for r in range(len(COMPONENTS) + 1):
        for combo in itertools.combinations(range(len(COMPONENTS)), r):
            combinations.append(list(combo))
    
    # 結果格納用
    # results[combo_tuple] = success_rate (0.0 ~ 1.0)
    results: Dict[tuple, float] = {}

    for combo in combinations:
        success_count = 0
        total_tasks = len(evaluation_tasks)
        
        combo_names = [COMPONENTS[i] for i in combo]
        combo_str = ", ".join(combo_names) if combo_names else "∅"
        print(f"\n=== Testing Combination: [{combo_str}] ===")

        for i, task in enumerate(evaluation_tasks):
            is_success = await evaluate_combination(combo, i, task["query"])
            if is_success:
                success_count += 1
        
        success_rate = success_count / total_tasks
        results[tuple(combo)] = success_rate
        print(f"Combination [{combo_str}] Success Rate: {success_rate:.2%}")

    # --------------------------------------------------------------
    # 5. シャープレイ値の計算
    # --------------------------------------------------------------
    print("\n\n--- 📈 全16組み合わせの性能スコア (v(S)) ---")
    for combo, score in results.items():
        names = [COMPONENTS[i] for i in combo]
        name_str = ", ".join(names) if names else "∅（全てベースモデル）"
        print(f"連合 [{name_str:<40}]：成功率 {score:.2%}")

    print("\n\n--- 📊 各コンポーネントのシャープレイ値（貢献度） ---")
    print("この値は，各部品をGeminiに替えた際の平均的な性能向上率を示します．")
    
    import math

    n = len(COMPONENTS)
    shapley_values = {i: 0.0 for i in range(n)}

    # 定義通りの計算式: sum [ (|S|! * (n-|S|-1)!) / n! ] * (v(S U {i}) - v(S))
    for i in range(n):
        shapley_sum = 0.0
        
        # iを含まない全ての部分集合Sを探す
        for combo in combinations:
            if i in combo:
                continue # iが含まれていたらスキップ
            
            # S
            S = tuple(combo)
            # S U {i}
            S_union_i = tuple(sorted(list(combo) + [i]))
            
            v_S = results[S]
            v_S_union_i = results[S_union_i]
            
            marginal_contribution = v_S_union_i - v_S
            
            # 重み計算
            s_len = len(S)
            weight = (math.factorial(s_len) * math.factorial(n - s_len - 1)) / math.factorial(n)
            
            shapley_sum += weight * marginal_contribution
        
        shapley_values[i] = shapley_sum

    # 表示
    for i in range(n):
        print(f"貢献度 [{COMPONENTS[i]:<10}]：{shapley_values[i] * 100:+.2f}")

if __name__ == "__main__":
    asyncio.run(main())