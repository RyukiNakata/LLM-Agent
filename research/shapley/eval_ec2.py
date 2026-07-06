from __future__ import annotations

import itertools
import math
import os
import time
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent

load_dotenv()

# =================================================================
# ⚙️ 設定 (EC2用)
# =================================================================
# Base: 軽量モデル (8B)
BASE_MODEL_NAME = "llama3.1" # 8Bは自動で落ちてくるか、別途 pull してください

# Target: 高性能モデル (EC2上の Llama 3.1 70B)
TARGET_MODEL_NAME = "llama3.1:70b"

NUM_TEST_TASKS = 50
print(f"🔧 Config: Base={BASE_MODEL_NAME}, Target={TARGET_MODEL_NAME} (On EC2)")
# =================================================================


# ------------------------------------------------------------------
# 1. エージェント用モデルの定義
# ------------------------------------------------------------------
def build_agent_models() -> Dict[str, object]:
    return {
        "base": ChatOllama(
            model=BASE_MODEL_NAME,
            temperature=0,
        ),
        "target": ChatOllama(
            model=TARGET_MODEL_NAME,
            temperature=0,
            # EC2の強力なGPUを使うため、コンテキスト長を長めに確保
            num_ctx=8192, 
            # 頻繁なロード/アンロードを防ぐ
            keep_alive="24h",
        ),
    }

agent_models = build_agent_models()

# ------------------------------------------------------------------
# 2. 判定用モデル (Judge: GPT-4o)
# ------------------------------------------------------------------
judge_llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ------------------------------------------------------------------
# 3. 評価タスク (50件)
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
    {"query": "帰宅したばかりで部屋がすごく暑い気がする。今の温度を確認して、28度以上なら急速冷房ですぐに涼しくして。"},
    # --- 新規追加タスク (16-50) ---
    {"query": "今、誰も部屋にいない（心拍データがない）なら、エアコンと加湿器がつけっぱなしになっていないか確認して、ついていたらOFFにして。"},
    {"query": "室温が24度から26度の範囲内なら快適なので、エアコンがついている場合は電気代節約のためにOFFにして。"},
    {"query": "外の天気が「晴れ」で、かつ外気温が20度以上なら、暖房は不要なのでOFFにして。"},
    {"query": "CO2濃度が低い（600ppm以下）なら換気の必要はないので、今の室温を保つようにエアコンを調整して。"},
    {"query": "寝ている間（深夜2時から5時）にエアコンがついていたら、体に悪いのでOFFにするよう設定して（今の時刻を確認して判断）。"},
    {"query": "肌の乾燥が気になるので、湿度が50%を下回っていたら加湿器をつけて。上回っていたら何もしなくていいよ。"},
    {"query": "熱中症が心配なので、室温が30度を超えている、または湿度が70%を超えているなら、すぐに冷房を最強にして。"},
    {"query": "寒気がするので、現在の室温を確認して、もし22度以下なら暖房を入れて25度まで上げて。"},
    {"query": "心拍数が100を超えていて興奮状態のようだから、深呼吸するようにアドバイスして、室温を少し下げて。"},
    {"query": "部屋のカビ対策をしたい。湿度が80%を超えているなら、加湿器をOFFにして、エアコンを除湿（ドライ）モードにするか、冷房をつけて。"},
    {"query": "朝起きて喉が痛い。今の湿度を確認して、もし乾燥していたら加湿器をONにして。"},
    {"query": "CO2濃度が高くなると頭痛がするので、今の値を確認して800ppm以上なら「換気推奨」と教えて。"},
    {"query": "運動をして暑くなった。室温に関わらず、とりあえずエアコンを冷房にして風を送って。"},
    {"query": "花粉症で窓を開けられないので、CO2濃度が上がらないようにエアコンで空気を循環させて（送風か冷房）。"},
    {"query": "お風呂上がりで暑いので、現在の設定温度より2度下げて。"},
    {"query": "明日の朝9時に会議があるので、その1時間前にエアコンが入るようにリマインダーをカレンダーに登録しておいて。"},
    {"query": "今日これから来客がある。部屋の空気がきれいか（CO2濃度）と、温度が適温かを確認して、ダメなら調整して。"},
    {"query": "週末（土曜日または日曜日）なら、ゆっくり寝たいので朝の予定が入っていないか確認して。"},
    {"query": "「集中タイム」という予定がカレンダーに入っている時間帯は、室温を低め（23度）に設定して集中しやすくして。"},
    {"query": "今から1時間後に外出する予定がある（カレンダー確認）。それまでに部屋を冷やしておきたいので冷房をONにして。"},
    {"query": "今日の残りの予定を確認して、もし何もなければ「リラックス」とカレンダーに入れて、照明（もし操作できれば）かエアコンを落ち着く設定にして。"},
    {"query": "出張で数日いないので、カレンダーに「出張」と入れて、念のため全ての空調機器をOFFにして。"},
    {"query": "外が大雪の予報なら、帰宅時に寒くないように暖房を早めにつけるようアドバイスして。"},
    {"query": "今日は一日中雨の予報？もしそうなら、洗濯物を室内干しにするので除湿（エアコン冷房）をしておいて。"},
    {"query": "外気温と室温の差が10度以上あるとヒートショックが怖い。差が大きければ室温を外気温に少し近づける（調整する）ようにして。"},
    {"query": "台風が近づいている（気圧が低い、または荒天）なら、窓を閉め切るためCO2が上がりやすいから注意するよう教えて。"},
    {"query": "これから暑くなる予報なら、今のうちにカーテンを閉めるようアドバイスして（冷房効率のため）。"},
    {"query": "研究室のCO2濃度が自宅より低いなら「研究室へ行こう」とアドバイスし、逆なら「自宅で作業しよう」と言って。"},
    {"query": "湿度が高く（60%以上）かつ気温も高い（28度以上）なら「不快指数が高い」と判断して、冷房と除湿を優先して。"},
    {"query": "心拍数が低くてリラックスしている状態なら、今の環境（温度・湿度）を維持して。もし寒そう（温度が低い）なら暖房をつけて。"},
    {"query": "部屋のCO2濃度、温度、湿度を全て確認して、すべて基準値内（CO2<1000, 18<温<28, 40<湿<60）なら「環境良好」と報告して。"},
    {"query": "カレンダーを見て、次の予定まで30分以内なら、今の作業を中断するよう促して、エアコンを外出モード（OFFまたは弱）にして。"},
    {"query": "もし室温が20度以下で、かつエアコンが冷房設定になっていたら、設定ミスかもしれないので暖房に切り替えて。"},
    {"query": "今の環境データを取得して、家族（ユーザー）にメールで送るような形式で、温度・湿度・CO2をまとめて文章にして。"},
    {"query": "現在、加湿器がついているのに湿度が30%以下なら、加湿器の水がないかもしれないので確認するよう言って。"},
    {"query": "自宅の温度が研究室の温度より5度以上高いなら、帰宅前に冷房をつけておく必要があるか判断して。"},
    {"query": "私の心拍数が高いのに、部屋も暑い（28度以上）なら危険。緊急で冷房をつけて、水分補給をするようカレンダーに「水」と入れて。"}
]

evaluation_tasks = evaluation_tasks[:NUM_TEST_TASKS]

# ------------------------------------------------------------------
# 4. 判定関数
# ------------------------------------------------------------------
def evaluate_success(response: str, task: dict) -> bool:
    if not response: return False
    query = task["query"]
    prompt = f"""
    あなたはIoTエージェントの動作評価者です。
    ユーザー要求: {query}
    エージェント回答: {response}
    
    判定基準:
    1. 要求された情報が含まれているか
    2. 条件判定を行っているか
    3. エラーで終わっていないか
    
    成功なら "SUCCESS" 、失敗なら "FAILURE" と出力。
    """
    try:
        judgment = judge_llm.invoke(prompt).content.strip()
        is_success = "SUCCESS" in judgment
        judge_icon = "✅ SUCCESS" if is_success else "❌ FAILURE"
        print(f"    └─ Judge: {judge_icon}")
        return is_success
    except Exception as e:
        print(f"    [Judge Error] {e}")
        return False

# ------------------------------------------------------------------
# 5. 評価実行ループ
# ------------------------------------------------------------------
def run_evaluation() -> Dict[FrozenSet[str], float]:
    print("🤖 EC2 (Llama 3.1 70B) による評価を開始します．")
    
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
        for idx, task in enumerate(evaluation_tasks):
            print(f"  Task ({idx+1}/{len(evaluation_tasks)}): {task['query'][:10]}...")
            try:
                response = agent.run(task["query"])
            except Exception as e:
                response = f"実行エラー: {e}"
                print(f"    ⚠️ Error: {e}")
            
            clean_res = response.replace('\n', ' ')[:60]
            print(f"    -> Agent: {clean_res}...")

            if evaluate_success(response, task):
                success_count += 1

        success_rate = (success_count / len(evaluation_tasks)) * 100.0
        performance_scores[coalition] = success_rate
        print(f"--- 結果: 成功率 = {success_rate:.2f}% ---")

    return performance_scores

# ------------------------------------------------------------------
# 6. シャープレイ値計算
# ------------------------------------------------------------------
def calculate_shapley_values(scores, components):
    shapley_values = {comp: 0.0 for comp in components}
    n = len(components)
    for component_i in components:
        other_components = [c for c in components if c != component_i]
        for k in range(len(other_components) + 1):
            for S_tuple in itertools.combinations(other_components, k):
                S = frozenset(S_tuple)
                S_with_i = S.union({component_i})
                v_S = scores.get(S, 0.0)
                v_S_with_i = scores.get(S_with_i, 0.0)
                marginal = v_S_with_i - v_S
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n))
                shapley_values[component_i] += weight * marginal
    return shapley_values

if __name__ == "__main__":
    scores = run_evaluation()
    print("\n\n--- 📈 評価結果 ---")
    for coalition, score in sorted(scores.items(), key=lambda x: len(x[0])):
        name = ", ".join(sorted(list(coalition))) if coalition else "∅（全てベースモデル）"
        print(f"[{name.ljust(45)}] : {score:.2f}%")
        
    shapley = calculate_shapley_values(scores, ["Planning", "Reasoning", "Action", "Reflection"])
    print("\n--- 📊 シャープレイ値 (Target: Llama 70B on EC2) ---")
    for k, v in sorted(shapley.items(), key=lambda x: x[1], reverse=True):
        print(f"{k.ljust(15)}: {v:+.2f}")