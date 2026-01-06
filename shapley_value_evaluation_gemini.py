from __future__ import annotations

import itertools
import math
import os
import time
import random
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# Agent用: Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# Judge用: GPT
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent


load_dotenv()

# =================================================================
# ★Gemini特化：絶対に諦めないリトライ機能
# =================================================================
def robust_invoke_llm(self, llm, system, user):
    # 設定：かなりしつこく待つ設定にします
    max_retries = 20        # 20回までリトライ（実質無限に近い）
    base_wait_seconds = 65  # Geminiの制限は1分で解除されることが多いので、65秒待つのが確実
    
    # モデル名判定
    model_name = getattr(llm, "model", "")
    is_gemma = "gemma" in model_name.lower()

    for attempt in range(max_retries):
        try:
            # Gemma対策: System Prompt結合
            if is_gemma:
                merged_content = f"Instructions:\n{system}\n\nUser Input:\n{user}"
                messages = [HumanMessage(content=merged_content)]
            else:
                messages = [SystemMessage(content=system), HumanMessage(content=user)]
            
            # 実行
            resp = llm.invoke(messages)
            return getattr(resp, "content", str(resp))
        
        except Exception as e:
            err_msg = str(e).lower()
            
            # RESOURCE_EXHAUSTED (429) 検知
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                # 待機時間: 60秒 + 回数ごとの増加 + ゆらぎ
                wait_time = base_wait_seconds + (attempt * 10) + random.uniform(0, 5)
                
                print(f"\n⚠️ Gemini API制限発生 (Attempt {attempt+1}/{max_retries})")
                print(f"   🛑 {int(wait_time)}秒間、完全に停止して回復を待ちます...")
                time.sleep(wait_time)
                print("   🔄 再開します...")
            
            # サーバーエラー (500系)
            elif "500" in err_msg or "503" in err_msg or "internal" in err_msg:
                print(f"\n⚠️ サーバーエラー。10秒待機...")
                time.sleep(10)
                
            else:
                # その他のエラーはそのまま出す
                raise e

    raise Exception("APIレート制限により、リトライ回数の上限に達しました。")

# モンキーパッチ適用
PaperWorkflowAgent._invoke_llm = robust_invoke_llm
# =================================================================


# ------------------------------------------------------------------
# 1. エージェント用モデルの定義 (Base: Llama, Target: Gemini)
# ------------------------------------------------------------------
def build_agent_models() -> Dict[str, object]:
    base_model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    
    # モデル名: 課金しているなら "gemini-1.5-flash" が最も安くて高速です
    target_model_name = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")

    print(f"🔧 Target Model: {target_model_name}")

    return {
        "base": ChatOllama(model=base_model_name),
        "target": ChatGoogleGenerativeAI(
            model=target_model_name,
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            max_retries=1, # 自前でやるので1
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
    # --- 既存のタスク (1-15) ---
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
    # {"query": "明日の朝9時に会議があるので、その1時間前にエアコンが入るようにリマインダーをカレンダーに登録しておいて。"},
    # {"query": "今日これから来客がある。部屋の空気がきれいか（CO2濃度）と、温度が適温かを確認して、ダメなら調整して。"},
    # {"query": "週末（土曜日または日曜日）なら、ゆっくり寝たいので朝の予定が入っていないか確認して。"},
#     {"query": "「集中タイム」という予定がカレンダーに入っている時間帯は、室温を低め（23度）に設定して集中しやすくして。"},
#     {"query": "今から1時間後に外出する予定がある（カレンダー確認）。それまでに部屋を冷やしておきたいので冷房をONにして。"},
#     {"query": "今日の残りの予定を確認して、もし何もなければ「リラックス」とカレンダーに入れて、照明（もし操作できれば）かエアコンを落ち着く設定にして。"},
#     {"query": "出張で数日いないので、カレンダーに「出張」と入れて、念のため全ての空調機器をOFFにして。"},
#     {"query": "外が大雪の予報なら、帰宅時に寒くないように暖房を早めにつけるようアドバイスして。"},
#     {"query": "今日は一日中雨の予報？もしそうなら、洗濯物を室内干しにするので除湿（エアコン冷房）をしておいて。"},
#     {"query": "外気温と室温の差が10度以上あるとヒートショックが怖い。差が大きければ室温を外気温に少し近づける（調整する）ようにして。"},
#     {"query": "台風が近づいている（気圧が低い、または荒天）なら、窓を閉め切るためCO2が上がりやすいから注意するよう教えて。"},
#     {"query": "これから暑くなる予報なら、今のうちにカーテンを閉めるようアドバイスして（冷房効率のため）。"},
#     {"query": "研究室のCO2濃度が自宅より低いなら「研究室へ行こう」とアドバイスし、逆なら「自宅で作業しよう」と言って。"},
#     {"query": "湿度が高く（60%以上）かつ気温も高い（28度以上）なら「不快指数が高い」と判断して、冷房と除湿を優先して。"},
#     {"query": "心拍数が低くてリラックスしている状態なら、今の環境（温度・湿度）を維持して。もし寒そう（温度が低い）なら暖房をつけて。"},
#     {"query": "部屋のCO2濃度、温度、湿度を全て確認して、すべて基準値内（CO2<1000, 18<温<28, 40<湿<60）なら「環境良好」と報告して。"},
#     {"query": "カレンダーを見て、次の予定まで30分以内なら、今の作業を中断するよう促して、エアコンを外出モード（OFFまたは弱）にして。"},
#     {"query": "もし室温が20度以下で、かつエアコンが冷房設定になっていたら、設定ミスかもしれないので暖房に切り替えて。"},
#     {"query": "今の環境データを取得して、家族（ユーザー）にメールで送るような形式で、温度・湿度・CO2をまとめて文章にして。"},
#     {"query": "現在、加湿器がついているのに湿度が30%以下なら、加湿器の水がないかもしれないので確認するよう言って。"}
]

# ------------------------------------------------------------------
# 4. GPTによる成功判定関数
# ------------------------------------------------------------------
def evaluate_success(response: str, task: dict) -> bool:
    if not response:
        return False
    query = task["query"]
    prompt = f"""
    あなたはIoTエージェントの動作評価者です。
    以下の「ユーザーの要求」に対して、「エージェントの回答」が適切かどうかを厳格に判定してください。
    ### ユーザーの要求
    {query}
    ### エージェントの回答
    {response}
    ### 判定基準
    1. 要求された情報（数値や状態）が含まれているか？
    2. 条件付きの指示に対し、条件判定を行った形跡があるか？
    3. 実行エラーや「できませんでした」という内容で終わっていないか？
    4. 最終的にユーザーの目的が達成されたか？
    ### 出力形式
    成功の場合は "SUCCESS" 、失敗の場合は "FAILURE" とだけ出力してください。
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
    print("🤖 Google Gemini 超堅牢版評価を開始します．")
    print(f"   Tasks: {len(evaluation_tasks)} tasks")

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

        # コンボ切り替え時も少し休む
        time.sleep(2)
        
        agent = PaperWorkflowAgent(**config_map, tools=tools, verbose=False)

        success_count = 0
        for idx, task in enumerate(evaluation_tasks):
            print(f"  Task ({idx+1}/{len(evaluation_tasks)}): {task['query'][:10]}...")
            try:
                response = agent.run(task["query"])
            except Exception as e:
                response = f"実行エラー: {e}"
                print(f"    ⚠️ Fatal Error: {e}")
            
            clean_res = response.replace('\n', ' ')[:60]
            print(f"    -> Agent: {clean_res}...")

            if evaluate_success(response, task):
                success_count += 1
            
            # ★重要: タスク間にも強制的に休憩を入れる (API制限のリセットを促す)
            # 有料版でもRPM制限があるため、これを入れないと連続実行で死にます
            time.sleep(2) 

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
        name = ", ".join(sorted(list(coalition))) if coalition else "All Base"
        print(f"[{name.ljust(45)}] : {score:.2f}%")
        
    shapley = calculate_shapley_values(scores, ["Planning", "Reasoning", "Action", "Reflection"])
    print("\n--- 📊 シャープレイ値 (Target: Gemini) ---")
    for k, v in sorted(shapley.items(), key=lambda x: x[1], reverse=True):
        print(f"{k.ljust(15)}: {v:+.2f}")