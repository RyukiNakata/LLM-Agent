import os
import math
import itertools
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from typing import Dict, List, FrozenSet

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ユーザー環境の自作モジュール (これらが同じ階層にある前提です)
from .core.shapley_tools import tools
from .core.decomposed_agent import PaperWorkflowAgent

# 環境変数の読み込み (強制上書き設定)
load_dotenv(override=True)

# ==========================================
# 1. 評価用IoTタスク30個の定義 (実際のツールセットに最適化)
# ==========================================
EVALUATION_TASKS = [
    # --- 簡単 (Level 1: 情報取得・単一操作) ---
    "今の自宅の室温を教えて．",
    "エアコンをつけて．",
    "加湿器の現在の設定を教えて．",
    "今日の天気を教えて．",
    "今の時刻は？",
    "今の自宅のCO2濃度はどれくらい？",
    "今日の予定を教えて．",
    "加湿器をオンにして．",
    "エアコンを冷房にして．",
    "明日の15時に「ミーティング」という予定を入れて．",
    
    # --- 普通 (Level 2: 条件分岐・中程度の推論) ---
    "自宅の室温が28度以上なら冷房をつけて．",
    "湿度が40%以下になったら加湿器をオンにして．",
    "自宅の心拍データを確認して，誰もいなさそうならエアコンを消して．",
    "次の会議の開始時間を確認して，それまでに部屋を暖めておいて．",
    "外気温が自宅の室温より低ければ，冷房を消して．",
    "午前中の天気が雨予報なら，湿度が上がるので加湿器をオフにして．",
    "今の心拍数が平常時より高いようなら，冷房にして室温を少し下げて．",
    "研究室のCO2濃度が自宅より低いなら「研究室へ行こう」とアドバイスして．",
    "今日の午後に予定が入っていなければ，15時から「集中作業」という予定を入れて．",
    "CO2濃度が1000ppmを超えているなら教えて．超えていなければそのまま維持して．",
    
    # --- 難しい (Level 3: コンフリクト解決・複数ツールの連動・意図理解) ---
    "運動をして暑くなった（心拍数が高い）けど，外は寒いので急激に冷やさないようにエアコンを調整して．",
    "これから作業をする場所を決めたいので、自宅と研究室の環境（CO2濃度や温度）を比較して、より快適な方を教えて．",
    "最近体調が優れないので、今の自宅の環境（温湿度・CO2）と心拍数を見て、健康に悪そうな要因があればエアコンか加湿器で解消して．",
    "雨の予報で窓を開けられない（換気できない）けどCO2が高いから，エアコンの送風などで空気を循環させて．",
    "カレンダーを見て、次の予定まで30分以内なら今の作業を中断するよう促して、エアコンを外出モード（OFFまたは弱）にして．",
    "今の天気、室温、湿度、CO2を総合的に判断して、私が今一番快適に過ごせる設定にエアコンと加湿器を自動でセットして．",
    "研究室の心拍データを確認して誰もいなさそうなら、今日は自宅で仕事をするので「自宅作業」と予定に入れ、自宅の環境を整えて．",
    "もし湿度が40%以下で、かつ気温が20度を下回っているなら、風邪予防のために加湿器をONにして、エアコンで暖房をつけて．",
    "外気温と室温の差が10度以上あるとヒートショックが怖いので、差が大きければ室温を外気温に少し近づけるようにエアコンを調整して．",
    "今日これから来客がある（カレンダー確認）。部屋の空気がきれいか（CO2濃度）と温度が適温かを確認して、ダメなら調整して。"
]

# ==========================================
# 2. モデルの初期化 (Target & Judge: GPT-4o)
# ==========================================
def build_agent_models():
    base_model_name = os.environ.get("OLLAMA_BASE_MODEL", "llama3.1")
    target_model_name = "gpt-4o"

    print(f"🔧 Base Model  : {base_model_name} (Ollama)")
    print(f"🔧 Target Model: {target_model_name} (OpenAI)")

    # APIキーが読み込めているかチェック
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise ValueError("🚨 OPENAI_API_KEY が正しく設定されていません。.envファイルを確認してください。")

    return {
        "base": ChatOllama(model=base_model_name, temperature=0),
        "target": ChatOpenAI(model=target_model_name, temperature=0, api_key=api_key),
    }

agent_models = build_agent_models()
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# 3. 成功判定とShapley値計算関数
# ==========================================
def evaluate_success(response: str, query: str) -> bool:
    if not response or "実行エラー" in response:
        return False
    prompt = f"""
    あなたはIoTエージェントの動作評価者です。
    ユーザーの要求: {query}
    エージェントの回答: {response}
    目的が達成され、必要なツールが正しく呼ばれている場合は "SUCCESS" 、そうでない場合は "FAILURE" と出力してください。
    """
    try:
        judgment = judge_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return "SUCCESS" in judgment
    except:
        return False

def calculate_shapley_values_for_task(query: str, components: List[str]) -> Dict[str, float]:
    """1つのタスクに対して全16構成を回し、Shapley値を算出する"""
    model_choices = ["base", "target"]
    all_combinations = list(itertools.product(model_choices, repeat=len(components)))
    scores = {}

    print(f"   🔄 全16構成の評価を実行中...")
    for combo in all_combinations:
        config_map = {
            "planning_llm": agent_models[combo[0]],
            "reasoning_llm": agent_models[combo[1]],
            "action_llm": agent_models[combo[2]],
            "reflection_llm": agent_models[combo[3]],
        }
        coalition = frozenset({components[j] for j, m in enumerate(combo) if m == "target"})
        
        agent = PaperWorkflowAgent(**config_map, tools=tools, verbose=False)
        try:
            response = agent.run(query)
            is_success = evaluate_success(response, query)
            scores[coalition] = 1.0 if is_success else 0.0
        except Exception as e:
            scores[coalition] = 0.0

    # シャープレイ値の計算式
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

# ==========================================
# 4. フェーズ1: データ収集 (オフライン分析)
# ==========================================
def run_offline_analysis(csv_path="task_shapley_results.csv"):
    if os.path.exists(csv_path):
        print(f"✅ 保存済みの実験データ ({csv_path}) を読み込みます。長時間の再実行をスキップします。")
        return pd.read_csv(csv_path)

    print("⚠️ 実験データがありません。30タスク×16構成 (計480回) の評価を開始します。API課金が発生し、長時間が予想されます。")
    components = ["Planning", "Reasoning", "Action", "Reflection"]
    results = []

    for idx, task_query in enumerate(EVALUATION_TASKS):
        print(f"\n[Task {idx+1}/30]: {task_query}")
        sv = calculate_shapley_values_for_task(task_query, components)
        results.append({
            "Task": task_query,
            "Planning": sv["Planning"],
            "Reasoning": sv["Reasoning"],
            "Action": sv["Action"],
            "Reflection": sv["Reflection"]
        })
        print(f"   => P:{sv['Planning']:.2f}, R:{sv['Reasoning']:.2f}, A:{sv['Action']:.2f}, F:{sv['Reflection']:.2f}")

        # 途中で止まってもいいように毎回上書き保存
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    print(f"\n🎉 全480回の実験が完了しました！データを {csv_path} に保存しました。")
    return df

# ==========================================
# 5. フェーズ2: クラスタリングとルーター生成
# ==========================================
def create_data_driven_router(df: pd.DataFrame):
    print("\n--- 🤖 Shapley値に基づくK-Meansクラスタリングを実行中 ---")
    
    X = df[['Planning', 'Reasoning', 'Action', 'Reflection']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Reasoningの値でクラスタ番号をソート (0: 簡単, 1: 普通, 2: 難しい)
    centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(centers[:, 1])
    mapping = {old: new for new, old in enumerate(sorted_indices)}
    df['Cluster'] = df['Cluster'].map(mapping)

    # 各クラスタの代表例を抽出してFew-shotプロンプトを作成
    examples = {0: [], 1: [], 2: []}
    for cluster_id in range(3):
        cluster_tasks = df[df['Cluster'] == cluster_id]['Task'].tolist()
        examples[cluster_id] = cluster_tasks[:3] # 各クラスタから3件抽出
    
    # ルータープロンプトの動的生成
    router_prompt = f"""
    あなたはユーザーの要求を分類するアシスタントです．
    過去のShapley値クラスタリング分析に基づき，タスクは3つのクラスタに分類されます．

    【Cluster 0】(全てBaseモデルで処理可能)
    例:
    - {examples[0][0]}
    - {examples[0][1]}
    
    【Cluster 1】(ReasoningにTargetモデルが必要)
    例:
    - {examples[1][0]}
    - {examples[1][1]}
    
    【Cluster 2】(ReasoningとActionにTargetモデルが必要)
    例:
    - {examples[2][0]}
    - {examples[2][1]}

    以下の新しい要求が，どのクラスタ（0, 1, 2）に最も近いか予測し，数字のみを出力してください．
    """
    return router_prompt

# ==========================================
# 6. 動的ルーティング・エージェント本体
# ==========================================
class DataDrivenDynamicAgent:
    def __init__(self, base_llm, target_llm, tools, router_prompt_base):
        self.base_llm = base_llm
        self.target_llm = target_llm
        self.tools = tools
        self.router_llm = base_llm
        self.router_prompt_base = router_prompt_base

    def run(self, query: str) -> str:
        # 1. 動的生成されたプロンプトでクラスタ予測
        prompt = self.router_prompt_base + f"\n新しい要求: {query}"
        try:
            res = self.router_llm.invoke([SystemMessage(content="数字のみ回答．"), HumanMessage(content=prompt)])
            cluster_id = int(res.content.strip())
            cluster_id = max(0, min(2, cluster_id))
        except:
            cluster_id = 2 # エラー時は安全側に倒す

        # 2. クラスタごとの構成割り当て (Shapley値分析に基づく)
        print(f"📊 ルーティング判定: Cluster {cluster_id}")
        if cluster_id == 0:
            config = {"P": self.base_llm, "R": self.base_llm, "A": self.base_llm, "F": self.base_llm}
            print("   -> 構成: [All Base (Llama)]")
        elif cluster_id == 1:
            config = {"P": self.base_llm, "R": self.target_llm, "A": self.base_llm, "F": self.base_llm}
            print("   -> 構成: [Reasoning 強化 (GPT-4o)]")
        else:
            config = {"P": self.base_llm, "R": self.target_llm, "A": self.target_llm, "F": self.base_llm}
            print("   -> 構成: [Reasoning & Action 強化 (GPT-4o)]")

        # 3. 実行
        agent = PaperWorkflowAgent(
            planning_llm=config["P"], reasoning_llm=config["R"],
            action_llm=config["A"], reflection_llm=config["F"], tools=self.tools
        )
        return agent.run(query)

# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    # フェーズ1: データ収集 (時間がかかります。完了後はCSVから読み込みます)
    df_results = run_offline_analysis("task_shapley_results.csv")
    
    # フェーズ2: クラスタリングとルータープロンプトの生成
    dynamic_prompt = create_data_driven_router(df_results)
    
    # フェーズ3: オンライン実行テスト
    print("\n--- 🚀 Data-Driven Agent Online Test ---")
    dynamic_agent = DataDrivenDynamicAgent(
        base_llm=agent_models["base"], 
        target_llm=agent_models["target"], 
        tools=tools,
        router_prompt_base=dynamic_prompt
    )
    
    test_queries = [
        "エアコンの電源を切って．",
        "自宅の室温が30度を超えたら，冷房を最強にして．",
        "これから友達が来るから，みんなが快適に過ごせるように空調と環境を調整して．"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        response = dynamic_agent.run(q)
        print(f"Answer:\n{response}")