import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

# ツールとエージェント本体のみインポート
from .core.shapley_tools import tools
from .core.decomposed_agent import PaperWorkflowAgent

load_dotenv()

# ==========================================
# 1. モデルの定義 (インポートエラー回避のためここに直接書く)
# ==========================================
def build_agent_models():
    base_model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    return {
        "base": ChatOllama(model=base_model_name, temperature=0),
        "target": ChatAnthropic(
            model="claude-sonnet-4-6",  # 最新のClaudeを指定
            temperature=0,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=1,
        ),
    }

agent_models = build_agent_models()

# ==========================================
# 2. 動的ルーティングエージェントの定義
# ==========================================
class DynamicWorkflowAgent:
    def __init__(self, base_llm, target_llm, tools, verbose=False):
        self.base_llm = base_llm
        self.target_llm = target_llm
        self.tools = tools
        self.verbose = verbose
        # ルーティング用には高速・低コストなBaseモデルを使用
        self.router_llm = base_llm 

    def analyze_complexity(self, query: str) -> int:
        """タスクの難易度を1〜3で判定するルーター"""
        prompt = f"""
        あなたはIoTスマートホームのタスク分類器です．
        以下のユーザーの要求の複雑度を1，2，3のいずれかで判定し，数字のみを出力してください．

        レベル1: 単純な情報取得や，条件分岐のない単一の機器操作（例: 「今の室温を教えて」「エアコンをつけて」）
        レベル2: 複数の情報を組み合わせた推論や，簡単な条件付き操作（例: 「室温が28度以上なら冷房をつけて」）
        レベル3: 矛盾の解決，未来の予測，ユーザーの意図を汲み取った高度な環境調整（例: 「雨で換気できないがCO2を下げて」「集中したいので環境を整えて」）

        ユーザーの要求: {query}
        """
        try:
            messages = [SystemMessage(content="数字のみを回答してください．"), HumanMessage(content=prompt)]
            response = self.router_llm.invoke(messages)
            level = int(response.content.strip())
            return max(1, min(3, level)) # 1~3に収める
        except Exception as e:
            print(f"[Router Error] {e}．Defaulting to Level 3．")
            return 3 # エラー時は安全のため最高難易度にする

    def run(self, query: str) -> str:
        # 1. クエリの難易度を判定
        complexity = self.analyze_complexity(query)
        
        # 2. シャープレイ値に基づく動的構成の決定
        # Reflectionは常にBase（Shapley: -4.72）
        # Planningは常にBase（Shapley: +5.83）
        if complexity == 1:
            config = {"P": self.base_llm, "R": self.base_llm, "A": self.base_llm, "F": self.base_llm}
            print(f"🔄 ルーティング: Level 1 (All Base)")
        elif complexity == 2:
            config = {"P": self.base_llm, "R": self.target_llm, "A": self.base_llm, "F": self.base_llm}
            print(f"🔄 ルーティング: Level 2 (Reasoning強化)")
        else:
            config = {"P": self.base_llm, "R": self.target_llm, "A": self.target_llm, "F": self.base_llm}
            print(f"🔄 ルーティング: Level 3 (Reasoning & Action強化)")

        # 3. PaperWorkflowAgentを動的に生成して実行
        agent = PaperWorkflowAgent(
            planning_llm=config["P"],
            reasoning_llm=config["R"],
            action_llm=config["A"],
            reflection_llm=config["F"],
            tools=self.tools,
            verbose=self.verbose
        )
        
        return agent.run(query)

# ==========================================
# 3. 実行テスト
# ==========================================
if __name__ == "__main__":
    print("🤖 Dynamic Routing Agent のテストを開始します．..")
    
    # エージェントの初期化
    dynamic_agent = DynamicWorkflowAgent(
        base_llm=agent_models["base"], 
        target_llm=agent_models["target"], 
        tools=tools,
        verbose=True # 思考プロセスを表示する
    )
    
    test_query = "帰宅したばかりで部屋がすごく暑い気がする．今の温度を確認して，28度以上なら急速冷房ですぐに涼しくして．"
    print(f"\nユーザーの要求: {test_query}")
    
    response = dynamic_agent.run(test_query)
    print("\n最終回答:")
    print(response)