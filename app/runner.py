import time
from agent import run_agent

def run_fixed_query_loop(interval_minutes=2):
    print(" 初回のみクエリを入力してください。その後、15分おきに同じ内容を自動実行します。")
    user_query = input("\n 実行したいクエリを入力: ").strip()

    if not user_query:
        print("クエリが空のため、プログラムを終了します。")
        return

    try:
        while True:
            print(f"\n {time.strftime('%Y-%m-%d %H:%M:%S')} - 実行クエリ: {user_query}")
            try:
                result = run_agent(user_query)
                print(f"\n エージェント応答:\n{result}")
            except Exception as e:
                print(f" エラーが発生しました: {e}")
            print(f"\n 次の実行まで {interval_minutes} 分待機します...")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\n プログラムを終了しました。")

if __name__ == "__main__":
    run_fixed_query_loop()
