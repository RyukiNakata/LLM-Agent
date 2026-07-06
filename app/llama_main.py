from llama_agent import run_agent

if __name__ == "__main__":
    user_input = input("あなたの相談内容を入力してください：\n> ")
    answer, _ = run_agent(user_input)  # ← ログは捨てる
    print("\n--- AIアシスタントからの提案 ---\n")
    print(answer)