from agent import run_agent

if __name__ == "__main__":
    user_input = input("あなたの相談内容を入力してください：\n> ")
    response = run_agent(user_input)
    print("\n--- AIアシスタントからの提案 ---\n")
    print(response)
