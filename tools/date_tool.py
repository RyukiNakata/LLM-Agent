from datetime import datetime

# @tool デコレータは付けないでください
def get_current_datetime(query: str = "") -> str:
    """
    現在の日付と時刻を返します。
    """
    now = datetime.now()
    # 日本語で分かりやすく返す
    return now.strftime("%Y年%m月%d日 %H時%M分%S秒")

# テスト用
if __name__ == "__main__":
    print(get_current_datetime())