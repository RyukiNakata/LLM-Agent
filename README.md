# iot-llm-agents

家電・カレンダーを制御する LangChain エージェントと、Shapley 値による
マルチLLM 貢献度評価・ルーティングの実験コード群。

## 構成概要

| 領域 | 主なファイル |
|------|-------------|
| エージェント本体 | `agent.py` (OpenAI版) / `llama_agent.py` (Ollama版) |
| 実行入口 | `main.py` / `llama_main.py` / `agent_runner.py` |
| ツール群 (IoT/天気/カレンダー) | `tools/` |
| Shapley 評価 | `shapley_value_evaluation*.py`, `shapley_decomposed_agent.py`, `shapley_tools.py` |
| ルーティング | `router.py`, `auto_shapley_router_gpt.py` |
| カレンダーAPI | `google_calendar_api.py` |

## セットアップ

### 1. 仮想環境の有効化

本リポジトリでは `llmagent`（Python 3.10）を使用します。

```bash
source llmagent/bin/activate
```

新規に作る場合:

```bash
python3 -m venv llmagent
source llmagent/bin/activate
pip install -r requirements.txt        # 整理後は requirements.proposed.txt を参照
```

### 2. 環境変数 (.env) の設定

雛形 `.env.example` をコピーして各キーを設定します（`.env` は Git 管理外）。

```bash
cp .env.example .env
# エディタで実際のAPIキーを記入
```

主なキー: `OPENAI_API_KEY`（judge 用で全 shapley 版が必須）, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `OPENWEATHER_API_KEY`。詳細は `.env.example` を参照。

### 3. Ollama（ローカルLLM）の起動

`base` モデルや Ollama 系 target を使う場合に必要です。

```bash
ollama serve                 # サーバ起動 (別ターミナル)
ollama pull llama3.1         # 既定 base モデル
# 必要に応じて: ollama pull qwen2.5:32b など
```

### 4. 外部依存（実行時に到達が必要なもの）

- **AWS API Gateway**: IoT センサー/家電制御 API（`tools/` が参照）
- **Google Calendar API**: `credentials.json` / `token.json`（初回 OAuth）
- **各クラウド LLM**: OpenAI / Anthropic / Google GenAI（該当キー必須）

## 実行方法（エントリポイント）

| コマンド | 内容 |
|----------|------|
| `python main.py` | OpenAI 版エージェントを対話実行 |
| `python llama_main.py` | ローカル(Ollama)版エージェントを実行 |
| `python agent_runner.py` | エージェントを連続実行するランナー |
| `python shapley_value_evaluation.py` | Shapley 評価（基準/GPT-4o target） |
| `python shapley_value_evaluation_claude.py` | 同・Claude target 版 |
| `python shapley_value_evaluation_gemini.py` | 同・Gemini 版 |
| `python shapley_value_evaluation_gemma.py` | 同・Gemma 版 |
| `python shapley_value_evaluation_llama.py` | 同・Llama 版 |
| `python shapley_value_evaluation_qwen.py` | 同・Qwen 版 |
| `python shapley_value_evaluation_ec2.py` | 同・Ollama 同士（EC2運用想定） |
| `python router.py` | Ollama/Claude による動的ルーティング |
| `python auto_shapley_router_gpt.py` | KMeans+Shapley による自動ルーティング |
| `python google_calendar_api.py` | Google Calendar 初回 OAuth |

> 各 shapley 版の違いは主に **target LLM の種類**と一部のリトライ処理のみで、
> judge には共通して `gpt-4o`（OpenAI）を使用します。

## 注意

- `.env`, `*.pem`, `credentials.json`, `token.json` は機密情報のためコミットしないでください（`.gitignore` 済み）。
- 依存パッケージの整理版は `requirements.proposed.txt` を参照してください。
