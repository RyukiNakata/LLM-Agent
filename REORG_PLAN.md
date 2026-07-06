# フォルダ再編 実行手順書 (REORG_PLAN)

本書は**実行手順のドキュメント**です。本書自体を読んだ時点では何も変更されません。
実行は「GO」承認後、**段階ごとに**行ってください。一括実行はしないこと。

## 前提・共通ルール

- 作業ディレクトリは常に**リポジトリルート** `iot-llm-agents/` とする。
- 仮想環境を有効化: `source .venv/bin/activate`
- 実行は必ず `python -m <pkg.module>`（ルートから）。`python app/main.py` 形式は
  `from tools.X` / 相対import が壊れるので**禁止**。
- 判断確定事項:
  - (a) `tools/` は**ルート据置**（+19編集を回避）
  - (b) shapley core は `shapley_tools.py` **据置名** / `decomposed_agent.py` へ**リネーム**
  - (c) `config.py` は**今回移動しない（凍結）**
- `new_repo` の追跡解除は**本再編に含めない**（末尾「別タスク」参照）。
- 鍵/APIキーのローテーション（Tier 0）は範囲外。
- 各段階は **①git mv のみ → ②import編集** の 2 コミットに分ける
  （移動と編集を混ぜると rename 検出が閾値割れし `git log --follow` が切れるため）。

### 検証方針（重要）
- **構文チェック**: `python -m py_compile <file>`（import解決は見ない）
- **起動だけ確認**: `python -c "import <pkg.module>"` を使う。
  - 理由: `python -m <pkg.module>` は `if __name__ == "__main__"` 配下の**実処理（LLM/API呼び出し）まで走る**。
    本確認では実処理を走らせたくないので、`import` でモジュール読込（＝import解決＋モジュール直下コード）だけを実施し、
    `__main__` ブロックは実行しない。
  - APIキー非依存にするため、ダミーキーを一時付与して実行（`.env` の `load_dotenv()` は既定 `override=False` なので
    シェル側の環境変数が優先され、実キーに触れずに済む）:
    ```
    OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import <pkg.module>"
    ```
    → ImportError が出なければ合格。モジュール直下での LLM クライアント構築はオフライン（ネットワーク未使用）。

### 各段階の開始前に必ず実施（ロールバック用アンカー）
```
git rev-parse HEAD          # 出力SHAを「段階Nの開始点」として控える
git status --short          # 変更が無いクリーン状態から始める
```

---

## 段階 1: app 系

### 1-1. 追加する `__init__.py`
- `app/__init__.py`（空ファイルでよい）

### 1-2. 移動コマンド（git mv / plain mv の区別）
対象はすべて**追跡済み** → `git mv`。
```
mkdir -p app
git mv agent.py            app/agent.py
git mv llama_agent.py      app/llama_agent.py
git mv main.py             app/main.py
git mv llama_main.py       app/llama_main.py
git mv agent_runner.py     app/runner.py        # ★リネーム
: > app/__init__.py        # 空 __init__.py 作成
git add app/__init__.py
```

### 1-3. コミット① （移動のみ・内容変更なし）
```
git commit -m "reorg(app): move agent/main entrypoints into app/ package"
```

### 1-4. import 編集（file:行 旧→新）
※ 行番号は現行ファイル基準。リネーム後ファイルで該当行を確認して編集すること。
```
app/main.py:1        from agent import run_agent          →  from app.agent import run_agent
app/runner.py:2      from agent import run_agent          →  from app.agent import run_agent
app/llama_main.py:1  from llama_agent import run_agent    →  from app.llama_agent import run_agent
```
- `app/agent.py` / `app/llama_agent.py` の `from tools.X ...` は **無編集**（tools はルート据置）。

### 1-5. コミット② （import編集）
```
git add app/main.py app/runner.py app/llama_main.py
git commit -m "reorg(app): fix imports to app.* package paths"
```

### 1-6. 検証
```
# 構文
python -m py_compile app/agent.py app/llama_agent.py app/main.py app/llama_main.py app/runner.py
# 起動だけ確認（実処理は走らせない）
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.agent"
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.llama_agent"
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.main"
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.llama_main"
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.runner"
```
- 期待: すべて ImportError 無しで戻る。
- この段階では `google_calendar_api.py` はルート据置のまま（`tools/calendar_tool.py` の
  `from google_calendar_api import` はまだ有効）なので import チェーンは通る。

### 1-7. ロールバック（段階1）
- コミット前（編集途中）: `git restore app/main.py app/runner.py app/llama_main.py`
- 移動を戻す:
  ```
  git mv app/agent.py agent.py
  git mv app/llama_agent.py llama_agent.py
  git mv app/main.py main.py
  git mv app/llama_main.py llama_main.py
  git mv app/runner.py agent_runner.py
  rm app/__init__.py && rmdir app
  ```
- コミット済みをまとめて破棄（未push前提）: `git reset --hard <段階1の開始点SHA>` の後 `rmdir app`（残れば）。

---

## 段階 2: research 系

### 2-1. 追加する `__init__.py`（3個）
- `research/__init__.py`
- `research/shapley/__init__.py`
- `research/shapley/core/__init__.py`

### 2-2. 移動コマンド（追跡/未追跡を区別）
**追跡済み → `git mv`:**
```
mkdir -p research/shapley/core
git mv shapley_decomposed_agent.py     research/shapley/core/decomposed_agent.py   # ★リネーム
git mv shapley_tools.py                research/shapley/core/shapley_tools.py       # 据置名
git mv shapley_value_evaluation.py     research/shapley/eval_base.py                # ★(無印=base)
git mv shapley_value_evaluation_ec2.py     research/shapley/eval_ec2.py
git mv shapley_value_evaluation_gemini.py  research/shapley/eval_gemini.py
git mv shapley_value_evaluation_gemma.py   research/shapley/eval_gemma.py
git mv shapley_value_evaluation_llama.py   research/shapley/eval_llama.py
git mv shapley_value_evaluation_qwen.py    research/shapley/eval_qwen.py
```
**未追跡（`git status` が `??`）→ plain mv ＋ git add:**
```
mv router.py                          research/shapley/router.py
mv auto_shapley_router_gpt.py         research/shapley/auto_router_gpt.py           # ★リネーム
mv shapley_value_evaluation_claude.py research/shapley/eval_claude.py
```
**__init__.py 作成:**
```
: > research/__init__.py
: > research/shapley/__init__.py
: > research/shapley/core/__init__.py
git add research/__init__.py research/shapley/__init__.py research/shapley/core/__init__.py
git add research/shapley/router.py research/shapley/auto_router_gpt.py research/shapley/eval_claude.py
```
- 注: 未追跡3ファイルは元々 rename 履歴が無いため、コミット①で新規追加として入れてよい。

### 2-3. コミット① （移動のみ）
```
git commit -m "reorg(research): move shapley eval scripts and core into research/shapley/"
```

### 2-4. import 編集（各ファイル2行・相対import化）
共通置換:
```
from shapley_tools import tools
    → from .core.shapley_tools import tools
from shapley_decomposed_agent import PaperWorkflowAgent
    → from .core.decomposed_agent import PaperWorkflowAgent
```
対象 file:行（現行ファイル基準。移動後ファイルの該当行を確認して編集）:
```
research/shapley/eval_base.py:13,14
research/shapley/eval_ec2.py:13,14
research/shapley/eval_claude.py:18,19
research/shapley/eval_gemini.py:18,19
research/shapley/eval_gemma.py:16,17
research/shapley/eval_llama.py:15,16
research/shapley/eval_qwen.py:15,16
research/shapley/router.py:9,10
research/shapley/auto_router_gpt.py:15,16
```
- `research/shapley/core/shapley_tools.py` の `from tools.X ...` は **無編集**（tools ルート据置）。
- `research/shapley/core/decomposed_agent.py` は内部import無し → **無編集**。

### 2-5. コミット② （import編集）
```
git add research/shapley/eval_base.py research/shapley/eval_ec2.py research/shapley/eval_claude.py \
        research/shapley/eval_gemini.py research/shapley/eval_gemma.py research/shapley/eval_llama.py \
        research/shapley/eval_qwen.py research/shapley/router.py research/shapley/auto_router_gpt.py
git commit -m "reorg(research): fix shapley imports to relative .core.* paths"
```

### 2-6. 検証
```
# 構文
python -m py_compile research/shapley/core/decomposed_agent.py research/shapley/core/shapley_tools.py \
  research/shapley/eval_base.py research/shapley/eval_ec2.py research/shapley/eval_claude.py \
  research/shapley/eval_gemini.py research/shapley/eval_gemma.py research/shapley/eval_llama.py \
  research/shapley/eval_qwen.py research/shapley/router.py research/shapley/auto_router_gpt.py
# 起動だけ確認（実処理は走らせない・相対importは -m 文脈が必要なので import で確認）
for m in eval_base eval_ec2 eval_claude eval_gemini eval_gemma eval_llama eval_qwen router auto_router_gpt; do
  OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy \
    python -c "import research.shapley.$m" && echo "OK $m" || echo "FAIL $m"
done
```
- 期待: 全 `OK`。相対import (`from .core...`) は `python -c "import research.shapley.<m>"`
  でパッケージ文脈が与えられるため解決する。

### 2-7. ロールバック（段階2）
- 編集途中: `git restore research/shapley/eval_*.py research/shapley/router.py research/shapley/auto_router_gpt.py`
- 追跡済みの移動を戻す（`git mv` 逆順、代表例）:
  ```
  git mv research/shapley/core/decomposed_agent.py shapley_decomposed_agent.py
  git mv research/shapley/core/shapley_tools.py     shapley_tools.py
  git mv research/shapley/eval_base.py              shapley_value_evaluation.py
  # ...eval_ec2/gemini/gemma/llama/qwen も同様に元名へ...
  ```
- 未追跡だったファイルを戻す（plain mv）:
  ```
  mv research/shapley/router.py            router.py
  mv research/shapley/auto_router_gpt.py   auto_shapley_router_gpt.py
  mv research/shapley/eval_claude.py       shapley_value_evaluation_claude.py
  ```
- `__init__.py` とディレクトリ削除: `rm research/shapley/core/__init__.py research/shapley/__init__.py research/__init__.py && rmdir research/shapley/core research/shapley research`
- まとめて破棄（未push前提）: `git reset --hard <段階2の開始点SHA>`（＋未追跡ファイルは上記 plain mv で手戻し）。

---

## 段階 3: integrations 系

### 3-1. 追加する `__init__.py`
- `integrations/__init__.py`

### 3-2. 移動コマンド（追跡済み → git mv）
```
mkdir -p integrations
git mv google_calendar_api.py integrations/google_calendar_api.py
: > integrations/__init__.py
git add integrations/__init__.py
```

### 3-3. コミット① （移動のみ）
```
git commit -m "reorg(integrations): move google_calendar_api into integrations/ package"
```

### 3-4. import 編集（file:行 旧→新）
```
tools/calendar_tool.py:4  from google_calendar_api import get_upcoming_events, add_event
    → from integrations.google_calendar_api import get_upcoming_events, add_event
```

### 3-5. コミット② （import編集）
```
git add tools/calendar_tool.py
git commit -m "reorg(integrations): point tools.calendar_tool to integrations.google_calendar_api"
```

### 3-6. 検証（この段階は tools/ 共有依存が変わるため app/research も再確認）
```
python -m py_compile integrations/google_calendar_api.py tools/calendar_tool.py
# integrations 単体
OPENAI_API_KEY=dummy python -c "import integrations.google_calendar_api"
# 共有依存の再確認（tools 経由で integrations を引くルート）
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import app.agent"
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy python -c "import research.shapley.eval_base"
```
- 期待: 全て ImportError 無し。`tools/calendar_tool.py` が `integrations.google_calendar_api` を
  正しく解決できることを確認する。

### 3-7. ロールバック（段階3）
- 編集途中: `git restore tools/calendar_tool.py`
- 移動を戻す:
  ```
  git mv integrations/google_calendar_api.py google_calendar_api.py
  rm integrations/__init__.py && rmdir integrations
  ```
- まとめて破棄: `git reset --hard <段階3の開始点SHA>`

---

## 最終検証（3段階すべて完了後）

### A. 全ファイル 構文一括チェック
```
python -m py_compile \
  app/agent.py app/llama_agent.py app/main.py app/llama_main.py app/runner.py \
  research/shapley/core/decomposed_agent.py research/shapley/core/shapley_tools.py \
  research/shapley/eval_base.py research/shapley/eval_ec2.py research/shapley/eval_claude.py \
  research/shapley/eval_gemini.py research/shapley/eval_gemma.py research/shapley/eval_llama.py \
  research/shapley/eval_qwen.py research/shapley/router.py research/shapley/auto_router_gpt.py \
  integrations/google_calendar_api.py \
  tools/__init__.py tools/ac_tool.py tools/calendar_tool.py tools/co2_tool.py tools/date_tool.py \
  tools/humidifier_tool.py tools/humidity_tool.py tools/sensor_tool.py tools/temp_tool.py tools/weather_tool.py \
  config.py
```
- 期待: 28 ファイル（再編後の配置）すべて OK。`config.py` は凍結（据置）。

### B. 実行コマンド一覧の「起動だけ確認」チェックリスト（地図#3準拠）
各行を `python -c "import <module>"`（ダミーキー付き）で ImportError が出ないこと。
実運用時の実コマンドは右列（実処理が走るので最終検証では実行しない）。

| モジュール（import確認用） | 実運用コマンド |
|---|---|
| `import app.main` | `python -m app.main` |
| `import app.llama_main` | `python -m app.llama_main` |
| `import app.runner` | `python -m app.runner` |
| `import research.shapley.eval_base` | `python -m research.shapley.eval_base` |
| `import research.shapley.eval_claude` | `python -m research.shapley.eval_claude` |
| `import research.shapley.eval_gemini` | `python -m research.shapley.eval_gemini` |
| `import research.shapley.eval_gemma` | `python -m research.shapley.eval_gemma` |
| `import research.shapley.eval_llama` | `python -m research.shapley.eval_llama` |
| `import research.shapley.eval_qwen` | `python -m research.shapley.eval_qwen` |
| `import research.shapley.eval_ec2` | `python -m research.shapley.eval_ec2` |
| `import research.shapley.router` | `python -m research.shapley.router` |
| `import research.shapley.auto_router_gpt` | `python -m research.shapley.auto_router_gpt` |
| `import integrations.google_calendar_api` | `python -m integrations.google_calendar_api` |

一括チェック例:
```
for m in app.main app.llama_main app.runner \
         research.shapley.eval_base research.shapley.eval_claude research.shapley.eval_gemini \
         research.shapley.eval_gemma research.shapley.eval_llama research.shapley.eval_qwen \
         research.shapley.eval_ec2 research.shapley.router research.shapley.auto_router_gpt \
         integrations.google_calendar_api; do
  OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy GOOGLE_API_KEY=dummy \
    python -c "import $m" >/dev/null 2>&1 && echo "OK  $m" || echo "FAIL $m"
done
```

### C. 実行時リスクの再確認（地図#7準拠・コード変更はしない／確認のみ）
- R1/R2: `token.json` `credentials.json` `task_shapley_results.csv` は**CWD基準**。
  再編後も**必ずリポジトリルートから `python -m` 実行**すれば従来どおりルート直下を参照。
- R3: `load_dotenv()` はCWD上方探索 → ルート実行なら root `.env` を発見。
- R5: `python <path>.py` 形式は使わない（ImportError）。常に `-m`。

---

## 別タスク（本再編には含めない）: new_repo の追跡解除

**本段階1〜3では実施しない。** 再編完了・安定後に、別途承認の上で実施する候補。
```
# 事前確認（読み取りのみ・既に実施済み: 中身は .gitignore 1ファイル / 未コミット作業なし / remoteなし）
git -C new_repo status --short
git -C new_repo log --oneline -1
# 追跡解除（gitlink を外す。フォルダ実体は残す）
git rm --cached new_repo
# 必要なら .gitignore に new_repo/ を追記してコミット
git commit -m "chore: untrack accidental embedded repo new_repo (gitlink)"
```
- 失うものは無い（.gitignore のみで実コードなし）。フォルダ実体を消すかは別判断。

---

## 完了条件
- 段階1〜3 の各コミット①②が作成され、各段階検証がパス。
- 最終検証 A（py_compile 一括）と B（全 `-m` モジュールの import 確認）が全 OK。
- `config.py` / `tools/` / venv / `pytz-layer` / `new_repo` は本再編で移動していない
  （`pytz-layer` の infra/ 移動と new_repo 追跡解除は別タスクとして分離）。
