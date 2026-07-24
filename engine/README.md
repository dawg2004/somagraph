# SōmaGraph Engine — 馬の歩様動画解析

競走馬の歩様動画から骨格を推定し、歩様メトリクス（左右差・リズム・接地）を算出するエンジン。

技術スタックは [幻獣競馬のブログ記事](https://note.com/gugenkakeiba/n/n45032876f319) と同じ
**MMDetection + MMPose のトップダウン方式** を採用している。

```
フレーム → RTMDet で馬検出 (COCO class 17)
        → bboxで切り出し
        → HRNet-W32 (AnimalPose 20キーポイント) で骨格推定
        → 骨格オーバーレイ動画 + キーポイントCSV + ダッシュボードJSON
```

## 使用モデル

| 役割 | モデル | config名 |
|---|---|---|
| 馬検出 | RTMDet-M (COCO) | `rtmdet_m_8xb32-300e_coco` |
| 骨格推定 | HRNet-W32 (AnimalPose) | `td-hm_hrnet-w32_8xb64-210e_animalpose-256x256` |

## セットアップ (GPU推奨 / Colabなら notebooks/SomaGraph_MMPose_Engine.ipynb)

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
pip install opencv-python

# モデル取得 (engine/models/ に config + checkpoint が落ちる)
cd engine
python -m somagraph --download-models
```

### macOS (Apple Silicon) ローカルCPU実行 — 動作確認済みの手順

mmcv はソースビルドになる (10分程度)。chumpy は py3.11+ でビルド不能だが
2Dトップダウン推論には不要なので mmpose を `--no-deps` で入れる。

```bash
cd engine
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python "torch==2.1.2" "torchvision==0.16.2" "numpy<2"
uv pip install -p .venv/bin/python mmengine openmim "opencv-python<4.10" \
    "setuptools==75.8.0" wheel cython ninja
MAX_JOBS=4 uv pip install -p .venv/bin/python --no-build-isolation mmcv==2.1.0
uv pip install -p .venv/bin/python "mmdet==3.3.0"
uv pip install -p .venv/bin/python --no-deps "mmpose==1.3.2"
uv pip install -p .venv/bin/python json_tricks munkres matplotlib pillow
uv pip install -p .venv/bin/python --no-build-isolation xtcocotools

.venv/bin/python -m somagraph --download-models
.venv/bin/python -m somagraph samples/horse_walk.mp4 -o samples/results --device cpu
```

CPU (M1/8GB) での目安: 640x480・287フレームで4分弱。

## 使い方

```bash
cd engine
python -m somagraph input.mp4 -o results/          # GPU
python -m somagraph input.mp4 -o results/ --device cpu
```

出力 (`results/`):

| ファイル | 内容 |
|---|---|
| `annotated.mp4` | 骨格オーバーレイ動画（ダッシュボードUIと同配色） |
| `keypoints.csv` | `frame, time_s, keypoint, x, y, score` のロング形式 |
| `dashboard.json` | 歩様スコア・左右差・リズム等（UI表示用） |

### Webダッシュボード (アップロード解析)

```bash
cd engine
uv pip install -p .venv/bin/python fastapi "uvicorn[standard]" python-multipart
.venv/bin/python server.py        # http://localhost:8760
```

ブラウザで http://localhost:8760 (PC) / http://localhost:8760/mobile.html (スマホUI) を開き、
「動画をアップロード」から解析を実行する。進捗表示ののち、骨格オーバーレイ動画と
歩様スコア・左右差ドーナツが実測値に更新される。

API:

| エンドポイント | 内容 |
|---|---|
| `POST /api/analyze` (multipart `file`) | ジョブ投入 → `{job_id}` |
| `POST /api/analyze_url` (`{url, start_s?, duration_s?}`) | YouTube等のURLから取得して解析 |
| `GET /api/jobs/{id}` | 状態・進捗・dashboard JSON |
| `GET /api/jobs/{id}/video` | 骨格オーバーレイ動画 (H.264) |
| `GET /api/jobs/{id}/keypoints.csv` | キーポイントCSV |

URL解析は yt-dlp で取得し、既定で冒頭 **30秒** を切り出して解析する。
YouTube URLの `t=` パラメータを開始位置として拾い、ffmpegがあれば
**必要区間だけダウンロード**するので長尺動画でも取得は速い。
権利のある動画・許可された動画のみ利用すること。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `SOMAGRAPH_URL_MAX_SEC` | 30 | 解析する長さの上限(秒)。GPUなら7200(2時間)等も可 |
| `SOMAGRAPH_SOURCE_MAX_SEC` | 7200 | 受け付ける元動画の長さ上限(秒) |

処理時間の目安: CPU(M1)で約1秒/フレーム = 30秒動画で約15分。
2時間全編はCPUでは約60時間かかるため、GPU (Colab等) を推奨。
歩様メトリクスは連続歩行10〜30秒あれば十分なので、長尺動画は
`t=` でシーンを指定して解析窓を絞るのが実用的。

GPUで動かす場合は `SOMAGRAPH_DEVICE=cuda:0 python server.py`。

#### 静的ホスティング (Vercel等) から使う場合

Vercel にはPython解析サーバーが載らないため、ダッシュボードだけが配信される。
アップロード時に「解析サーバーに接続できません」のプロンプトが出るので、
手元で `server.py` を起動して `http://localhost:8760` を入力する
(URLは localStorage に保存され、次回以降は聞かれない)。

- Chrome/Edge: https ページから `http://localhost` への接続は許可される
- Safari 等でブロックされる場合や、別マシンから使う場合は
  `cloudflared tunnel --url http://localhost:8760` などで https の
  トンネルURLを作ってそれを入力する
- CORS はサーバー側で許可済み (`allow_origins=["*"]`)

### Python API

```python
from somagraph import SomaGraphEngine, EngineConfig

engine = SomaGraphEngine(EngineConfig(device="cuda:0"))
dashboard = engine.analyze_video("input.mp4", "results/")
```

## 馬体評価 (`somagraph/conformation.py`)

歩様と別軸で、動画から**立ち姿フレーム**を自動抽出して馬体を評価する。
(立ち姿写真の分類で競走成績と相関するスコアが得られたという
[報告](https://note.com/kjmd1/n/n3e1f871ed278) の枠組みを取り込んだもの)

1. **立ち姿抽出** — キーポイント信頼度 × 静止度 × 横向き度が最大のフレームを選出。
   `standing_frame.jpg` (全体) / `standing_crop.jpg` (馬クロップ) を出力
2. **体型計測** — トップライン傾き(度)・脚長/体長比・前後肢接地間隔比
   (撮影距離に依存しない相対値) を `dashboard.json` の `conformation.measurements` へ
3. **スコアリング (プラガブル)** — TorchScript分類器
   (入力 1x3x224x224 RGB [0,1] / 出力 logit) を `models/conformation.pt` か
   `SOMAGRAPH_CONFORMATION_MODEL` に置くと `conformation_score` (0..1) が出る。
   **モデルが無い場合は null** (立ち姿抽出と計測のみ動作)。
   学習データは「立ち姿クロップ × 走った/走らないラベル」で用意する

API: `GET /api/jobs/{id}/standing.jpg` で立ち姿クロップを取得できる。
UIはスコアがあるときのみ「馬体」欄を実測値 (0-100表示) に更新する。

## 血統評価 (`somagraph/pedigree.py`)

**データ源について:** studbook.jp (JAIRS) と JBIS-Search はいずれも利用規約で
「データベースの解析」や「私的利用/軽種馬生産・育成牧場の内部利用を超える
複製」を禁止しており (JBISはrobots.txtでClaudeBotも明示的に拒否)、
本機能は**それらへの自動アクセスは一切行わない**。あくまでユーザーが
手動で確認・入力した血統CSVを読み込むだけの仕組み。

1. **血統CSV** — `馬名,父,母,生年` のフラットな系図テーブル
   (`engine/data/pedigree.csv`、gitignore対象。テンプレは
   `engine/samples/pedigree.template.csv` に架空データで例示)。
   `POST /api/pedigree` でCSVアップロード → 既存データにマージ(同名は上書き)
2. **血統樹の再構築** — 指定した馬名からテーブルを再帰的に辿り、何代分
   判明しているか (`generations_known`) を計算。祖先が未登録ならその枝で打ち切り
3. **近親係数 (Wright's coefficient) の近似計算** — 父方・母方に共通して
   現れる祖先を検出し、`0.5^(n1+n2+1)` の総和で概算。クロス配合の
   手がかりになる `notable_duplicate_ancestors` も出力
4. **スコアリング (プラガブル)** — 特徴量への重み付け(JSON)を
   `models/pedigree_weights.json` に置くとロジスティック回帰的に
   `pedigree_score` (0..1) を計算。**無ければ null** (構造分析は出る)

API: `POST /api/pedigree` (CSVアップロード) / `GET /api/pedigree/{馬名}`
(動画解析なしで血統だけ確認)。動画解析時に `horse_name` を渡すと
`dashboard.json` に自動でマージされる。

## メトリクスの定義 (`somagraph/metrics.py`)

- **左右差 前肢/後肢** — 左右の蹄の上下動振幅の非対称率(%)
- **接地差** — 左右の蹄の接地時間率の差(%)。蹄が可動域最下端15%帯にいる時間で近似
- **リズム安定性** — 蹄の上下動の自己相関ピーク強度 (0-100)
- **ストライド周期** — 自己相関の最大ピーク位置 (秒)
- **歩様総合スコア** — 上記の減点方式ヒューリスティック (0-100)

メトリクス計算は numpy のみで動くため、キーポイント時系列があればモデルなしで再計算できる。

## ディレクトリ

```
engine/
  somagraph/
    skeleton.py    # AnimalPose 20kp 定義・骨格・配色
    pipeline.py    # RTMDet + HRNet 推論・動画ループ
    metrics.py     # 歩様メトリクス (numpyのみ)
    visualize.py   # 骨格描画
    export.py      # CSV / dashboard.json
    __main__.py    # CLI
  models/          # mim download 先 (git管理外)
  requirements.txt
```

## 旧実装

- `notebooks/SomaGraph_PoseEngine.ipynb` — DeepLabCut SuperAnimal 版 (legacy)
- `notebooks/opencv_poc_engine.py` — シルエットヒューリスティック PoC (legacy)
