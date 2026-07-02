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

### Python API

```python
from somagraph import SomaGraphEngine, EngineConfig

engine = SomaGraphEngine(EngineConfig(device="cuda:0"))
dashboard = engine.analyze_video("input.mp4", "results/")
```

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
