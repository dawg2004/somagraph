"""馬体(コンフォメーション)評価。

動画のキーポイント時系列から「立ち姿」に最も近いフレームを選び、
体型の客観計測と、学習済みモデルがあればスコアリングを行う。

参考: 1歳馬の立ち姿写真の二項分類で競走成績と相関するスコアが得られた
という報告 (https://note.com/kjmd1/n/n3e1f871ed278)。本モジュールは
その分類器を差し込める枠組みで、モデル無しでも立ち姿抽出と計測は動く。

スコアラー: TorchScript形式 (入力 1x3x224x224 RGB [0,1]、出力 logit 1値)。
`models/conformation.pt` または環境変数 SOMAGRAPH_CONFORMATION_MODEL のパス。
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .skeleton import KP
from .metrics import DEFAULT_KPT_THR

# 立ち姿の判定に使う主要キーポイント
_CORE = [KP["Withers"], KP["TailBase"],
         KP["L_F_Paw"], KP["R_F_Paw"], KP["L_B_Paw"], KP["R_B_Paw"]]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "conformation.pt"


def standing_frame_scores(keypoints: np.ndarray, scores: np.ndarray,
                          kpt_thr: float = DEFAULT_KPT_THR) -> np.ndarray:
    """フレームごとの「立ち姿らしさ」(0..1) を返す。

    観点: 主要キーポイントの信頼度 / 静止していること / 真横を向いていること。
    """
    T = keypoints.shape[0]
    out = np.zeros(T)
    if T < 2:
        return out

    conf = scores[:, _CORE].mean(axis=1)                       # (T,)
    body_vec = keypoints[:, KP["TailBase"]] - keypoints[:, KP["Withers"]]
    body_len = np.linalg.norm(body_vec, axis=1)                # (T,)

    # 静止度: 主要点の移動量を体長で正規化
    disp = np.linalg.norm(np.diff(keypoints[:, _CORE], axis=0), axis=2).mean(axis=1)
    motion = np.zeros(T)
    motion[1:] = disp / np.maximum(body_len[1:], 1e-6)
    # 0.5秒相当の移動平均でならす
    win = max(3, int(0.5 * 30))
    kernel = np.ones(win) / win
    motion = np.convolve(np.pad(motion, win // 2, mode="edge"), kernel, mode="valid")[:T]

    # 横向き度: 体軸の水平成分の割合 (真横=1)
    side = np.abs(body_vec[:, 0]) / np.maximum(body_len, 1e-6)

    valid = (conf >= kpt_thr) & (body_len > 1)
    stillness = np.exp(-motion / 0.01)   # 体長の1%/フレーム動くと大きく減点
    out = np.where(valid, conf * side * stillness, 0.0)
    return out


def select_standing_frame(keypoints: np.ndarray, scores: np.ndarray,
                          kpt_thr: float = DEFAULT_KPT_THR,
                          min_score: float = 0.2) -> int | None:
    """最も立ち姿らしいフレーム番号を返す。条件を満たすフレームが無ければNone。"""
    s = standing_frame_scores(keypoints, scores, kpt_thr)
    if len(s) == 0 or s.max() < min_score:
        return None
    return int(np.argmax(s))


def measure_conformation(kps: np.ndarray, scs: np.ndarray,
                         kpt_thr: float = DEFAULT_KPT_THR) -> dict:
    """1フレームのキーポイント (20,2) から体型計測値を返す。

    いずれも単位なし/度で、撮影距離に依存しない相対値。
    信頼度不足の項目は None。
    """
    def pt(name):
        i = KP[name]
        return kps[i] if scs[i] >= kpt_thr else None

    withers, tail = pt("Withers"), pt("TailBase")
    m: dict = {"topline_slope_deg": None, "leg_body_ratio": None,
               "stance_width_ratio": None}
    if withers is None or tail is None:
        return m

    body_vec = tail - withers
    body_len = float(np.linalg.norm(body_vec))
    if body_len < 1:
        return m

    # トップラインの傾き (+ = 尻高)。画像座標系はy下向きなので符号反転
    m["topline_slope_deg"] = round(
        math.degrees(math.atan2(-(tail[1] - withers[1]), abs(tail[0] - withers[0]))), 1)

    # 脚長/体長比: キ甲から前蹄までの垂直距離 ÷ 体長
    fore_paws = [p for p in (pt("L_F_Paw"), pt("R_F_Paw")) if p is not None]
    if fore_paws:
        leg_h = float(np.mean([p[1] for p in fore_paws])) - float(withers[1])
        if leg_h > 0:
            m["leg_body_ratio"] = round(leg_h / body_len, 3)

    # 前後肢の接地間隔 ÷ 体長 (立ち姿の踏込み)
    hind_paws = [p for p in (pt("L_B_Paw"), pt("R_B_Paw")) if p is not None]
    if fore_paws and hind_paws:
        fx = float(np.mean([p[0] for p in fore_paws]))
        hx = float(np.mean([p[0] for p in hind_paws]))
        m["stance_width_ratio"] = round(abs(fx - hx) / body_len, 3)
    return m


def crop_horse(frame: np.ndarray, bbox_xyxy: np.ndarray, pad: float = 0.08) -> np.ndarray:
    """bboxを少し広げて切り出す。"""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy[:4]
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = max(int(x0 - px), 0); y0 = max(int(y0 - py), 0)
    x1 = min(int(x1 + px), W); y1 = min(int(y1 + py), H)
    return frame[y0:y1, x0:x1]


def resolve_model_path() -> Path | None:
    p = os.environ.get("SOMAGRAPH_CONFORMATION_MODEL")
    path = Path(p) if p else DEFAULT_MODEL_PATH
    return path if path.exists() else None


def score_conformation(crop_bgr: np.ndarray, model_path: Path | None = None) -> float | None:
    """立ち姿クロップを 0..1 のスコアにする。モデルが無ければ None。

    モデルは TorchScript (入力 1x3x224x224 RGB [0,1]、出力 logit) を想定。
    """
    path = model_path or resolve_model_path()
    if path is None:
        return None
    import cv2
    import torch

    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        logit = model(x)
    return round(float(torch.sigmoid(logit.reshape(-1)[0])), 4)
