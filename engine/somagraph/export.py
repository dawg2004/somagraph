"""解析結果のエクスポート: keypoints.csv と dashboard.json。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .skeleton import KEYPOINT_NAMES


def write_keypoints_csv(path: str | Path, keypoints: np.ndarray,
                        scores: np.ndarray, fps: float) -> None:
    """フレーム×関節のロング形式CSV。"""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "keypoint", "x", "y", "score"])
        for t in range(keypoints.shape[0]):
            for i, name in enumerate(KEYPOINT_NAMES):
                w.writerow([
                    t,
                    round(t / fps, 4),
                    name,
                    round(float(keypoints[t, i, 0]), 2),
                    round(float(keypoints[t, i, 1]), 2),
                    round(float(scores[t, i]), 4),
                ])


def grade_of(score: float | None) -> str | None:
    if score is None:
        return None
    for threshold, grade in [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C+")]:
        if score >= threshold:
            return grade
    return "C"


def build_dashboard_json(gait: dict, video_path: str | Path,
                         conformation: dict | None = None,
                         pedigree: dict | None = None) -> dict:
    """ダッシュボード(index.html / mobile.html)が読む想定のJSON。

    歩様系は動画から導出。馬体(conformation)は立ち姿フレームの計測と
    学習済みモデルがあればスコア。血統(pedigree)はユーザーが用意した
    血統CSVに馬名が見つかった場合のみ埋まる。気性などは対応データが無くnull。
    """
    score = gait.get("gait_score")
    conf_score = (conformation or {}).get("score")
    ped_score = (pedigree or {}).get("score")
    return {
        "source_video": str(video_path),
        "engine": "somagraph-mmpose",
        "gait_score": score,
        "gait_grade": grade_of(score),
        "lateral": {
            "foreleg_asymmetry_pct": gait.get("fore_asymmetry_pct"),
            "hindleg_asymmetry_pct": gait.get("hind_asymmetry_pct"),
            "ground_contact_asymmetry_pct": gait.get("contact_asymmetry_pct"),
        },
        "rhythm_stability": gait.get("rhythm_stability"),
        "stride_period_s": gait.get("stride_period_s"),
        "frames": gait.get("frames"),
        "fps": gait.get("fps"),
        # 馬体: 立ち姿フレームからの計測 + プラガブルな分類器スコア
        "conformation": conformation,
        "conformation_score": conf_score,
        # 血統: ユーザー提供の血統CSVからの構造分析 + プラガブルなスコア
        "pedigree": pedigree,
        "pedigree_score": ped_score,
        # 動画・血統CSVから導出できない項目 (外部データで埋める)
        "temperament_score": None,
        "growth_potential": None,
        "injury_risk_grade": None,
    }


def write_dashboard_json(path: str | Path, gait: dict, video_path: str | Path,
                         conformation: dict | None = None,
                         pedigree: dict | None = None) -> dict:
    data = build_dashboard_json(gait, video_path, conformation, pedigree)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
