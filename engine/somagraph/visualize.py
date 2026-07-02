"""骨格オーバーレイ描画 (OpenCV)。ダッシュボードUIと同じ配色。"""
from __future__ import annotations

import cv2
import numpy as np

from .skeleton import BONES, GROUP_COLORS_BGR

POINT_COLOR_BGR = (76, 168, 201)   # gold
POINT_EDGE_BGR = (24, 13, 8)       # #080d18


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                  kpt_thr: float = 0.3, bone_thickness: int = 2,
                  point_radius: int = 4) -> np.ndarray:
    """1フレームに骨格を描画して返す (入力は変更しない)。

    keypoints: (20, 2), scores: (20,)
    """
    vis = frame.copy()
    for a, b, group in BONES:
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue
        pa = tuple(np.round(keypoints[a]).astype(int))
        pb = tuple(np.round(keypoints[b]).astype(int))
        cv2.line(vis, pa, pb, GROUP_COLORS_BGR[group], bone_thickness, cv2.LINE_AA)
    for i in range(keypoints.shape[0]):
        if scores[i] < kpt_thr:
            continue
        p = tuple(np.round(keypoints[i]).astype(int))
        cv2.circle(vis, p, point_radius + 1, POINT_EDGE_BGR, -1, cv2.LINE_AA)
        cv2.circle(vis, p, point_radius, POINT_COLOR_BGR, -1, cv2.LINE_AA)
    return vis


def draw_bbox(frame: np.ndarray, bbox_xyxy: np.ndarray,
              color_bgr: tuple = (180, 204, 45)) -> np.ndarray:
    vis = frame.copy()
    x0, y0, x1, y1 = np.round(bbox_xyxy[:4]).astype(int)
    cv2.rectangle(vis, (x0, y0), (x1, y1), color_bgr, 1, cv2.LINE_AA)
    return vis
