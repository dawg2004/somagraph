"""動画解析パイプライン (トップダウン方式)。

参考: https://note.com/gugenkakeiba/n/n45032876f319 と同じ構成
  各フレーム → RTMDet で馬検出 (COCOクラス17)
            → バウンディングボックスで切り出し
            → HRNet-W32 (AnimalPose 20kp) で骨格推定
            → 可視化フレーム + キーポイント時系列

依存: torch / mmengine / mmcv / mmdet / mmpose (GPU推奨)。
モデルの取得は `somagraph.pipeline.download_models()` か
`mim download` (README参照)。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from .skeleton import COCO_HORSE_CLASS_ID, KEYPOINT_NAMES
from .metrics import compute_gait_metrics
from .visualize import draw_skeleton
from .export import write_keypoints_csv, write_dashboard_json

# mim download で使うモデル名 (config/checkpointが models/ に落ちる)
DET_MODEL_NAME = "rtmdet_m_8xb32-300e_coco"
POSE_MODEL_NAME = "td-hm_hrnet-w32_8xb64-210e_animalpose-256x256"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclasses.dataclass
class EngineConfig:
    det_config: str | None = None      # None なら model_dir から自動解決
    det_checkpoint: str | None = None
    pose_config: str | None = None
    pose_checkpoint: str | None = None
    model_dir: Path = DEFAULT_MODEL_DIR
    device: str = "cuda:0"
    det_score_thr: float = 0.5
    kpt_score_thr: float = 0.3
    horse_class_id: int = COCO_HORSE_CLASS_ID
    max_detect_miss: int = 10  # 検出が途切れても直前bboxを使い続ける最大フレーム数


def download_models(model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
    """mim で検出・ポーズ両モデルのconfigとcheckpointを取得する。"""
    from mim import download
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    download("mmdet", [DET_MODEL_NAME], dest_root=str(model_dir))
    download("mmpose", [POSE_MODEL_NAME], dest_root=str(model_dir))


def _resolve(model_dir: Path, stem: str, suffix: str) -> str:
    matches = sorted(model_dir.glob(f"{stem}*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"{model_dir} に {stem}*{suffix} が見つかりません。"
            f"somagraph.pipeline.download_models() を先に実行してください。")
    return str(matches[-1])


class SomaGraphEngine:
    """RTMDet + HRNet(AnimalPose) のトップダウン推論エンジン。"""

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        from mmdet.apis import init_detector
        from mmpose.apis import init_model
        from mmpose.utils import adapt_mmdet_pipeline

        c = self.cfg
        det_config = c.det_config or _resolve(c.model_dir, DET_MODEL_NAME, ".py")
        det_ckpt = c.det_checkpoint or _resolve(c.model_dir, DET_MODEL_NAME.split("_8xb")[0], ".pth")
        pose_config = c.pose_config or _resolve(c.model_dir, POSE_MODEL_NAME, ".py")
        pose_ckpt = c.pose_checkpoint or _resolve(c.model_dir, "hrnet_w32_animalpose", ".pth")

        self.detector = init_detector(det_config, det_ckpt, device=c.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_model = init_model(pose_config, pose_ckpt, device=c.device)

    # ---- 1フレーム処理 ----

    def detect_horse(self, frame: np.ndarray) -> np.ndarray | None:
        """フレームから最も確からしい馬のbbox(xyxy)を返す。いなければNone。"""
        from mmdet.apis import inference_detector
        result = inference_detector(self.detector, frame)
        inst = result.pred_instances.cpu().numpy()
        keep = (inst.labels == self.cfg.horse_class_id) & \
               (inst.scores >= self.cfg.det_score_thr)
        if not keep.any():
            return None
        bboxes = inst.bboxes[keep]
        scores = inst.scores[keep]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas * scores))  # 大きく写っている個体を主対象にする
        return bboxes[best]

    def estimate_pose(self, frame: np.ndarray,
                      bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """bbox内の馬の20キーポイントと信頼度を返す。"""
        from mmpose.apis import inference_topdown
        results = inference_topdown(self.pose_model, frame, bbox_xyxy[None, :4])
        pred = results[0].pred_instances
        keypoints = np.asarray(pred.keypoints[0], dtype=float)        # (20, 2)
        scores = np.asarray(pred.keypoint_scores[0], dtype=float)     # (20,)
        return keypoints, scores

    # ---- 動画処理 ----

    def analyze_video(self, video_path: str | Path, out_dir: str | Path,
                      render: bool = True, progress: bool = True) -> dict:
        """動画全体を解析して out_dir に結果一式を書き出す。

        出力:
          annotated.mp4  — 骨格オーバーレイ動画 (render=True時)
          keypoints.csv  — フレーム×関節のロング形式
          dashboard.json — 歩様メトリクス (ダッシュボード用)
        戻り値: dashboard.json と同内容のdict
        """
        video_path = Path(video_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"動画を開けません: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if render:
            writer = cv2.VideoWriter(
                str(out_dir / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        n_kp = len(KEYPOINT_NAMES)
        all_kps: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        last_bbox: np.ndarray | None = None
        miss = 0
        t = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                bbox = self.detect_horse(frame)
                if bbox is None and last_bbox is not None and miss < self.cfg.max_detect_miss:
                    bbox = last_bbox      # 一時的な検出落ちは直前のbboxで補う
                    miss += 1
                elif bbox is not None:
                    last_bbox = bbox
                    miss = 0

                if bbox is not None:
                    kps, scores = self.estimate_pose(frame, bbox)
                else:
                    kps = np.zeros((n_kp, 2))
                    scores = np.zeros(n_kp)
                all_kps.append(kps)
                all_scores.append(scores)

                if writer is not None:
                    vis = draw_skeleton(frame, kps, scores, self.cfg.kpt_score_thr) \
                        if bbox is not None else frame
                    writer.write(vis)

                t += 1
                if progress and total and t % 30 == 0:
                    print(f"\r  {t}/{total} frames", end="", flush=True)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if progress:
            print(f"\r  {t} frames done")

        keypoints = np.stack(all_kps) if all_kps else np.zeros((0, n_kp, 2))
        scores = np.stack(all_scores) if all_scores else np.zeros((0, n_kp))

        write_keypoints_csv(out_dir / "keypoints.csv", keypoints, scores, fps)
        gait = compute_gait_metrics(keypoints, scores, fps, self.cfg.kpt_score_thr)
        return write_dashboard_json(out_dir / "dashboard.json", gait, video_path)
