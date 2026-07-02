"""SōmaGraph 動画解析エンジン (MMDetection + MMPose トップダウン方式)。

軽量モジュール(skeleton / metrics / export / visualize)はnumpy+OpenCVのみで動く。
推論(pipeline.SomaGraphEngine)には mmdet / mmpose が必要。
"""
from .skeleton import KEYPOINT_NAMES, KP, BONES
from .metrics import compute_gait_metrics
from .export import build_dashboard_json

__version__ = "0.1.0"

__all__ = [
    "KEYPOINT_NAMES", "KP", "BONES",
    "compute_gait_metrics", "build_dashboard_json",
    "SomaGraphEngine", "EngineConfig", "download_models",
]


def __getattr__(name):
    # pipeline は mmdet/mmpose 依存なので遅延インポート
    if name in ("SomaGraphEngine", "EngineConfig", "download_models"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
