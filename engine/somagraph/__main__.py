"""CLI: python -m somagraph <video> [-o outdir] [--device cpu] ..."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="somagraph",
        description="馬の歩様動画を解析して骨格オーバーレイ動画とメトリクスを出力する")
    p.add_argument("video", nargs="?", help="入力動画 (mp4等)")
    p.add_argument("-o", "--out", default="results", help="出力ディレクトリ (default: results)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu (default: cuda:0)")
    p.add_argument("--det-thr", type=float, default=0.5, help="馬検出の信頼度しきい値")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="キーポイント信頼度しきい値")
    p.add_argument("--no-video", action="store_true", help="オーバーレイ動画を出力しない")
    p.add_argument("--download-models", action="store_true",
                   help="モデル(config+checkpoint)を取得して終了")
    args = p.parse_args(argv)

    from .pipeline import EngineConfig, SomaGraphEngine, download_models

    if args.download_models:
        download_models()
        print("models downloaded")
        return 0

    if args.video is None:
        p.error("入力動画を指定してください (モデル取得のみなら --download-models)")

    cfg = EngineConfig(device=args.device,
                       det_score_thr=args.det_thr,
                       kpt_score_thr=args.kpt_thr)
    engine = SomaGraphEngine(cfg)
    dashboard = engine.analyze_video(args.video, args.out,
                                     render=not args.no_video)
    print(json.dumps(dashboard, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
