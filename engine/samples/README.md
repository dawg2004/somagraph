# サンプル動画

## horse_walk.mp4

- 出典: [Wikimedia Commons — Horse walking in corral MVI 7490.MOV.ogv](https://commons.wikimedia.org/wiki/File:Horse_walking_in_corral_MVI_7490.MOV.ogv)
- ライセンス: CC BY-SA 2.0 fr
- 640x480 / 30fps / 287フレーム (約9.6秒)。Theora(ogv) から H.264 に変換済み。

実行例:

```bash
cd engine
.venv/bin/python -m somagraph samples/horse_walk.mp4 -o samples/results --device cpu
```

出力 (`samples/results/` — git管理外):
`annotated.mp4` / `keypoints.csv` / `dashboard.json`
