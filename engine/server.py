"""SōmaGraph 解析サーバー。

動画アップロード → バックグラウンドで MMPose エンジン実行 → 進捗/結果API。
ダッシュボード (index.html / mobile.html) も同一オリジンで配信する。

起動:
  cd engine
  .venv/bin/python server.py                 # http://localhost:8760
  SOMAGRAPH_DEVICE=cuda:0 .venv/bin/python server.py   # GPU
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ENGINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ENGINE_DIR.parent
JOBS_DIR = ENGINE_DIR / "jobs"

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = FastAPI(title="SomaGraph API")

# 静的ホスティング(Vercel等)上のダッシュボードからも叩けるようにする
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from somagraph import EngineConfig, SomaGraphEngine
        device = os.environ.get("SOMAGRAPH_DEVICE", "cpu")
        _engine = SomaGraphEngine(EngineConfig(device=device))
    return _engine


def _worker():
    """1本ずつ順番に解析する常駐ワーカー (モデルは同時実行不可)。"""
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        try:
            job["status"] = "loading_model"
            engine = _get_engine()
            job["status"] = "running"

            def cb(done: int, total: int) -> None:
                job["progress"] = {"done": done, "total": total}

            out_dir = Path(job["dir"])
            dashboard = engine.analyze_video(
                job["input"], out_dir, render=True, progress=False, progress_cb=cb)

            raw = out_dir / "annotated.mp4"
            web = out_dir / "annotated_h264.mp4"
            if shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(web)],
                    check=True)
            else:  # ffmpegが無ければそのまま (ブラウザ再生は保証されない)
                shutil.copy(raw, web)

            job["dashboard"] = dashboard
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 — ジョブ失敗はAPIで返す
            job["status"] = "error"
            job["error"] = str(e)


threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "engine": "somagraph-mmpose"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"未対応の形式です: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"

    size = 0
    with input_path.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "ファイルが大きすぎます (上限500MB)")
            f.write(chunk)

    _jobs[job_id] = {
        "status": "queued",
        "input": str(input_path),
        "dir": str(job_dir),
        "filename": file.filename,
        "progress": {"done": 0, "total": 0},
    }
    _queue.put(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "status": job["status"],
        "filename": job.get("filename"),
        "progress": job.get("progress"),
        "dashboard": job.get("dashboard"),
        "error": job.get("error"),
    }


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "video not ready")
    return FileResponse(Path(job["dir"]) / "annotated_h264.mp4", media_type="video/mp4")


@app.get("/api/jobs/{job_id}/keypoints.csv")
def job_keypoints(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(Path(job["dir"]) / "keypoints.csv", media_type="text/csv")


# ---- ダッシュボード配信 ----

_MOBILE_UA = ("android", "iphone", "ipod", "windows phone",
              "blackberry", "iemobile", "opera mini", "mobile")


@app.get("/")
def index(request: Request):
    """アクセス元デバイスでPC/スマホビューを自動切替 (URLは / のまま)。"""
    ua = request.headers.get("user-agent", "").lower()
    page = "mobile.html" if any(k in ua for k in _MOBILE_UA) else "index.html"
    return FileResponse(REPO_ROOT / page, media_type="text/html")


@app.get("/index.html")
def desktop():
    return FileResponse(REPO_ROOT / "index.html", media_type="text/html")


@app.get("/mobile.html")
def mobile():
    return FileResponse(REPO_ROOT / "mobile.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8760)))
