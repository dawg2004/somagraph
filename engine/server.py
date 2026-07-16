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
from pydantic import BaseModel

ENGINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ENGINE_DIR.parent
JOBS_DIR = ENGINE_DIR / "jobs"

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# URL解析時に切り出す最大秒数。既定30秒 (CPUでは約1秒/フレームかかるため)。
# GPU環境なら SOMAGRAPH_URL_MAX_SEC=7200 等に伸ばして全編解析も可能。
URL_MAX_SEC = float(os.environ.get("SOMAGRAPH_URL_MAX_SEC", "30"))
# ダウンロードを拒否する元動画の長さ上限 (既定2時間)
URL_SOURCE_MAX_SEC = float(os.environ.get("SOMAGRAPH_SOURCE_MAX_SEC", str(2 * 60 * 60)))
# URLダウンロードのファイルサイズ上限 (2時間720pを想定して4GB)
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024

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


def _download_video(url: str, dest_dir: Path,
                    start_s: float | None, duration_s: float | None) -> Path:
    """yt-dlpでURLから動画を取得し、解析用に H.264 mp4 へ切り出す。

    ffmpegがあれば必要区間だけダウンロードするため、長尺動画でも
    取得は解析窓ぶんのサイズで済む。
    """
    import yt_dlp

    start = max(float(start_s or 0), 0.0)
    dur = min(float(duration_s) if duration_s else URL_MAX_SEC, URL_MAX_SEC)
    has_ffmpeg = shutil.which("ffmpeg") is not None

    base_opts = {
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "format": "mp4[height<=720]/best[height<=720]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": MAX_DOWNLOAD_BYTES,
    }

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    src_dur = info.get("duration")
    if src_dur and src_dur > URL_SOURCE_MAX_SEC:
        raise ValueError(
            f"動画が長すぎます ({src_dur / 60:.0f}分)。"
            f"{URL_SOURCE_MAX_SEC / 60:.0f}分以内にしてください"
            " (SOMAGRAPH_SOURCE_MAX_SEC で変更可)")

    # 必要区間 (+前後バッファ) のみ取得。直リンク等のgeneric extractorでは
    # ffmpegダウンローダが失敗するため全編取得にフォールバックする。
    use_range = has_ffmpeg and info.get("extractor_key", "").lower() != "generic"
    dl_opts = dict(base_opts)
    if use_range:
        dl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(start, start + dur + 2)])
        dl_opts["force_keyframes_at_cuts"] = True

    def _do_download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            i = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(i))

    range_applied = False
    try:
        src = _do_download(dl_opts)
        range_applied = use_range
    except yt_dlp.utils.DownloadError:
        if not use_range:
            raise
        src = _do_download(base_opts)  # 区間取得が効かないサイト向けフォールバック
    if not src.exists():
        raise FileNotFoundError("ダウンロードに失敗しました")

    out = dest_dir / "input.mp4"
    if has_ffmpeg:
        # 区間ダウンロード済みなら先頭から、全編落ちてきていれば start から切り出す。
        # 区間指定が黙って無視されるサイト対策として実測の長さでも裏取りする。
        src_len = _probe_duration(src)
        seek = start
        if range_applied and (src_len is None or src_len <= dur + 6):
            seek = 0.0
        if seek > 0 and src_len is not None and seek >= src_len:
            raise ValueError(
                f"開始位置({seek:.0f}秒)が動画の長さ({src_len:.0f}秒)を超えています")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(seek), "-i", str(src),
             "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
            check=True)
        src.unlink(missing_ok=True)
    else:
        if src.suffix.lower() != ".mp4":
            raise RuntimeError("ffmpegが無い環境ではmp4のURLのみ対応です")
        src.rename(out)
    return out


def _probe_duration(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _worker():
    """1本ずつ順番に解析する常駐ワーカー (モデルは同時実行不可)。"""
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        try:
            if job.get("source_url"):
                job["status"] = "downloading"
                job["input"] = str(_download_video(
                    job["source_url"], Path(job["dir"]),
                    job.get("start_s"), job.get("duration_s")))
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


class AnalyzeUrlRequest(BaseModel):
    url: str
    start_s: float | None = None
    duration_s: float | None = None


@app.post("/api/analyze_url")
def analyze_url(req: AnalyzeUrlRequest):
    """YouTube等のURLから動画を取得して解析ジョブに投入する。"""
    if not req.url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "http/https のURLを指定してください")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _jobs[job_id] = {
        "status": "queued",
        "source_url": req.url,
        "start_s": req.start_s,
        "duration_s": req.duration_s,
        "dir": str(job_dir),
        "filename": req.url,
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


@app.get("/api/jobs/{job_id}/standing.jpg")
def job_standing(job_id: str):
    """立ち姿として抽出したフレーム (馬体評価用クロップ)。"""
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    path = Path(job["dir"]) / "standing_crop.jpg"
    if not path.exists():
        raise HTTPException(404, "standing frame not found")
    return FileResponse(path, media_type="image/jpeg")


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
