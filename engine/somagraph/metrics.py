"""キーポイント時系列から歩様メトリクスを計算する。

入力はパイプラインが出力する
  keypoints: (T, 20, 2) float — フレームごとの関節座標(px)
  scores:    (T, 20)    float — 信頼度 0..1
のみで、モデルには依存しない (numpyだけで動く)。

計算するもの:
  - 左右差 前肢/後肢: 左右の蹄の上下動振幅の非対称率 (%)
  - 接地差: 左右の蹄の接地時間率の差 (%)
  - リズム安定性: 蹄の上下動の自己相関ピーク (0-100)
  - ストライド周期 (秒)
  - 歩様総合スコア (0-100, ヒューリスティック)
"""
from __future__ import annotations

import numpy as np

from .skeleton import KP

DEFAULT_KPT_THR = 0.3


def clean_series(series: np.ndarray, conf: np.ndarray,
                 kpt_thr: float = DEFAULT_KPT_THR) -> np.ndarray | None:
    """低信頼度フレームをNaN化して線形補間。有効点が2割未満ならNone。"""
    s = series.astype(float).copy()
    s[conf < kpt_thr] = np.nan
    ok = ~np.isnan(s)
    if ok.sum() < max(4, int(0.2 * len(s))):
        return None
    idx = np.arange(len(s))
    return np.interp(idx, idx[ok], s[ok])


def smooth(s: np.ndarray, win: int = 5) -> np.ndarray:
    """移動平均。端はエッジ複製。"""
    if win <= 1 or len(s) < win:
        return s
    pad = win // 2
    padded = np.pad(s, pad, mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def oscillation_amplitude(y: np.ndarray) -> float:
    """外れ値に頑健な振幅 (5-95パーセンタイル幅)。"""
    return float(np.percentile(y, 95) - np.percentile(y, 5))


def lateral_asymmetry_pct(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """左右の上下動振幅の非対称率 (%). 0=完全対称。"""
    a_l = oscillation_amplitude(y_left)
    a_r = oscillation_amplitude(y_right)
    denom = max(a_l, a_r)
    if denom < 1e-6:
        return 0.0
    return abs(a_l - a_r) / denom * 100.0


def stance_fraction(y: np.ndarray, contact_band: float = 0.15) -> float:
    """接地時間率: 蹄が自分の最下端付近(可動域の下15%)にいるフレーム割合。

    画像座標系では下ほどyが大きい。
    """
    lo, hi = np.percentile(y, 5), np.percentile(y, 95)
    rng = hi - lo
    if rng < 1e-6:
        return 1.0
    return float(np.mean(y > hi - contact_band * rng))


def ground_contact_asymmetry_pct(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """左右の接地時間率の差 (パーセントポイント)。"""
    return abs(stance_fraction(y_left) - stance_fraction(y_right)) * 100.0


def stride_cycle(y: np.ndarray, fps: float,
                 min_period_s: float = 0.3,
                 max_period_s: float = 2.5) -> tuple[float | None, float]:
    """自己相関からストライド周期(秒)とピーク強度(0..1)を返す。

    周期が見つからなければ (None, 0.0)。
    """
    y0 = y - y.mean()
    n = len(y0)
    if n < int(min_period_s * fps) * 3 or np.allclose(y0, 0):
        return None, 0.0
    ac = np.correlate(y0, y0, mode="full")[n - 1:]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    lag_min = max(2, int(min_period_s * fps))
    lag_max = min(n - 2, int(max_period_s * fps))
    if lag_min >= lag_max:
        return None, 0.0
    seg = ac[lag_min:lag_max]
    # 局所最大のうち最大値を採用
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(peaks, key=lambda i: seg[i])
    peak_val = float(seg[best])
    if peak_val <= 0.1:
        return None, 0.0
    period_s = (best + lag_min) / fps
    return period_s, peak_val


def rhythm_stability_score(peak_val: float) -> float:
    """自己相関ピーク(0..1)を0-100スコアへ。"""
    return float(np.clip(peak_val, 0.0, 1.0) * 100.0)


def compute_gait_metrics(keypoints: np.ndarray, scores: np.ndarray, fps: float,
                         kpt_thr: float = DEFAULT_KPT_THR) -> dict:
    """パイプライン出力からダッシュボード用の歩様メトリクス一式を計算する。"""
    T = keypoints.shape[0]
    result: dict = {
        "frames": int(T),
        "fps": float(fps),
        "fore_asymmetry_pct": None,
        "hind_asymmetry_pct": None,
        "contact_asymmetry_pct": None,
        "rhythm_stability": None,
        "stride_period_s": None,
        "gait_score": None,
    }
    if T < 8:
        return result

    def paw_y(name: str) -> np.ndarray | None:
        i = KP[name]
        s = clean_series(keypoints[:, i, 1], scores[:, i], kpt_thr)
        return smooth(s) if s is not None else None

    lf, rf = paw_y("L_F_Paw"), paw_y("R_F_Paw")
    lb, rb = paw_y("L_B_Paw"), paw_y("R_B_Paw")

    if lf is not None and rf is not None:
        result["fore_asymmetry_pct"] = round(lateral_asymmetry_pct(lf, rf), 1)
        result["contact_asymmetry_pct"] = round(ground_contact_asymmetry_pct(lf, rf), 1)
    if lb is not None and rb is not None:
        result["hind_asymmetry_pct"] = round(lateral_asymmetry_pct(lb, rb), 1)

    # リズム: 有効な蹄系列のうち自己相関ピークが最大のもの
    best_period, best_peak = None, 0.0
    for y in (lf, rf, lb, rb):
        if y is None:
            continue
        period, peak = stride_cycle(y, fps)
        if period is not None and peak > best_peak:
            best_period, best_peak = period, peak
    if best_period is not None:
        result["stride_period_s"] = round(best_period, 3)
        result["rhythm_stability"] = round(rhythm_stability_score(best_peak), 1)

    result["gait_score"] = _gait_score(result)
    return result


def _gait_score(m: dict) -> float | None:
    """歩様総合スコア (0-100)。非対称・接地差・リズム不安定を減点するヒューリスティック。"""
    fore = m["fore_asymmetry_pct"]
    hind = m["hind_asymmetry_pct"]
    contact = m["contact_asymmetry_pct"]
    rhythm = m["rhythm_stability"]
    if fore is None and hind is None:
        return None
    score = 100.0
    if fore is not None:
        score -= 1.2 * fore
    if hind is not None:
        score -= 1.0 * hind
    if contact is not None:
        score -= 0.8 * contact
    if rhythm is not None:
        score -= 0.3 * (100.0 - rhythm)
    return round(float(np.clip(score, 0.0, 100.0)), 1)
