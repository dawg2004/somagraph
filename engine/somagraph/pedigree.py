"""血統(ペディグリー)分析。

**データ源について:** studbook.jp (JAIRS) や JBIS-Search は利用規約で
「データベースの解析」「私的利用/軽種馬生産・育成牧場の内部利用を超える
複製」を禁止しており、自動取得・自動解析には使えない
(詳細は engine/README.md #血統評価 を参照)。
本モジュールは、ユーザーが**手動で**確認・入力した血統情報 (CSV) を
入力として受け取る前提で設計している。自動収集ロジックは含まない。

CSV形式: `馬名,父,母,生年` のフラットな系図テーブル (1行=1頭)。
任意の馬について、このテーブルを再帰的に辿って血統樹を組み立てる。
祖先がテーブルに無ければその枝で打ち切る (エラーにはしない)。

計算する指標:
  - 近親係数 (Wright's coefficient of inbreeding) の近似値
  - 世代内の重複祖先 (ニックス/クロス表記の元になる情報)
  - 判明している血統の深さ (何代まで埋まっているか)

スコアリング: JSON定義の線形モデル (特徴量名 -> 重み) + シグモイド。
`models/pedigree_weights.json` か環境変数 SOMAGRAPH_PEDIGREE_WEIGHTS。
重い依存(sklearn等)を避け、透明で調整しやすい形にしている。
モデルが無ければ pedigree_score は null (構造分析は null にならない)。
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "pedigree_weights.json"
DEFAULT_MAX_GEN = 5


def load_pedigree_table(csv_path: str | Path) -> dict[str, dict]:
    """`馬名,父,母,生年` のCSVを {馬名: {sire, dam, birth_year}} に読み込む。

    列名は日本語(馬名/父/母/生年)・英語(horse/sire/dam/birth_year)どちらも許容。
    ファイルが無ければ空の辞書を返す (エラーにしない)。
    """
    path = Path(csv_path)
    table: dict[str, dict] = {}
    if not path.exists():
        return table
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("馬名") or row.get("horse") or "").strip()
            if not name:
                continue
            sire = (row.get("父") or row.get("sire") or "").strip() or None
            dam = (row.get("母") or row.get("dam") or "").strip() or None
            year = (row.get("生年") or row.get("birth_year") or "").strip() or None
            table[name] = {"sire": sire, "dam": dam, "birth_year": year}
    return table


def upsert_pedigree_rows(csv_path: str | Path, rows: list[dict]) -> int:
    """既存CSVに行を追記/上書き(同名なら更新)する。書き込んだ行数を返す。"""
    path = Path(csv_path)
    table = load_pedigree_table(path)
    for r in rows:
        name = (r.get("horse") or r.get("馬名") or "").strip()
        if not name:
            continue
        table[name] = {
            "sire": (r.get("sire") or r.get("父") or "").strip() or None,
            "dam": (r.get("dam") or r.get("母") or "").strip() or None,
            "birth_year": (r.get("birth_year") or r.get("生年") or "").strip() or None,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["horse", "sire", "dam", "birth_year"])
        for name, rec in table.items():
            w.writerow([name, rec.get("sire") or "", rec.get("dam") or "", rec.get("birth_year") or ""])
    return len(table)


def _ancestors_by_gen(name: str, table: dict[str, dict],
                      max_gen: int = DEFAULT_MAX_GEN) -> list[list[str | None]]:
    """世代ごとの祖先名リスト。gen[0]=[父,母], gen[1]=[父父,父母,母父,母母], ..."""
    gens: list[list[str | None]] = []
    frontier = [name]
    for _ in range(max_gen):
        nxt: list[str | None] = []
        for n in frontier:
            rec = table.get(n) if n else None
            nxt.append(rec.get("sire") if rec else None)
            nxt.append(rec.get("dam") if rec else None)
        gens.append(nxt)
        frontier = nxt
    return gens


def known_generations(name: str, table: dict[str, dict],
                      max_gen: int = DEFAULT_MAX_GEN) -> int:
    """何世代まで祖先情報が(部分的にでも)埋まっているか。"""
    if name not in table:
        return 0
    gens = _ancestors_by_gen(name, table, max_gen)
    depth = 0
    for g in gens:
        if not any(g):
            break
        depth += 1
    return depth


def inbreeding_coefficient(name: str, table: dict[str, dict],
                           max_gen: int = DEFAULT_MAX_GEN) -> tuple[float, list[dict]]:
    """Wright's coefficient of inbreeding の近似値と重複祖先の内訳を返す。

    F ≈ Σ (0.5)^(n1+n2+1) — 共通祖先ごとに、父方・母方それぞれの経路長 n1,n2 で。
    祖先自身の近親係数(1+F_A)項は簡略化のため省略した近似値。
    """
    if name not in table:
        return 0.0, []

    # 祖先名 -> [その祖先に到達するまでの世代数のリスト] を父方/母方別に集計
    def paths_from(root: str | None, depth_limit: int) -> dict[str, list[int]]:
        found: dict[str, list[int]] = {}
        if root is None:
            return found
        stack = [(root, 1)]
        while stack:
            n, depth = stack.pop()
            found.setdefault(n, []).append(depth)
            if depth >= depth_limit:
                continue
            rec = table.get(n)
            if not rec:
                continue
            if rec.get("sire"):
                stack.append((rec["sire"], depth + 1))
            if rec.get("dam"):
                stack.append((rec["dam"], depth + 1))
        return found

    root = table[name]
    sire_paths = paths_from(root.get("sire"), max_gen)
    dam_paths = paths_from(root.get("dam"), max_gen)

    common = set(sire_paths) & set(dam_paths)
    total = 0.0
    details = []
    for anc in common:
        contrib = 0.0
        for n1 in sire_paths[anc]:
            for n2 in dam_paths[anc]:
                contrib += 0.5 ** (n1 + n2 + 1)
        total += contrib
        details.append({
            "ancestor": anc,
            "sire_side_gens": sorted(sire_paths[anc]),
            "dam_side_gens": sorted(dam_paths[anc]),
            "contribution": round(contrib, 5),
        })
    details.sort(key=lambda d: -d["contribution"])
    return round(total, 5), details


def pedigree_summary(name: str, table: dict[str, dict],
                     max_gen: int = DEFAULT_MAX_GEN) -> dict | None:
    """馬名から血統サマリーを組み立てる。テーブルに無ければNone。"""
    if name not in table:
        return None
    rec = table[name]
    coef, dup = inbreeding_coefficient(name, table, max_gen)
    return {
        "horse": name,
        "sire": rec.get("sire"),
        "dam": rec.get("dam"),
        "damsire": table.get(rec.get("dam") or "", {}).get("sire"),
        "generations_known": known_generations(name, table, max_gen),
        "inbreeding_coefficient": coef,
        "notable_duplicate_ancestors": dup[:5],
    }


def resolve_weights_path() -> Path | None:
    p = os.environ.get("SOMAGRAPH_PEDIGREE_WEIGHTS")
    path = Path(p) if p else DEFAULT_WEIGHTS_PATH
    return path if path.exists() else None


def score_pedigree(summary: dict, weights_path: Path | None = None) -> float | None:
    """血統サマリーを 0..1 のスコアにする。重み定義が無ければNone。

    重みファイル形式 (JSON):
      {"bias": 0.0, "weights": {"generations_known": 0.1, "inbreeding_coefficient": -2.0}}
    未知の特徴量キーは無視。summary の数値項目のみ特徴量として使う。
    """
    path = weights_path or resolve_weights_path()
    if path is None:
        return None
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    bias = float(cfg.get("bias", 0.0))
    weights = cfg.get("weights", {})
    z = bias
    for key, w in weights.items():
        val = summary.get(key)
        if isinstance(val, (int, float)):
            z += float(w) * float(val)
    return round(1.0 / (1.0 + math.exp(-z)), 4)
