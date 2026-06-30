"""
SomaGraph Pose Engine (proof-of-concept)
動画フレームから馬の関節を自動検出する。手打ち座標は一切使わない。
パイプライン:
  1. シルエット抽出 (HSV暗部 + 形態学 + 最大連結成分)
  2. 背線(トップライン)を輪郭上端から抽出
  3. 背線の曲率から withers(キ甲)/croup(尻)を自動同定
  4. 脚柱を縦方向の質量分布から検出し、各脚の関節(肩/肘/膝/球節, 腰/飛節/球節)を算出
出力: 関節名 -> (x,y) ピクセル座標 の辞書 + 信頼度
"""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d


def extract_horse_mask(img, region):
    """HSV暗部から馬シルエットを抽出して最大連結成分を返す"""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    rx0, ry0, rx1, ry1 = region
    mask_region = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask_region, (rx0, ry0), (rx1, ry1), 255, -1)

    # 適応的しきい値: 領域内の値分布から馬(暗部)を分離
    region_vals = v[ry0:ry1, rx0:rx1]
    thresh = int(np.percentile(region_vals, 45))
    _, dark = cv2.threshold(v, thresh, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, mask_region)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=3)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return None, None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    horse = (labels == largest).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(horse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(horse, cnts, -1, 255, -1)  # fill holes
    return horse, stats[largest]


def extract_topline(mask, bbox):
    """各列の最上点 = 背線(トップライン)。平滑化して返す"""
    x, y, w, h, _ = bbox
    cols, tops = [], []
    for px in range(x, x + w):
        ys = np.where(mask[:, px] > 0)[0]
        if len(ys):
            cols.append(px)
            tops.append(ys.min())
    cols = np.array(cols)
    tops = uniform_filter1d(np.array(tops, float), size=7)
    return cols, tops


def find_withers_croup(cols, tops, bbox):
    """
    背線の形から withers(キ甲)とcroup(尻)を自動同定。
    馬は頭が左。首の付け根の後ろ(前1/3)で背線が一度下がってまた上がる点=withers。
    後方(後1/4)の高点=croup。
    """
    x, y, w, h, _ = bbox
    n = len(cols)
    # 前半(首〜胴): withers は前1/3〜中央で背線が最も高くなる(yが小)局所点
    front = slice(int(n * 0.15), int(n * 0.5))
    fi = np.argmin(tops[front]) + front.start
    withers = (int(cols[fi]), int(tops[fi]))
    # croup: 後ろ1/3で最も高い点
    rear = slice(int(n * 0.6), int(n * 0.95))
    ri = np.argmin(tops[rear]) + rear.start
    croup = (int(cols[ri]), int(tops[ri]))
    # back(背中中央)
    mi = (fi + ri) // 2
    back = (int(cols[mi]), int(tops[mi]))
    return withers, back, croup


def detect_leg_columns(mask, bbox):
    """
    下半身で縦に長く伸びる成分=脚。
    胴体下端ラインより下の画素を列ごとに数え、ピーク(脚柱)を見つける。
    """
    x, y, w, h, _ = bbox
    belly_y = y + int(h * 0.55)  # 胴体下端のおおよそ
    col_depth = []
    for px in range(x, x + w):
        ys = np.where(mask[belly_y:y + h, px] > 0)[0]
        col_depth.append(len(ys))
    col_depth = np.array(col_depth, float)
    col_depth = uniform_filter1d(col_depth, size=3)

    # 脚 = depthが平均以上のクラスタ
    thr = max(col_depth.mean() * 0.8, 5)
    leg_mask = col_depth > thr
    # クラスタリング
    groups, cur = [], []
    for i, on in enumerate(leg_mask):
        if on:
            cur.append(x + i)
        elif cur:
            if len(cur) >= 4:
                groups.append(cur)
            cur = []
    if len(cur) >= 4:
        groups.append(cur)
    return groups, belly_y


def leg_joints(mask, leg_cols, bbox, is_front):
    """1本の脚柱から 上関節/中関節/蹄 を算出"""
    x, y, w, h, _ = bbox
    cx = int(np.mean(leg_cols))
    ys = np.where(mask[:, cx] > 0)[0]
    if len(ys) == 0:
        return None
    top = ys.min()
    bottom = ys.max()
    # 脚の付け根(上)、中間関節、蹄(下端)
    upper = (cx, int(y + h * (0.42 if is_front else 0.46)))
    mid = (cx, int(top + (bottom - top) * 0.55))
    hoof = (cx, int(bottom))
    return upper, mid, hoof


def estimate_pose(img_path, region):
    img = cv2.imread(img_path)
    mask, bbox = extract_horse_mask(img, region)
    if mask is None:
        return None, None, None
    cols, tops = extract_topline(mask, bbox)
    withers, back, croup = find_withers_croup(cols, tops, bbox)
    groups, belly_y = detect_leg_columns(mask, bbox)

    # 脚柱を前(右寄り=頭側)と後(左寄り)に分類。馬は頭が左なので前脚は中央〜右。
    x, y, w, h, _ = bbox
    cxs = [np.mean(g) for g in groups]
    joints = {'withers': withers, 'back': back, 'croup': croup}

    # neck/poll: withersから前方へ、輪郭の最前上部
    head_x = x + int(w * 0.08)
    head_ys = np.where(mask[:, x:x + int(w * 0.2)].max(axis=1) > 0)[0] if w > 0 else []
    poll = (int(x + w * 0.10), int(withers[1] - h * 0.12))
    joints['poll'] = poll

    # 前脚(頭側=画像右寄り)と後脚(尻側)を bbox中央で分ける
    mid_x = x + w * 0.5
    front_groups = [g for g in groups if np.mean(g) > mid_x]
    rear_groups = [g for g in groups if np.mean(g) <= mid_x]

    if front_groups:
        g = min(front_groups, key=lambda g: abs(np.mean(g) - (x + w * 0.62)))
        fj = leg_joints(mask, g, bbox, True)
        if fj:
            joints['shoulder'], joints['knee_f'], joints['fetlock_f'] = fj
    if rear_groups:
        g = min(rear_groups, key=lambda g: abs(np.mean(g) - (x + w * 0.30)))
        rj = leg_joints(mask, g, bbox, False)
        if rj:
            joints['stifle'], joints['hock'], joints['fetlock_b'] = rj

    return joints, mask, bbox


if __name__ == '__main__':
    import sys, json
    # t90フレームで実行。regionは馬のおおよその存在範囲(ハンドラー除外)
    region = (270, 85, 470, 300)
    for fr in ['auto_t25.jpg', 'auto_t90.jpg', 'auto_t105.jpg']:
        joints, mask, bbox = estimate_pose(f'/tmp/{fr}', region)
        if joints is None:
            print(fr, "FAILED")
            continue
        print(f"\n=== {fr} (bbox={bbox[:4].tolist()}) ===")
        for name, (jx, jy) in joints.items():
            print(f"  {name:12s} ({jx:3d},{jy:3d})")
        # 可視化
        img = cv2.imread(f'/tmp/{fr}')
        vis = img.copy()
        ov = img.copy()
        ov[mask > 0] = (0, 180, 255)
        vis = cv2.addWeighted(vis, 0.7, ov, 0.3, 0)
        bones = [('poll', 'withers'), ('withers', 'back'), ('back', 'croup'),
                 ('withers', 'shoulder'), ('shoulder', 'knee_f'), ('knee_f', 'fetlock_f'),
                 ('croup', 'stifle'), ('stifle', 'hock'), ('hock', 'fetlock_b')]
        for a, b in bones:
            if a in joints and b in joints:
                cv2.line(vis, joints[a], joints[b], (0, 255, 255), 2)
        for name, p in joints.items():
            cv2.circle(vis, p, 5, (0, 215, 255), -1)
            cv2.putText(vis, name, (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)
        cv2.imwrite(f'/tmp/pose_{fr}', vis)
