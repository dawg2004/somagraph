"""AnimalPose (20キーポイント) の骨格定義。

MMPose の `animalpose` データセット規約に準拠する。
HRNet-W32 AnimalPose モデル (td-hm_hrnet-w32_8xb64-210e_animalpose-256x256)
の出力インデックスと一対一で対応。
"""

KEYPOINT_NAMES = [
    "L_Eye",      # 0
    "R_Eye",      # 1
    "L_EarBase",  # 2
    "R_EarBase",  # 3
    "Nose",       # 4
    "Throat",     # 5
    "TailBase",   # 6
    "Withers",    # 7
    "L_F_Elbow",  # 8
    "R_F_Elbow",  # 9
    "L_B_Elbow",  # 10 (後肢の付け根/膝上)
    "R_B_Elbow",  # 11
    "L_F_Knee",   # 12
    "R_F_Knee",   # 13
    "L_B_Knee",   # 14 (飛節)
    "R_B_Knee",   # 15
    "L_F_Paw",    # 16 (前蹄)
    "R_F_Paw",    # 17
    "L_B_Paw",    # 18 (後蹄)
    "R_B_Paw",    # 19
]

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# 部位グループ: 描画色とメトリクスのグルーピングに使う
GROUP_HEAD_SPINE = "head_spine"
GROUP_FORE = "foreleg"
GROUP_HIND = "hindleg"

# (始点, 終点, グループ)
BONES = [
    # 頭部
    (KP["L_Eye"], KP["Nose"], GROUP_HEAD_SPINE),
    (KP["R_Eye"], KP["Nose"], GROUP_HEAD_SPINE),
    (KP["L_EarBase"], KP["L_Eye"], GROUP_HEAD_SPINE),
    (KP["R_EarBase"], KP["R_Eye"], GROUP_HEAD_SPINE),
    (KP["Nose"], KP["Throat"], GROUP_HEAD_SPINE),
    # 体幹
    (KP["Throat"], KP["Withers"], GROUP_HEAD_SPINE),
    (KP["Withers"], KP["TailBase"], GROUP_HEAD_SPINE),
    # 前肢 (左右)
    (KP["Throat"], KP["L_F_Elbow"], GROUP_FORE),
    (KP["L_F_Elbow"], KP["L_F_Knee"], GROUP_FORE),
    (KP["L_F_Knee"], KP["L_F_Paw"], GROUP_FORE),
    (KP["Throat"], KP["R_F_Elbow"], GROUP_FORE),
    (KP["R_F_Elbow"], KP["R_F_Knee"], GROUP_FORE),
    (KP["R_F_Knee"], KP["R_F_Paw"], GROUP_FORE),
    # 後肢 (左右)
    (KP["TailBase"], KP["L_B_Elbow"], GROUP_HIND),
    (KP["L_B_Elbow"], KP["L_B_Knee"], GROUP_HIND),
    (KP["L_B_Knee"], KP["L_B_Paw"], GROUP_HIND),
    (KP["TailBase"], KP["R_B_Elbow"], GROUP_HIND),
    (KP["R_B_Elbow"], KP["R_B_Knee"], GROUP_HIND),
    (KP["R_B_Knee"], KP["R_B_Paw"], GROUP_HIND),
]

# ダッシュボード(index.html)と同じ配色。OpenCV用にBGR。
GROUP_COLORS_BGR = {
    GROUP_HEAD_SPINE: (245, 143, 74),   # #4a8ff5 blue
    GROUP_FORE: (180, 204, 45),         # #2DCCB4 teal
    GROUP_HIND: (76, 168, 201),         # #C9A84C gold
}

# COCO 80クラス(0始まり)の horse
COCO_HORSE_CLASS_ID = 17
