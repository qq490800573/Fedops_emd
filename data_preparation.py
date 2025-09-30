# data_preparation.py
import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
# ====== HTTP 下载 & 解压 & 目录管理（无第三方依赖） ======
import urllib.request, shutil, hashlib, zipfile, tarfile
from pathlib import Path

# 统一的数据根目录（优先环境变量；兼容容器/本地）
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./dataset")).resolve()

# 可选：HTTP 下载地址（例如一个 zip）
DATASET_URL = os.environ.get("DATASET_URL", "")  # 例：https://example.com/fitbit_export.zip
DATASET_MD5 = os.environ.get("DATASET_MD5", "")  # 可选：MD5 校验
# 解压后用于读取 CSV 的二级目录（可根据你的压缩包结构调整）
# 默认自动匹配：archive/mturkfitbit_export_*/Fitabase Data */
FITBIT_EXPORT_GLOB = os.environ.get("FITBIT_EXPORT_GLOB", "archive/mturkfitbit_export_*")
FITBIT_SUBDIR_GLOB = os.environ.get("FITBIT_SUBDIR_GLOB", "Fitabase Data */")

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_progress(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    def _report(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1.0 / totalsize * 100
            print(f"\rDownloading {url}  {percent:5.1f}% ({readsofar/1e6:.1f}MB/{totalsize/1e6:.1f}MB)", end="")
        else:
            print(f"\rDownloading {url}  {readsofar/1e6:.1f}MB", end="")
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp.as_posix(), _report)
        print()
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

def _extract_archive(archive_path: Path, to_dir: Path):
    to_dir.mkdir(parents=True, exist_ok=True)
    ap = archive_path.as_posix()
    if zipfile.is_zipfile(ap):
        with zipfile.ZipFile(ap, "r") as zf:
            zf.extractall(to_dir)
    elif ap.endswith(".tar.gz") or ap.endswith(".tgz"):
        with tarfile.open(ap, "r:gz") as tf:
            tf.extractall(to_dir)
    elif ap.endswith(".tar"):
        with tarfile.open(ap, "r:") as tf:
            tf.extractall(to_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def _auto_locate_fitbit_base(root: Path) -> Path:
    """
    自动匹配：archive/mturkfitbit_export_*/Fitabase Data */
    返回实际的 CSV 根目录（即包含 sleepDay_merged.csv 等文件的目录）
    """
    export_dirs = sorted((root / "archive").glob(Path(FITBIT_EXPORT_GLOB).name))
    if not export_dirs:
        raise FileNotFoundError(f"没有找到 export 目录：{root}/archive/{FITBIT_EXPORT_GLOB}")
    # 取最新的一个
    export_dir = export_dirs[-1]
    fitabase_dirs = list(export_dir.glob(FITBIT_SUBDIR_GLOB))
    if not fitabase_dirs:
        raise FileNotFoundError(f"没有找到 Fitabase 子目录：{export_dir}/{FITBIT_SUBDIR_GLOB}")
    return fitabase_dirs[0].resolve()

def _ensure_fitbit_data() -> Path:
    """
    确保 DATASET_DIR 下已经存在可读取的 Fitbit CSV 目录。
    - 若不存在且提供了 DATASET_URL：下载压缩包到 DATASET_DIR/archive_downloads 并解压到 DATASET_DIR
    - 返回可用的 FITBIT_BASE_DIR（包含 *_merged.csv 的目录）
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 如果已经有匹配的目录，直接返回
    try:
        base = _auto_locate_fitbit_base(DATASET_DIR)
        print(f"[DATA] Using existing Fitbit base: {base}")
        return base
    except FileNotFoundError:
        pass

    if not DATASET_URL:
        raise FileNotFoundError(
            f"数据未就绪：在 {DATASET_DIR} 下未找到 'archive/mturkfitbit_export_*'；"
            f"且未设置 DATASET_URL 以供自动下载。"
        )

    # 下载
    downloads = DATASET_DIR / "archive_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    fname = DATASET_URL.split("?")[0].rstrip("/").split("/")[-1] or "dataset.zip"
    archive_path = downloads / fname

    if not archive_path.exists():
        print(f"[DATA] Downloading dataset from: {DATASET_URL}")
        _download_with_progress(DATASET_URL, archive_path)
    else:
        print(f"[DATA] Found existing archive: {archive_path}")

    # 可选校验
    if DATASET_MD5:
        md5_now = _md5(archive_path)
        if md5_now.lower() != DATASET_MD5.lower():
            raise ValueError(f"MD5 不一致：{md5_now} != {DATASET_MD5}  ({archive_path})")
        else:
            print(f"[DATA] MD5 OK: {md5_now}")

    # 解压
    print(f"[DATA] Extracting to: {DATASET_DIR}")
    _extract_archive(archive_path, DATASET_DIR)

    # 再次定位
    base = _auto_locate_fitbit_base(DATASET_DIR)
    print(f"[DATA] Ready. Fitbit base: {base}")
    return base

# ----------------------------
# Logging
# ----------------------------
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)

# ----------------------------
# 数据根目录（可用环境变量覆盖）
# 例如：/home/shi/d_folder/.../Fitabase Data 4.12.16-5.12.16
# ----------------------------
FITBIT_BASE_DIR = None

'''FITBIT_BASE_DIR = os.environ.get(
    "FITBIT_BASE_DIR",
    "/home/shi/d_folder/fedops/MNIST/dataset/archive/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16"
)'''

FEATURES = ["Steps", "Calories", "AvgHeartRate", "StressLevel"]
LABEL = "SleepQuality"


# ---- (可选) 统一标签到 0/1（若本来就是 0/1，此函数等价恒等） ----
def _binarize_np(y: np.ndarray) -> np.ndarray:
    u = set(np.unique(y).tolist())
    if u <= {0, 1}:
        return y.astype(int)
    if u <= {-1, 1}:
        return (y > 0).astype(int)
    if u <= {1, 2}:
        return (y >= 2).astype(int)
    return (y > 0).astype(int)


# ============================ 原始数据读取与融合 ============================

def _load_fitbit_raw() -> pd.DataFrame:
    """
    读取 Fitabase CSV，按 (UserId, Hour) 对齐步数/卡路里/心率，并与日级睡眠标签合并。
    产出列：['UserId','Hour','Steps','Calories','AvgHeartRate','StressLevel','SleepQuality']（按时间排序）
    """
    global FITBIT_BASE_DIR
    if FITBIT_BASE_DIR is None:
        FITBIT_BASE_DIR = _ensure_fitbit_data().as_posix()
    # 1) 睡眠（日级标签）
    sleep_fp = os.path.join(FITBIT_BASE_DIR, "sleepDay_merged.csv")
    sleep_df = pd.read_csv(sleep_fp, parse_dates=["SleepDay"])
    # 7 小时=420 分钟及以上记为高质量睡眠(1)
    sleep_df["SleepQuality"] = (sleep_df["TotalMinutesAsleep"] >= 420).astype(int)
    sleep_df["SleepDate"] = pd.to_datetime(sleep_df["SleepDay"].dt.date)

    # 2) 步数/卡路里（小时级）
    steps_fp = os.path.join(FITBIT_BASE_DIR, "hourlySteps_merged.csv")
    cals_fp  = os.path.join(FITBIT_BASE_DIR, "hourlyCalories_merged.csv")
    steps_df = pd.read_csv(steps_fp, parse_dates=["ActivityHour"])
    cals_df  = pd.read_csv(cals_fp,  parse_dates=["ActivityHour"])
    activity_df = pd.merge(steps_df, cals_df, on=["Id", "ActivityHour"], how="inner")

    # 3) 心率（秒级 → 按用户聚合到小时均值）
    hr_fp = os.path.join(FITBIT_BASE_DIR, "heartrate_seconds_merged.csv")
    hr_df = pd.read_csv(hr_fp, parse_dates=["Time"])
    hr_df["Hour"] = hr_df["Time"].dt.floor("H")
    hr_hourly = (hr_df.groupby(["Id", "Hour"])["Value"].mean()
                     .reset_index()
                     .rename(columns={"Hour": "ActivityHour", "Value": "AvgHeartRate"}))

    # 4) 合并：(Id, ActivityHour)
    merged = pd.merge(activity_df, hr_hourly, on=["Id", "ActivityHour"], how="inner")

    # 5) 用 ActivityHour 的日期与日级睡眠标签合并
    merged["SleepDate"] = pd.to_datetime(merged["ActivityHour"].dt.date)
    final = pd.merge(
        merged,
        sleep_df[["Id", "SleepDate", "SleepQuality"]],
        on=["Id", "SleepDate"],
        how="inner",
    )

    # 6) StressLevel = AvgHR - RestingHR（每人心率的 25 分位作为静息心率粗估）
    resting = final.groupby("Id")["AvgHeartRate"].quantile(0.25).rename("RestingHR").reset_index()
    final = final.merge(resting, on="Id", how="left")
    final["StressLevel"] = final["AvgHeartRate"] - final["RestingHR"]
    final.drop(columns=["RestingHR"], inplace=True)

    # 7) 整理列名与排序
    final = final[["Id", "ActivityHour", "StepTotal", "Calories", "AvgHeartRate", "StressLevel", "SleepQuality"]]
    final = final.rename(columns={
        "Id": "UserId",
        "ActivityHour": "Hour",
        "StepTotal": "Steps",
    }).dropna()
    final = final.sort_values(["UserId", "Hour"]).reset_index(drop=True)
    return final


# ============================ 窗口化与数据集 ============================

def create_sequences_by_user(final_df: pd.DataFrame,
                             seq_length: int = 6,
                             feature_cols=FEATURES,
                             label_col=LABEL,
                             restrict_hours=None):
    """
    按用户分组做滑窗：X[i]= t..t+T-1 的特征，y[i]= t+T 时刻的 SleepQuality。
    返回：X:(N,T,F) float32, y:(N,1) float32(0/1), groups:(N,) 记录 userId
    """
    df = final_df.copy()
    df["Hour"] = pd.to_datetime(df["Hour"])
    if restrict_hours is not None:
        hod = df["Hour"].dt.hour
        df = df[hod.isin(list(restrict_hours))]

    df = df.sort_values(["UserId", "Hour"])
    X, y, groups = [], [], []

    for uid, g in df.groupby("UserId", sort=False):
        f = g[feature_cols].to_numpy(dtype=np.float32)     # [M, F]
        l = g[label_col].to_numpy()                        # [M]
        if len(f) <= seq_length:
            continue
        for i in range(len(f) - seq_length):
            X.append(f[i:i+seq_length])                    # [T, F]
            y.append(l[i + seq_length])                    # 下一时刻标签
            groups.append(uid)

    if len(X) == 0:
        return (np.empty((0, seq_length, len(feature_cols)), np.float32),
                np.empty((0, 1), np.float32),
                np.array([], dtype=object))

    X = np.stack(X, axis=0)                 # (N, T, F)
    y = _binarize_np(np.array(y)).reshape(-1, 1).astype(np.float32)  # (N,1)
    groups = np.array(groups)
    return X, y, groups


class SeqDataset(Dataset):
    """窗口序列数据集：返回 (X, y)，其中 X:(T,F) float32, y:(1,) float32"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================ Loader 构建（按用户切分，训练集拟合缩放器） ============================

def _build_loaders_windowed(final_df: pd.DataFrame,
                            seq_length: int = 6,
                            batch_size: int = 64,
                            split_by_user: bool = True,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            seed: int = 42,
                            restrict_hours=None,
                            num_workers: int = 0,
                            pin_memory: bool = True):
    """
    主入口：窗口化 → 分割 → 仅用训练集拟合 MinMaxScaler → 产出 DataLoader（三分）。
    """
    X, y, groups = create_sequences_by_user(final_df, seq_length, FEATURES, LABEL, restrict_hours)

    N = len(X)
    if N == 0:
        raise ValueError("没有生成任何序列样本；请检查数据范围/seq_length/restrict_hours。")

    rng = np.random.default_rng(seed)

    # ---- 划分：按用户（推荐）或全局分层 ----
    if split_by_user:
        # 先切 train vs hold
        gss = GroupShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
        train_idx, hold_idx = next(gss.split(np.arange(N), y.reshape(-1), groups))

        # 再从 hold 中切 val/test（这里不再分组，避免样本太少）
        val_size = int(round(len(hold_idx) * val_ratio / max((1 - train_ratio), 1e-8)))
        hold_idx = rng.permutation(hold_idx)
        val_idx  = hold_idx[:val_size]
        test_idx = hold_idx[val_size:]
    else:
        # 全局分层随机
        idx = np.arange(N)
        tr_idx, hold_idx = train_test_split(idx, test_size=(1 - train_ratio),
                                            stratify=y.reshape(-1), random_state=seed)
        val_size = int(round(len(hold_idx) * val_ratio / max((1 - train_ratio), 1e-8)))
        hold_idx = rng.permutation(hold_idx)
        val_idx  = hold_idx[:val_size]
        test_idx = hold_idx[val_size:]
        train_idx = tr_idx

    # ---- 仅用 train 拟合缩放器（防泄漏） ----
    F = X.shape[-1]
    scaler = MinMaxScaler()
    X_train_2d = X[train_idx].reshape(-1, F)
    scaler.fit(X_train_2d)

    def _apply_scale(X_in: np.ndarray) -> np.ndarray:
        sh = X_in.shape
        return scaler.transform(X_in.reshape(-1, sh[-1])).reshape(sh).astype(np.float32)

    X_train = _apply_scale(X[train_idx]); y_train = y[train_idx]
    X_val   = _apply_scale(X[val_idx]);   y_val   = y[val_idx]
    X_test  = _apply_scale(X[test_idx]);  y_test  = y[test_idx]

    # ---- DataLoader ----
    ds_train = SeqDataset(X_train, y_train)
    ds_val   = SeqDataset(X_val,   y_val)
    ds_test  = SeqDataset(X_test,  y_test)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    # ---- 统计信息（训练集类别不平衡可见）----
    p_train = int(y_train.sum()); n_train = len(y_train) - p_train
    logger.info(f"[FITBIT_SLEEP] seq_len={seq_length}, feat_dim={F}, "
                f"train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}, "
                f"train_pos%={p_train/max(len(y_train),1):.3f} ({p_train}/{len(y_train)})")

    return dl_train, dl_val, dl_test


# ============================ 对外主接口（与原项目保持一致） ============================

def load_partition(dataset: str,
                   validation_split: float,
                   batch_size: int,
                   seq_length: int = 6,
                   test_split: float = 0.2,
                   seed: int = 42,
                   restrict_hours=None,
                   num_workers: int = 0,
                   pin_memory: bool = True):
    """
    统一入口（兼容你现有调用签名）：
      - 读取 Fitabase 数据，做 6 小时窗口化，按用户分组切分 train/val/test。
      - 返回 (train_loader, val_loader, test_loader)
    参数：
      - dataset: 仅用于日志（为了兼容旧签名）
      - validation_split: 验证集占比（例如 0.15）
      - batch_size: 批大小
      - seq_length: 窗口长度（默认 6 小时）
      - test_split: 测试集占比（默认 0.2）
    """
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'FL_Task - {json.dumps({"dataset": dataset, "start_execution_time": now_str})}')

    final_df = _load_fitbit_raw()

    # 计算各子集比例
    val_ratio = float(validation_split)
    test_ratio = float(test_split)
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError(f"train_ratio <= 0（validation_split={val_ratio}, test_split={test_ratio}），请调整比例。")

    return _build_loaders_windowed(
        final_df=final_df,
        seq_length=seq_length,
        batch_size=batch_size,
        split_by_user=True,          # 强烈建议按用户切分，避免泄漏
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        restrict_hours=restrict_hours,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

