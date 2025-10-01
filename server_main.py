"""pytorch-example: A Flower / PyTorch app."""

import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from fedops.server.app import FLServer
import models  # noqa: F401
import data_preparation  # noqa: F401

# ---------------- 仿真数据集 ----------------
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class FakeSleepSeqDataset(Dataset):
    """
    生成形状 [N, seq_len, 3] 的序列特征：
      f0=Steps(>=0)、f1=Stress(0~100)、f2=AvgHeartRate(40~180)
    规则可学且类均衡的二分类标签 y ∈ {0,1}
    """
    def __init__(self, n_samples=512, seq_len=6, seed=7):
        rng = np.random.default_rng(seed)
        steps  = rng.poisson(lam=350, size=(n_samples, seq_len, 1))          # 活动量
        stress = rng.normal(45, 18, size=(n_samples, seq_len, 1))            # 压力
        hr     = rng.normal(75, 15, size=(n_samples, seq_len, 1))            # 心率
        steps  = np.clip(steps, 0, 2000); stress = np.clip(stress, 0, 100); hr = np.clip(hr, 40, 180)
        x = np.concatenate([steps, stress, hr], axis=-1).astype(np.float32)

        mean_steps  = x[:, :, 0].mean(axis=1)
        mean_stress = x[:, :, 1].mean(axis=1)
        mean_hr     = x[:, :, 2].mean(axis=1)
        score = (-0.003 * mean_steps) + (-0.04 * mean_stress) + (-0.02 * mean_hr) + rng.normal(0, 0.2, size=(n_samples,))
        y = (score > np.median(score)).astype(np.int64)

        mu = x.mean(axis=(0, 1), keepdims=True)
        sigma = x.std(axis=(0, 1), keepdims=True) + 1e-6
        x = (x - mu) / sigma

        self.x = torch.from_numpy(x)    # [N, L, 3]
        self.y = torch.from_numpy(y)    # [N]

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def _make_fake_loader(n_samples=512, seq_len=6, batch_size=32, seed=7):
    ds = FakeSleepSeqDataset(n_samples=n_samples, seq_len=seq_len, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------- 设备解析与评测 ----------------
def _device_to_str(dev):
    """DictConfig/dict/str/torch.device -> 'cpu'/'cuda:0'"""
    try:
        from omegaconf import DictConfig
    except Exception:
        DictConfig = ()  # 允许缺省
    if isinstance(dev, torch.device):
        return str(dev)
    if isinstance(dev, (str, bytes)):
        return dev.decode() if isinstance(dev, bytes) else dev
    if isinstance(dev, (DictConfig, dict)):
        t = dev.get("type", "cpu"); idx = dev.get("index", None)
        return f"{t}:{idx}" if idx is not None else f"{t}"
    t = getattr(dev, "type", "cpu"); idx = getattr(dev, "index", None)
    return f"{t}:{idx}" if idx is not None else f"{t}"


def _resolve_eval_device(cfg_or_device):
    """
    既兼容传 cfg（优先取 cfg.server.device）也兼容直接传 device。
    """
    if isinstance(cfg_or_device, (str, bytes, dict, torch.device)):
        dev_str = _device_to_str(cfg_or_device)
    else:
        # 认为是 cfg
        dev = (
            getattr(getattr(cfg_or_device, "server", {}), "device", None)
            or getattr(cfg_or_device, "device", None)
            or "cpu"
        )
        dev_str = _device_to_str(dev)

    # cuda 不可用则回落
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev_str


@torch.no_grad()
def evaluate_loader(model, loader, device="cpu", class_weight=(1.0, 1.0)):
    """
    返回 dict：{"loss":..., "acc":..., "prec":..., "rec":...}
    - 自动判定二分类(1 logit) 或 多分类(C logits)
    - 自动尝试输入形状 [B, L, C] 与 [B, C, L]
    """
    dev = torch.device(_resolve_eval_device(device))
    model = model.to(dev).eval()

    # 预读一批决定头部形状
    try:
        first = next(iter(loader))
    except StopIteration:
        return {"loss": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0}

    xb, yb = first
    xb = xb.to(dev); yb = yb.to(dev)

    def _forward_safe(x):
        try:
            return model(x)
        except Exception:
            return model(x.transpose(1, 2))  # 兼容 [B,C,L] 期望

    out = _forward_safe(xb)
    num_classes = out.shape[-1] if out.dim() > 1 else 1

    # 损失函数（带可选权重）
    if num_classes == 1:
        # 将 (w0, w1) 映射为 pos_weight = w1/w0
        w0, w1 = float(class_weight[0]), float(class_weight[1])
        pos_weight = torch.tensor([w1 / max(w0, 1e-8)], device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # 若给了与类别数一致的权重则使用
        w = None
        if isinstance(class_weight, (list, tuple)) and len(class_weight) == num_classes:
            w = torch.tensor(class_weight, dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=w)

    # 累计指标
    total_loss, total_correct, total_samples = 0.0, 0, 0
    # 混淆计数（用于 prec/rec）
    if num_classes == 1:
        tp = fp = fn = 0
    else:
        tp = torch.zeros(num_classes, device=dev)
        fp = torch.zeros(num_classes, device=dev)
        fn = torch.zeros(num_classes, device=dev)

    # 遍历
    for xb, yb in loader:
        xb = xb.to(dev); yb = yb.to(dev)
        out = _forward_safe(xb)

        if num_classes == 1:
            logits = out.squeeze(-1)
            loss = criterion(logits, yb.float())
            pred = (logits > 0).long()
            total_correct += (pred == yb.long()).sum().item()
            # 统计
            tp += ((pred == 1) & (yb == 1)).sum()
            fp += ((pred == 1) & (yb == 0)).sum()
            fn += ((pred == 0) & (yb == 1)).sum()
        else:
            loss = criterion(out, yb.long())
            pred = out.argmax(dim=-1)
            total_correct += (pred == yb.long()).sum().item()
            # 统计每类
            for k in range(num_classes):
                tp[k] += ((pred == k) & (yb == k)).sum()
                fp[k] += ((pred == k) & (yb != k)).sum()
                fn[k] += ((pred != k) & (yb == k)).sum()

        total_loss += loss.item()
        total_samples += yb.numel()

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_samples, 1)

    if num_classes == 1:
        tp, fp, fn = float(tp), float(fp), float(fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        # macro 平均
        prec_k = tp / torch.clamp(tp + fp, min=1)
        rec_k  = tp / torch.clamp(tp + fn, min=1)
        prec = float(prec_k.mean().item())
        rec  = float(rec_k.mean().item())

    return {"loss": float(avg_loss), "acc": float(acc), "prec": float(prec), "rec": float(rec)}


def test_torch():
    """
    返回: custom_test(model, test_loader, cfg_or_device) -> (avg_loss, acc, metrics)
    metrics: {"prec":..., "rec":..., "f1":...}
    - 第三个参数既可传 cfg（从 cfg.server.device 取设备），也可直接传 device
    - 评测阶段 class_weight 只影响 loss，不影响 acc/prec/rec
    """
    def custom_test(model, test_loader, cfg_or_device):
        # 读取 class_weight（可选）
        if isinstance(cfg_or_device, (str, bytes, dict, torch.device)):
            class_weight = (1.0, 1.0)
            device_arg = cfg_or_device
        else:
            cw = (
                getattr(getattr(cfg_or_device, "eval", {}), "class_weight", None)
                or getattr(getattr(cfg_or_device, "server", {}), "class_weight", None)
            )
            class_weight = tuple(cw) if cw is not None else (1.0, 1.0)
            device_arg = cfg_or_device

        stats = evaluate_loader(
            model, test_loader,
            device=device_arg,
            class_weight=class_weight,
        )
        loss = float(stats["loss"])
        acc  = float(stats["acc"])
        prec = float(stats["prec"])
        rec  = float(stats["rec"])
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics = {"prec": prec, "rec": rec, "f1": f1}
        return loss, acc, metrics
    return custom_test


# ---------------- 主流程 ----------------
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 环境开关（可选）
    skip_data = bool(
        getattr(getattr(cfg, "server", {}), "skip_data", False)
        or os.environ.get("SKIP_DATA", "0") not in ("0", "", "false", "False")
    )
    skip_eval = bool(
        getattr(getattr(cfg, "server", {}), "skip_eval", False)
        or os.environ.get("SKIP_EVAL", "0") not in ("0", "", "false", "False")
    )

    # 1) 构建初始化全局模型
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    # 2) 全局评测集（仿真）
    seq_len = getattr(cfg.dataset, "seq_len", 6)
    n_fake  = int(os.environ.get("FAKE_EVAL_N", "512"))
    gl_val_loader = _make_fake_loader(n_samples=n_fake, seq_len=seq_len,
                                      batch_size=cfg.batch_size, seed=7)

    # 3) 评测函数（返回 (avg_loss, acc, metrics)）
    gl_test_torch = test_torch()

    # 4) 启动联邦服务端
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch,   # 注意：第三参可传 cfg 或 device，二者皆可
    )
    fl_server.start()


if __name__ == "__main__":
    main()
