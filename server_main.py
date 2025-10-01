"""pytorch-example: A Flower / PyTorch app."""

import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from fedops.server.app import FLServer
import models
import data_preparation


# --- 新增：仿真数据集 & 通用评测 ---
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class FakeSleepSeqDataset(Dataset):
    """
    生成形状 [N, seq_len, 3] 的序列特征：
      f0=Steps(非负)、f1=Stress(0~100)、f2=AvgHeartRate(40~180)
    标签 y ∈ {0,1} 由一个可解释的简单规则产生，便于模型可学。
    """
    def __init__(self, n_samples=512, seq_len=6, seed=7):
        rng = np.random.default_rng(seed)

        # --- 生成每条序列 ---
        steps  = rng.poisson(lam=350, size=(n_samples, seq_len, 1))           # ~活动量
        stress = rng.normal(45, 18, size=(n_samples, seq_len, 1))             # ~压力
        hr     = rng.normal(75, 15, size=(n_samples, seq_len, 1))             # ~心率

        # 裁剪到合理范围
        steps  = np.clip(steps, 0, 2000)
        stress = np.clip(stress, 0, 100)
        hr     = np.clip(hr, 40, 180)

        x = np.concatenate([steps, stress, hr], axis=-1).astype(np.float32)

        # --- 生成可学习的“伪标签” ---
        # 直觉：低步数 + 低压力 + 较低心率 => 正类(1)
        mean_steps  = x[:, :, 0].mean(axis=1)
        mean_stress = x[:, :, 1].mean(axis=1)
        mean_hr     = x[:, :, 2].mean(axis=1)
        score = (-0.003 * mean_steps) + (-0.04 * mean_stress) + (-0.02 * mean_hr) + rng.normal(0, 0.2, size=(n_samples,))
        y = (score > np.median(score)).astype(np.int64)   # 二分类 0/1，类均衡

        # 可选：做简单标准化（更贴近真实训练）
        mu = x.mean(axis=(0,1), keepdims=True)
        sigma = x.std(axis=(0,1), keepdims=True) + 1e-6
        x = (x - mu) / sigma

        self.x = torch.from_numpy(x)             # [N, L, 3]
        self.y = torch.from_numpy(y)             # [N]

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def _make_fake_loader(n_samples=512, seq_len=6, batch_size=32, seed=7):
    ds = FakeSleepSeqDataset(n_samples=n_samples, seq_len=seq_len, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def _make_eval_fn():
    """
    通用评测：自动适配二分类/多分类。
    要求：DataLoader 每步产出 (x, y)；模型前向返回 logits。
    """
    def _inner(model, loader, device):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        model.eval()
        n, correct, total_loss = 0, 0, 0.0
        # 先跑一个 batch 判断类别数
        it = iter(loader)
        try:
            xb, yb = next(it)
        except StopIteration:
            return 0.0, 0, {"acc": 0.0}

        xb = xb.to(dev); yb = yb.to(dev)
        with torch.no_grad():
            out = model(xb)                       # [B, C] 或 [B, 1]
        num_classes = out.shape[-1] if out.dim() > 1 else 1

        # 选择合适的 loss/pred
        if num_classes == 1:                      # 二分类（logits）
            criterion = nn.BCEWithLogitsLoss()
            pred = (out.squeeze(-1) > 0).long()
            correct += (pred == yb.long()).sum().item()
            total_loss += criterion(out.squeeze(-1), yb.float()).item()
            n += yb.numel()
        else:                                     # 多分类
            criterion = nn.CrossEntropyLoss()
            pred = out.argmax(dim=-1)
            correct += (pred == yb.long()).sum().item()
            total_loss += criterion(out, yb.long()).item()
            n += yb.numel()

        # 继续剩余 batch
        for xb, yb in it:
            xb = xb.to(dev); yb = yb.to(dev)
            with torch.no_grad():
                out = model(xb)
                if num_classes == 1:
                    pred = (out.squeeze(-1) > 0).long()
                    total_loss += criterion(out.squeeze(-1), yb.float()).item()
                else:
                    pred = out.argmax(dim=-1)
                    total_loss += criterion(out, yb.long()).item()
                correct += (pred == yb.long()).sum().item()
                n += yb.numel()

        acc = correct / max(n, 1)
        loss = total_loss / max(len(loader), 1)
        return float(loss), int(n), {"acc": float(acc)}
    return _inner


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- 读跳过开关：优先配置，其次环境变量 ---
    # 可在 config 里加 server.skip_data / server.skip_eval，或用环境变量 SKIP_DATA / SKIP_EVAL
    skip_data = bool(
        getattr(getattr(cfg, "server", {}), "skip_data", False)
        or os.environ.get("SKIP_DATA", "0") not in ("0", "", "false", "False")
    )
    skip_eval = bool(
        getattr(getattr(cfg, "server", {}), "skip_eval", False)
        or os.environ.get("SKIP_EVAL", "0") not in ("0", "", "false", "False")
    )

    # 1) 构建初始化全局模型（SleepSeqClassifier 等）
    model = instantiate(cfg.model)              # cfg.model._target_ -> models.SleepSeqClassifier
    model_type = cfg.model_type                 # 一般为 'torch'
    model_name = type(model).__name__

    # 2) （可选）构建 Fitbit 数据的 DataLoader；若跳过则给空 Loader

    # 2) 用仿真数据作为全局评测集
    seq_len = getattr(cfg.dataset, "seq_len", 6)
    n_fake  = int(os.environ.get("FAKE_EVAL_N", "512"))
    gl_val_loader = _make_fake_loader(n_samples=n_fake, seq_len=seq_len,
                                      batch_size=cfg.batch_size, seed=7)
    
    # 3) 评测函数（真正计算 loss/acc）
    gl_test_torch = _make_eval_fn()


    # 4) 启动联邦服务端
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch,
    )
    fl_server.start()


if __name__ == "__main__":
    main()

