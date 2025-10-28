#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from os.path import dirname, realpath, exists, join
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# (옵션) diffusers의 EMA / scheduler만 재활용
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# ---- 프로젝트 경로 설정
dir_project = dirname(dirname(realpath(__file__)))
print("dir_project :", dir_project)
sys.path.append(dir_project)
DATA_PATH = os.path.join(dir_project, "data")

# ---- 데이터셋/모델 임포트
# 절대 관절(7D) 타깃 데이터셋
from src.utils.CustomDataset_joint2 import CustomKinovaJointDataset
# dq 없이, RGB+Depth+q -> 미래 q 예측 Transformer
from models.transformer import TransformerJointGripperNoDQ


# ---------------- ROS2 안전한 stats 저장 ----------------
def save_stats_npz_ros_safe(stats: dict, save_path: str, meta: dict = None):
    """
    ROS2 Humble에서 allow_pickle 없이 안전하게 로드되도록
    object 배열을 쓰지 않고 'flat keys'로 저장한다.
      예) stats['q']['min'] -> key='stats__q__min'
    """
    payload = {}
    for k, sub in stats.items():
        if not isinstance(sub, dict):
            continue
        for sk, arr in sub.items():
            key = f"stats__{k}__{sk}"  # ex) stats__q__min
            payload[key] = np.asarray(arr, dtype=np.float32)
    if meta:
        for mk, mv in meta.items():
            if isinstance(mv, (int, float, np.integer, np.floating)):
                payload[f"meta__{mk}"] = np.array([mv], dtype=np.float32)
            elif isinstance(mv, str):
                payload[f"meta__{mk}"] = np.array(mv)
            else:
                try:
                    payload[f"meta__{mk}"] = np.asarray(mv)
                except Exception:
                    pass
    np.savez_compressed(save_path, **payload)
    print(f"[stats] ROS-safe saved: {save_path} (keys={list(payload.keys())[:6]} ...)")


@torch.no_grad()
def evaluate_ema_loss(model: nn.Module,
                      ema: EMAModel,
                      loader: DataLoader,
                      device: torch.device,
                      use_amp: bool) -> float:
    """EMA 가중치로 전환하여 검증 손실(MSE on pred vs gt_joint) 계산."""
    if loader is None or len(loader) == 0:
        return float("nan")

    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()

    losses = []
    for nbatch in loader:
        obs_rgb   = nbatch["obs_rgb"].to(device)     # (B,K,3,H,W) 0~1
        obs_depth = nbatch["obs_depth"].to(device)   # (B,K,1,H,W) 0~1
        obs_q     = nbatch["obs_q"].to(device)       # (B,K,7)     [-1,1]
        gt_joint  = nbatch["gt_joint"].to(device)    # (B,T,7)     [-1,1]

        use_autocast = use_amp and (device.type == 'cuda')
        with torch.amp.autocast('cuda', enabled=use_autocast):
            pred = model(obs_rgb, obs_depth, obs_q, pred_horizon=gt_joint.shape[1])  # (B,T,7)
            loss = nn.functional.mse_loss(pred, gt_joint)

        losses.append(float(loss.item()))

    ema.restore(model.parameters())
    model.train()
    return float(np.mean(losses)) if len(losses) else float("nan")


def save_checkpoint(path, model, ema, optimizer, lr_scheduler, scaler,
                    epoch_idx, best_val, global_step):
    ckpt = {
        "epoch": epoch_idx,
        "best_val": best_val,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "ema_state": (ema.state_dict() if ema is not None else None),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": (lr_scheduler.state_dict() if lr_scheduler is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, ema, optimizer, lr_scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])

    ema_state = ckpt.get("ema_state", None)
    if ema is not None and ema_state is not None:
        try:
            ema.load_state_dict(ema_state)
        except Exception as e:
            print(f"[resume][warn] ema.load_state_dict failed ({e}); continuing without EMA state.")
    else:
        print("[resume] no EMA to load (ema is None or ckpt has no ema_state).")

    opt_state = ckpt.get("optimizer_state", None)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    lrs = ckpt.get("lr_scheduler_state", None)
    if lr_scheduler is not None and lrs is not None:
        lr_scheduler.load_state_dict(lrs)

    scs = ckpt.get("scaler_state", None)
    if scaler is not None and scs is not None:
        scaler.load_state_dict(scs)

    epoch = int(ckpt.get("epoch", -1))
    best_val = float(ckpt.get("best_val", float("inf")))
    global_step = int(ckpt.get("global_step", 0))
    print(f"[resume] loaded checkpoint: {path} (epoch={epoch}, best_val={best_val:.6f}, step={global_step})")
    return epoch, best_val, global_step


# ---------------------------------------
# 하이퍼파라미터 / 실행
# ---------------------------------------
if __name__ == "__main__":
    model_name = "transformer_joint_abs_2"

    # split & resume & logging
    valid_tail_n = 10
    resume_training = True
    eval_every_epoch = 1
    save_every_epoch = True

    # train
    num_epochs = 50
    batch_size = 64
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-6
    grad_clip_norm = 1.0
    use_amp = True

    # sequence
    obs_horizon = 8
    pred_horizon = 12
    action_horizon = 6

    # model config
    d_model = 512
    nhead = 8
    tfm_layers = 6
    dim_ff = 2048
    tfm_dropout = 0.1
    img_feat_dim = 512
    q_feat_dim = 256
    out_dim = 7

    # ---------------------------------------
    # 데이터 split
    # ---------------------------------------
    use_file = 800
    DATA_FILES = sorted(glob(join(DATA_PATH, "scenario_*.npz")))
    DATA_FILES = DATA_FILES[:use_file]
    if len(DATA_FILES) == 0:
        raise FileNotFoundError(f"No NPZ files under {DATA_PATH}")
    for p in DATA_FILES:
        if not exists(p):
            raise FileNotFoundError(f"{p} not found")

    if valid_tail_n < 1 or valid_tail_n >= len(DATA_FILES):
        raise ValueError(f"valid_tail_n must be in [1, {len(DATA_FILES)-1}]")

    train_files = DATA_FILES[:-valid_tail_n]
    valid_files = DATA_FILES[-valid_tail_n:]

    print(f"[data] total={len(DATA_FILES)}, train={len(train_files)}, valid={len(valid_files)}")
    print("[data] valid files (tail):")
    for p in valid_files:
        print("  -", os.path.basename(p))

    # ---------------------------------------
    # Dataset / Dataloader
    # ---------------------------------------
    common_kwargs = dict(
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        resize_hw=None,              # 원본 해상도(예: 240x424) 그대로
        max_depth_m=1.0,
        max_episode_length=None,
        normalize_rgb01=True,
        normalize_depth01=True,
        clip_low=1.0,
        clip_high=99.0,
        grip_index=0,
    )

    train_dataset = CustomKinovaJointDataset(file_paths=train_files, **common_kwargs)
    valid_dataset = CustomKinovaJointDataset(file_paths=valid_files, **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    # ---------------------------------------
    # Model / Optim / EMA / Scheduler
    # ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerJointGripperNoDQ(
        d_model=d_model, nhead=nhead,
        num_encoder_layers=tfm_layers,
        num_decoder_layers=tfm_layers,
        dim_feedforward=dim_ff,
        dropout=tfm_dropout,
        img_feat_dim=img_feat_dim,
        q_feat_dim=q_feat_dim,
        pred_horizon=pred_horizon,
        out_dim=out_dim,
        use_tanh_out=True,          # [-1,1] 범위로 출력
    ).to(device)

    ema = EMAModel(parameters=model.parameters(), power=0.75)
    ema.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs
    )

    if use_amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.amp.GradScaler('cpu')

    # ---------------------------------------
    # Checkpoint/재시작
    # ---------------------------------------
    save_dir = join(dir_project, "trained_model")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = join(save_dir, f"{model_name}_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    last_ckpt_path = join(save_dir, f"{model_name}_last.ckpt")
    best_model_path = join(save_dir, f"{model_name}_model_best.pt")
    param_path = join(save_dir, f"{model_name}_param.npz")
    stats_path = join(save_dir, f"{model_name}_stats.npz")

    start_epoch = 0
    best_val = float("inf")
    global_step = 0

    if resume_training and exists(last_ckpt_path):
        start_epoch, best_val, global_step = load_checkpoint(
            last_ckpt_path, model, ema, optimizer, lr_scheduler, scaler
        )
        if ema is not None:
            ema.to(device)
        start_epoch += 1

    # ---------------------------------------
    # Training Loop
    # ---------------------------------------
    model.train()
    with tqdm(range(start_epoch, num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(train_loader, desc='Batch', leave=False, disable=True) as tepoch:
                for nbatch in tepoch:
                    obs_rgb   = nbatch["obs_rgb"].to(device)    # (B,K,3,H,W)
                    obs_depth = nbatch["obs_depth"].to(device)  # (B,K,1,H,W)
                    obs_q     = nbatch["obs_q"].to(device)      # (B,K,7)
                    gt_joint  = nbatch["gt_joint"].to(device)   # (B,T,7)

                    obs_rgb_np = obs_rgb.detach().cpu().numpy()
                    obs_depth_np = obs_depth.detach().cpu().numpy()
                    obs_q_np = obs_q.detach().cpu().numpy()
                    gt_joint_np = gt_joint.detach().cpu().numpy()

                    optimizer.zero_grad(set_to_none=True)

                    use_autocast = use_amp and (device.type == 'cuda')
                    with torch.amp.autocast('cuda', enabled=use_autocast):
                        pred = model(obs_rgb, obs_depth, obs_q, pred_horizon=gt_joint.shape[1])  # (B,T,7)
                        loss = nn.functional.mse_loss(pred, gt_joint)

                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    lr_scheduler.step()
                    ema.step(model.parameters())

                    epoch_loss.append(float(loss.item()))
                    global_step += 1
                    tepoch.set_postfix(loss=float(loss.item()))

            mean_train = float(np.mean(epoch_loss)) if epoch_loss else float("nan")

            # --- 검증 (EMA) ---
            do_eval = ((epoch_idx + 1 - start_epoch) % eval_every_epoch == 0)
            if do_eval:
                val_loss = evaluate_ema_loss(model, ema, valid_loader, device, use_amp)
            else:
                val_loss = float("nan")

            tglobal.set_postfix(train_loss=mean_train, val_loss=val_loss)
            print(f'[epoch {epoch_idx + 1}] train_loss: {mean_train:.6f} | val_loss(EMA): {val_loss:.6f}')

            # --- last.ckpt 저장 ---
            if save_every_epoch:
                save_checkpoint(join(save_dir, f"{model_name}_last.ckpt"),
                                model, ema, optimizer, lr_scheduler, scaler,
                                epoch_idx=epoch_idx, best_val=best_val, global_step=global_step)
                save_checkpoint(join(ckpt_dir, f"epoch_{epoch_idx:04d}.ckpt"),
                                model, ema, optimizer, lr_scheduler, scaler,
                                epoch_idx=epoch_idx, best_val=best_val, global_step=global_step)

            # --- best (EMA) 저장 ---
            if not np.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                # EMA 파라미터로 저장
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                torch.save(model.state_dict(), best_model_path)
                ema.restore(model.parameters())

                # 파라미터/통계 저장 (ROS-safe)
                np.savez(
                    param_path,
                    pred_horizon=pred_horizon,
                    obs_horizon=obs_horizon,
                    action_horizon=action_horizon,
                    d_model=d_model,
                    nhead=nhead,
                    tfm_layers=tfm_layers,
                    dim_ff=dim_ff,
                    tfm_dropout=tfm_dropout,
                    img_feat_dim=img_feat_dim,
                    q_feat_dim=q_feat_dim,
                    out_dim=out_dim,
                )
                save_stats_npz_ros_safe(
                    train_dataset.stats,   # .stats dict 그대로
                    stats_path,
                    meta={"model_name": model_name, "note": "ros-safe flat stats (q/dq with grip idx override)"}
                )
                print(f"[best] improved → saved best model & params & ROS-safe stats (val={best_val:.6f})")

    # 종료 후 last 저장
    save_checkpoint(join(save_dir, f"{model_name}_last.ckpt"),
                    model, ema, optimizer, lr_scheduler, scaler,
                    epoch_idx=num_epochs - 1, best_val=best_val, global_step=global_step)
    print("Finished. last.ckpt saved at:", join(save_dir, f"{model_name}_last.ckpt"))
    print("Best EMA model (if improved) at:", best_model_path)
