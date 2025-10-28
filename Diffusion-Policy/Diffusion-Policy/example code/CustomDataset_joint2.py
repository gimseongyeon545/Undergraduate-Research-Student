# robot_manipulation/src/utils/CustomDataset.py
# - 목표: 팔 6 + 그리퍼 1 = 총 7D 절대 관절값을 미래 T 프레임 타깃으로 예측
# - 포인트:
#   * 데이터의 q, dq는 [N, 7] (gripper 차원은 grip_index로 지정)
#   * 정규화 통계에서 gripper 차원만 0.0~0.8로 고정
#   * 관측: 이미지(RGB/Depth) + (q, dq)
#   * 타깃: gt_joint = 미래 절대 q [T, 7] (정규화된 [-1,1], 안전하게 보장)

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Dict, Tuple, Optional

def make_windows(seq_len: int, K: int, T: int) -> np.ndarray:
    """(K 관측, T 예측) 가능한 중심 인덱스 t들의 배열을 만든다. t는 관측의 마지막 인덱스."""
    t_max = (seq_len - 1) - T
    if t_max < (K-1):
        return np.empty((0,), dtype=np.int64)
    return np.arange(K-1, t_max+1, dtype=np.int64)

class CustomKinovaJointDataset(Dataset):
    """
    입력: RGB/Depth, q(7), dq(7)
    타깃: 미래 T 프레임의 절대 q(7)  → gt_joint ∈ [-1,1] 정규화 (안전 클램프 포함)
    - 정규화:
        * q/dq: 각 차원 전체 min/max 기반  ← (퍼센타일 사용 안 함)
        * 단, gripper 차원(grip_index)은 min=0.0, max=0.8로 고정 (데이터 분포 무관)
    - 반환 키:
        obs_rgb      [K, 3, H, W]
        obs_depth    [K, 1, H, W]
        obs_q        [K, 7]   (정규화)
        obs_dq       [K, 7]   (정규화)
        gt_joint     [T, 7]   (정규화된 절대 q)
        ep_idx, t_index
    """

    def __init__(
        self,
        file_paths: List[str],
        obs_horizon: int,
        pred_horizon: int,
        resize_hw: Optional[Tuple[int,int]] = None,   # None이면 원본 해상도 유지
        max_depth_m: float = 2.5,
        max_episode_length: Optional[int] = None,
        normalize_rgb01: bool = True,
        normalize_depth01: bool = True,
        clip_low: float = 1.0,      # ← 보존(미사용)
        clip_high: float = 99.0,    # ← 보존(미사용)
        grip_index: int = 0,         # q에서 그리퍼 인덱스(예: 0 또는 -1 등)
    ):
        super().__init__()
        self.paths = list(file_paths)
        self.K = obs_horizon
        self.T = pred_horizon
        self.resize_hw = resize_hw
        self.H = resize_hw[0] if resize_hw is not None else None
        self.W = resize_hw[1] if resize_hw is not None else None

        self.max_depth_m = max_depth_m
        self.max_len = max_episode_length
        self.normalize_rgb01 = normalize_rgb01
        self.normalize_depth01 = normalize_depth01
        self.clip_low = clip_low      # ← 유지(호환용)
        self.clip_high = clip_high    # ← 유지(호환용)
        self.grip_index = grip_index

        # (1) 에피소드 스캔
        self.episodes = []
        for p in self.paths:
            try:
                z = np.load(p, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] skip {p}: {e}")
                continue
            req = ["rgb", "depth", "q", "dq"]
            if not all(k in z.files for k in req):
                print(f"[WARN] missing required keys in {p}, have {z.files}")
                continue

            N = z["rgb"].shape[0]
            if self.max_len is not None and N > self.max_len:
                N = self.max_len

            if self.resize_hw is None and (self.H is None or self.W is None):
                h0, w0 = z["rgb"][0].shape[:2]
                self.H, self.W = int(h0), int(w0)
            self.episodes.append(dict(path=p, N=N))

        if len(self.episodes) == 0:
            print("[WARN] No valid episodes found.")

        # (2) 정규화 통계 (전체 min/max)
        self.stats = self._compute_stats()

        # (3) 샘플 인덱스
        self.samples = []
        for ei, ep in enumerate(self.episodes):
            N = ep["N"]
            ts = make_windows(N, self.K, self.T)
            for t in ts:
                self.samples.append((ei, t))
        print(f"[CustomKinovaJointDataset] episodes={len(self.episodes)}, samples={len(self.samples)}, HxW={self.H}x{self.W}")

    # ---------- 이미지 전처리 ----------
    def _prep_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if self.resize_hw is not None:
            x = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            x = rgb.astype(np.float32)
        if self.normalize_rgb01:
            x = x / 255.0
        x = np.transpose(x, (2,0,1))  # [H,W,3]→[3,H,W]
        return x

    def _prep_depth(self, depth: np.ndarray) -> np.ndarray:
        if self.resize_hw is not None:
            d = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        else:
            d = depth.astype(np.float32)
        d = np.clip(d, 0.0, self.max_depth_m) / self.max_depth_m if self.normalize_depth01 else d
        d = d[None, ...]  # [H,W]→[1,H,W]
        return d

    # ---------- 정규화/역정규화 ----------
    def _norm(self, x: torch.Tensor, key: str) -> torch.Tensor:
        stat = self.stats.get(key, None)
        if stat is None:
            return x
        mn = torch.as_tensor(stat["min"], dtype=x.dtype, device=x.device)  # [...,7] 브로드캐스트 OK
        mx = torch.as_tensor(stat["max"], dtype=x.dtype, device=x.device)
        rng = torch.clamp(mx - mn, min=1e-6)

        # ① 값 자체를 [mn, mx]로 윈저라이즈(먼저 클립)
        x = torch.minimum(torch.maximum(x, mn), mx)
        # ② [-1,1]로 매핑
        y = 2.0 * (x - mn) / rng - 1.0
        # ③ 수치오차 방지용 최종 클램프
        # return torch.clamp(y, -1.0, 1.0)
        return y

    def _denorm(self, x: torch.Tensor, key: str) -> torch.Tensor:
        stat = self.stats.get(key, None)
        if stat is None:
            return x
        mn = torch.as_tensor(stat["min"], dtype=x.dtype, device=x.device)
        mx = torch.as_tensor(stat["max"], dtype=x.dtype, device=x.device)
        # 대칭 보정: 입력을 [-1,1]로 한번 더 보장
        x = torch.clamp(x, -1.0, 1.0)
        return (x + 1.0) * 0.5 * (mx - mn) + mn

    # ---------- 통계 (퍼센타일 미사용: 전체 min/max) ----------
    def _compute_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        all_q, all_dq = [], []
        for ep in self.episodes:
            z = np.load(ep["path"], allow_pickle=True)
            N = ep["N"]
            all_q.append(z["q"][:N].astype(np.float32))   # [N,7]
            all_dq.append(z["dq"][:N].astype(np.float32)) # [N,7]

        stats: Dict[str, Dict[str, np.ndarray]] = {}
        if len(all_q) == 0:
            # 비정상 케이스 방어
            stats["q"]  = {"min": np.zeros(7, dtype=np.float32), "max": np.ones(7, dtype=np.float32)}
            stats["dq"] = {"min": -np.ones(7, dtype=np.float32), "max": np.ones(7, dtype=np.float32)}
            return stats

        q_all  = np.concatenate(all_q,  0)  # [...,7]
        dq_all = np.concatenate(all_dq, 0)  # [...,7]

        # ★ 전체 min/max로 계산 (퍼센타일 제거)
        qmn = q_all.min(axis=0).astype(np.float32)
        qmx = q_all.max(axis=0).astype(np.float32)
        dqmn = dq_all.min(axis=0).astype(np.float32)
        dqmx = dq_all.max(axis=0).astype(np.float32)

        # ★ gripper 차원 오버라이드 (절대/음수 인덱스 모두 지원)
        J = q_all.shape[1]
        gidx = int(self.grip_index)
        if gidx < 0:
            gidx = J + gidx  # 예: -1 -> 마지막
        if 0 <= gidx < J:
            qmn[gidx] = 0.0
            qmx[gidx] = 0.8
        else:
            print(f"[WARN] grip_index={self.grip_index} out of range for J={J}; skip override.")

        # 수치 안정성: min==max 방어
        eps = 1e-6
        qmx = np.maximum(qmx, qmn + eps)
        dqmx = np.maximum(dqmx, dqmn + eps)

        stats["q"]  = {"min": qmn,  "max": qmx}
        stats["dq"] = {"min": dqmn, "max": dqmx}
        return stats

    # ---------- dataset ----------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ei, t = self.samples[idx]
        ep = self.episodes[ei]
        z = np.load(ep["path"], allow_pickle=True)
        N = ep["N"]

        s = t - (self.K - 1); e = t + 1

        # 시퀀스 자르기
        rgb_seq   = z["rgb"][s:e]                         # [K,H,W,3]
        depth_seq = z["depth"][s:e]                       # [K,H,W]
        q_seq     = z["q"][s:e].astype(np.float32)        # [K,7]
        dq_seq    = z["dq"][s:e].astype(np.float32)       # [K,7]

        # --- 관측 이미지 ---
        obs_rgb   = torch.from_numpy(np.stack([self._prep_rgb(rgb_seq[i])     for i in range(self.K)], 0)).float()  # [K,3,H,W]
        obs_depth = torch.from_numpy(np.stack([self._prep_depth(depth_seq[i]) for i in range(self.K)], 0)).float()  # [K,1,H,W]

        # --- 관측 low-dim (정규화: 전체 min/max) ---
        obs_q  = self._norm(torch.from_numpy(q_seq).float(),  "q")   # [K,7]
        obs_dq = self._norm(torch.from_numpy(dq_seq).float(), "dq")  # [K,7]

        # ----- 타깃 (미래 절대 q) -----
        q_future = z["q"][t+1 : t+1+self.T].astype(np.float32)       # [T,7]
        gt_joint = self._norm(torch.from_numpy(q_future).float(), "q")  # [T,7]

        sample = {
            "obs_rgb":   obs_rgb,      # [K,3,H,W]
            "obs_depth": obs_depth,    # [K,1,H,W]
            "obs_q":     obs_q,        # [K,7]
            "obs_dq":    obs_dq,       # [K,7] (모델에서 미사용이라도 유지 가능)
            "gt_joint":  gt_joint,     # [T,7]
            "ep_idx":    torch.tensor([ei], dtype=torch.long),
            "t_index":   torch.tensor([t],  dtype=torch.long),
        }
        return sample
