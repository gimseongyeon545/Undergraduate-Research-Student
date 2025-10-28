# `run_train_transformer_v2.py`
- `from diffusers.optimization import get_scheduler`
  - scheduler
  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py#L30
    ```
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    PIECEWISE_CONSTANT = "piecewise_constant"
    ```

</br>

- dataset
  - `CustomKinovaJointDataset` (`CustomDataset_joint2.py`)
    - dataset(npz)
      ```
      "stamp_ns",
      "rgb",
      "depth",
      "q",
      "dq", # 속도
      "ee_xyz",
      "ee_rpy",
      "joint_names",
      "camera_K",
      "camera_D",
      "camera_width",
      "camera_height"
      ```
    - rgb, depth, q(7), dq(7) seq 를 읽음
      - 해상도는 `resize_hw = (H,w)` 로 resize
      - depth: `d = np.clip(d, 0.0, self.max_depth_m) / self.max_depth_m if self.normalize_depth01 else d`
        - depth 를 0~`max_depth_m` 범위로 자른 후 0~1 사이로 정규화
      - rgb: 0~1 rescale, [H,W,3] -> [3,H,W]
      - npz 길이 N 을 최대 `max_episode_length` 만큼 잘라서 사용
        ```
        N = z["rgb"].shape[0]
        if self.max_len is not None and N > self.max_len:
            N = self.max_len
        ```
      - q, dq 는 [-1, 1] 로 정규화
    - episode: `z['rgb'][s:e]` 는 `[t-K+1, t]` (`self.K = obs_horizon`) 로 현재 시점 t 에서 가장 최근 obs_horizon 만큼만 현재 episode sequence 로 사용함
    - dimension
      - rgb: `[K,3,H,W]`
      - depth: `[K,1,H,W]`
      - q: `[K, 7]`
      - dq: `[K,7]`
      - gt_joint: `[T,7]`
        - `self.T = pred_horizon`

</br>

- model
  - Transformer (`transformer.py`)
    - `TransformerJointGripperNoDQ`
      - 

</br>

- hyperparameter
  - sequence
    ```
    # sequence
    obs_horizon = 8
    pred_horizon = 12
    action_horizon = 6
    ```
    - diffusion policy paper의 2.3 Diffudion for visuomotor policy learning 에서 "closed-loop action-sequence prediction"
    - $$T_o$$: observation horizon, $$T_p$$: predicton horizon, $T_a$: executed horizon
    - 12개 예측 중 6개 행동만 사용하고, obs 의 경우 최근 `obs_horizon` 개수만큼만 사용
