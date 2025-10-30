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
  - 1. Transformer (`transformer.py`)
    - `TransformerJointGripperNoDQ`
      ```
      입력 (정규화됨)
      ┌─────────────────────────────────────────────────────────────────────────────────┐
      │ obs_rgb   : [B, K, 3, H, W]   │ obs_depth : [B, K, 1, H, W] │ obs_q : [B, K, 7] │
      
         -> [B(Batch), K(seq), channel, Height, Width]
      └─────────────────────────────────────────────────────────────────────────────────┘
              │                │                         │
              ▼                ▼                         ▼
         ConvEncoder(3→img)  ConvEncoder(1→img)    LowDimEncoder(7→q_feat)
         [self.rgb_enc]      [self.depth_enc]      [self.q_enc]
          → feat_rgb          → feat_depth          → feat_q
          [B, K, img]         [B, K, img]           [B, K, q_feat]
              │───────────────────┬───────────────────────────│
                                  ▼
                       concat([feat_rgb, feat_depth, feat_q])   # [B, K, img+img+q_feat]
                                  │
                                  ▼
                            Fuse MLP (Linear→ReLU→Linear)
                                  │
                                  ▼
                            tokens_enc : [B, K, D]    <- 그림의 `Input Embeddings`
                                  │
                           + PositionalEncoding
                                  │
                                  ▼
                      TransformerEncoder (L_enc layers)
                                  │
                           memory : [B, K, D]  ← 시간 K의 컨텍스트
                                  │
                                  │            ┌────────────────────────────────────┐
                                  │            │ Learned future queries (파라미터) │
                                  │            │  [T_max, D] → slice [:T] → [B,T,D]│
                                  │            └────────────────────────────────────┘
                                  │                          │
                                  │                     + PosEnc
                                  │                          │
                                  └──────────────►  TransformerDecoder (L_dec)  ◄──────┐
                                                   (tgt=[B,T,D], memory=[B,K,D])       │
                                                                                       │
                                               h_dec : [B, T, D]                       │
                                                      │                                │
                                                      ▼                                │
                                              Head MLP (Linear→ReLU→Linear)            │
                                                      │                                │
                                                      ▼                                │
                                         pred_joint : [B, T, 7] → (tanh) → [-1,1]      │
       ```
      - `d_model`: 한 토큰의 임베딩 길이
      - `LowDimEncoder` 에서 하나의 시점 t 에서의 q(7) (=`in_dim`) 을 받아서 `d_model` 만큼의 차원으로 embedding
      - TransformerEncoder & TransformerDecoder
        > <img width="305" height="434" alt="image" src="https://github.com/user-attachments/assets/1151636b-a84b-41d5-a734-573af4a4ea6a" />
          - `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)`
            - `d_model`: Input Embedding layer 에서의 출력 크기
            - `nhead`: Multi-Head Attention 개수
              - 각 head 별 각 Q,K,V 가중치 존재
              - input `[token 길이 L, d_model]` -> head 당 Q,K,V 가중치 dim `[d_model, d_model/nhead]` -> output Q,K,V dim [L, d_model/nhaed] 
              - 하나의 token 에 head 별 가중치를 통해 여러 관점
            - `dim_feedforward`: FFN 내부 dim
              - FFN: $$W_2 \sigma(W_1x + b_1) + b_2$$ (transformer 의 ffn 은 2 layers)
          - `nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)`
            - `num_layers`: 그림의 'N'

  - 2. EMAModel
    - https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L398
    - model 의 parameter 를 EMA(지수 이동 평균) 를 이용해여 학습 안정성
    - $$\theta_t^{EMA} ← decay \cdot \theta_{t-1}^{EMA} + (1-decay) \cdot \theta_t$$
    - args
      ```
      parameters: Iterable[torch.nn.Parameter],
      decay: float = 0.9999, # 1에 가까울수록 느리게 갱신
      min_decay: float = 0.0, # use_ema_warmup=True 일 때, decay 하한값
      update_after_step: int = 0, # 이 값을 지나기 전에는 EMA update 하지 않음
      use_ema_warmup: bool = False, # decay 작게 시작해서 점점 올리는 스케줄 사용 여부
      inv_gamma: Union[float, int] = 1.0, # use_ema_warmup=True 일 때, 언제부터 올라갈지 결정 (클수록 느리게)
      power: Union[float, int] = 2 / 3, # use_ema_warmup=True 일 때, 얼마나 올라갈지 결정 (쿨수록 더 빨리)
      foreach: bool = False,
      model_cls: Optional[Any] = None,
      model_config: Dict[str, Any] = None
      ```
    - `optimize.step()` 다음에 `ema.step(model.parameters())`

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

</br>

# `run_test_transformer_v2.py`
- node name: 'policy_runner'
- `declare_parameter` 으로 선언된 것 -> `--ros-args -p`
- `_load_model`: training 시 저장한 최적 parameters load 해서 TransformerJointGripperNoDQ 추론 모델 생성
- [1] subscriber
  |sub|msg type|topic|callback|
  |:--:|:--:|:--:|:--:|
  |sub_js|JointState|/joint_states|self.`cb_joint_states`|
  |sub_rgb|Image|/camera/color/image_raw|self.`cb_rgb`|
  |sub_dep|Image|/camera/depth/image_rect_raw|self.`cb_depth`|
  - `cb_joint_states`
    - `self.sub_js = self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)`
      - 해당 topic 을 publish 하는건 bringup 명령어
    - JointState type 의 msg 를 받아서 q(7) numpy array 에 현재 joint state 담기 (bringup 이 띄워져 있는 동안 sub callback 계속 자동 호출)
      - `q[j] = float(msg.position[name_to_idx[n]])`
      - `q[-1] = float(msg.position[name_to_idx[self.gripper_joint_name]])`
    - `self.q_buf.append(q)`
      - 최근 `obs_horizon` 만큼 뽑기 위해서 `tick` 함수에서 `q_seq = np.stack(list(self.q_buf)[-self.K:], axis=0)` (`self.K`: obs_horizon)
  - `cb_rgb`
    - `self.sub_rgb  = self.create_subscription(Image, self.color_topic, self.cb_rgb, 10)`
    - resize
    - `self.rgb_buf.append(arr)`
    - tick 함수: `rgb_seq = np.stack(list(self.rgb_buf)[-self.K:], axis=0)`
  - `cb_depth`
    - `self.sub_dep  = self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)`
    - resize
    - `self.dep_buf.append(arr.astype(np.float32))`
    - tick 함수: `dep_seq = np.stack(list(self.dep_buf)[-self.K:], axis=0)`
- [2] tick 함수
  - 1) buf 에서 가장 최근 obs_horizon (self.K) 만큼 뽑기
    ```
    rgb_seq = np.stack(list(self.rgb_buf)[-self.K:], axis=0)    
    dep_seq = np.stack(list(self.dep_buf)[-self.K:], axis=0)    
    q_seq   = np.stack(list(self.q_buf)[-self.K:], axis=0)      
    ```
  - 2) 정규화
    - rgb seq: 정규화(/255) + tensor 화 (permute)
    - depth seq: 정규화
    - q seq: [-1, 1] 정규화
  - 3) 추론 (TransformerJointGripperNoDQ)
  - 4) `TransformerJointGripperNoDQ` 반환 값인 예측할 길이 T_p 만큼의 [T, 7] 을 실제 joint 로 denorm -> 가장 처음 t 에서의 7개 joint 만 (T_a = 1)  `q_cmd` -> `_publish` 함수로 publish
- [3] _publish 함수
  - 1) `q_cmd` np array 를 인자로 받아서
  - 2) JointTrajectory: `Header`, `joint_names`, `points` field
    - JointTrajectoryPoint 를 `points` 필드에 넣기: `msg.points.append(pt)`
      - positions field: `q_cmd` (`pt.positions = list(q7.astype(float))`)
  - 3) 완성된 JointTrajectory msg 를 publish (`self.pub_traj.publish(msg)`)
       |pub|msg type|topic|
       |:--:|:--:|:--:|
       |pub_traj|JointTrajectory|/joint_trajectory_controller/joint_trajectory|
       |pub_grip|GripperCommand|/gripper_controller/command|
