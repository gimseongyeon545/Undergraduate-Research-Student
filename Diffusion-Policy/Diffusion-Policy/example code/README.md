# `run_train_transformer_v2.py`
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
- hyperparameter
  - sequence
    ```
    # sequence
    obs_horizon = 8
    pred_horizon = 12
    action_horizon = 6
    ```
    - diffusion policy paper의 2.3 Diffudion for visuomotor policy learning 에서 "closed-loop action-sequence prediction"
    - $$T_o$$: observation horizon, $$T_p$$: predicton horizon, $T_a$$: executed horizon
    - 12개 예측 중 6개 행동만 사용하고, obs 의 경우 최근 `obs_horizon` 개수만큼만 사용
