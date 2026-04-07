from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""

    betas: tuple[float, float] = (0.9, 0.999)
    """Beta parameters for Adam-family optimizers."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FPOModuleDictConfig:
    """Configuration for FPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration (used for FlowPolicy velocity field network)."""

    critic: ModuleConfig
    """Critic module configuration (PPOCritic reuse)."""


@dataclass(frozen=True)
class FPOConfig:
    """Configuration for FPO (Flow Policy Optimization) algorithm."""

    module_dict: FPOModuleDictConfig
    """FPO module configurations (actor, critic)."""

    # PPO-shared parameters
    num_learning_epochs: int = 3
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.05
    """PPO clipping parameter (FPO uses smaller clip than standard PPO)."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.0
    """Entropy coefficient (FPO does not use entropy regularization)."""

    actor_learning_rate: float = 1e-4
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-4
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 0.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations (PPO parent requires this)."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    # FPO-specific parameters
    time_embed_dim: int = 64
    """Dimension of sinusoidal timestep embedding."""

    use_ada_ln: bool = True
    """Whether to use adaptive layer normalization (adaLN) for time conditioning."""

    num_flow_steps: int = 64
    """Number of ODE Euler integration steps for action generation (paper: 64 for locomotion)."""

    num_mc_samples: int = 16
    """Number of Monte Carlo samples for CFM loss estimation (paper: 8-64, locomotion default 16)."""

    ratio_mode: str = "per_sample"
    """FPO ratio computation mode: 'per_sample' (FPO++) or 'legacy_avg'."""

    ratio_log_clip: float = 0.5
    """Clipping range for log ratio: clamp(log_r, -clip, clip)."""

    action_bound: float = 3.0
    """Scale factor for tanh bounding of ODE output: output = action_bound * tanh(raw)."""

    action_bound_warmup_iters: int = 0
    """Number of iterations to linearly ramp action_bound from init to full value. 0 = disabled."""

    action_bound_warmup_init_factor: float = 1.0 / 6.0
    """Initial action_bound = action_bound * this factor (default: 0.5 when action_bound=3.0)."""

    # Warm-start parameters
    warm_start_mode: str = "none"
    """Warm-start mode: 'none' (default), 'bc_teacher' (BC from PPO checkpoint)."""

    warm_start_checkpoint: str | None = None
    """Path to PPO checkpoint for warm-start. Required if warm_start_mode != 'none'."""

    warm_start_bc_steps: int = 200
    """Number of BC gradient steps during warm-start phase."""

    warm_start_load_critic: bool = False
    """Whether to load critic weights from PPO checkpoint (requires matching architecture)."""

    warm_start_teacher_module: ModuleConfig | None = None
    """PPO teacher actor module config. Required for 'bc_teacher' mode to reconstruct PPOActor."""

    flow_param_mode: str = "velocity"
    """Flow parameterization: 'velocity' (target=action-eps) or 'data' (target=action)."""

    cfm_reg_coef: float = 0.0
    """Coefficient for direct CFM loss regularization on actor loss (paper: 0, not used)."""

    cfm_reg_ema_beta: float = 0.99
    """EMA decay for CFM loss normalization."""

    onnx_export_num_flow_steps: int | None = None
    """Number of flow steps for ONNX export. None means use num_flow_steps."""

    trust_region_mode: str = "aspo"
    """Trust region mode: 'aspo' (Asymmetric SPO, paper default) or 'ppo_clip' (standard PPO clipping)."""

    cfm_loss_clip: float | None = None
    """Per-sample total CFM loss clamp upper bound (stage 1 δ). None disables."""

    cfm_loss_clip_adaptive: bool = False
    """Enable adaptive δ: auto-adjust cfm_loss_clip to maintain stage1_rate in target range."""

    cfm_loss_clip_quantile: float = 0.8
    """Quantile of CFM loss distribution to use as adaptive δ target."""

    cfm_loss_clip_ema_alpha: float = 0.1
    """EMA smoothing factor for adaptive δ updates (lower = smoother)."""

    cfm_loss_clip_min: float = 4.0
    """Lower bound for adaptive δ."""

    cfm_loss_clip_max: float = 30.0
    """Upper bound for adaptive δ."""

    cfm_loss_clip_update_interval: int = 10
    """Update adaptive δ every N iterations."""

    cfm_loss_dim_clip: float | None = None
    """Per-dimension squared-error clamp. Applied before sum/mean reduction. Usually not needed with mean reduction."""

    mc_chunk_size: int | None = 16
    """Chunk size along K (MC sample) dimension for compute_flow_loss. None disables chunking."""

    cfm_loss_reduction: str = "sum"
    """Reduction over action dim in CFM loss: 'sum', 'mean', or 'sqrt' (FPO++ variance-preserving)."""

    obs_normalization: bool = True
    """Whether to enable observation normalization (running mean/std)."""

    # ===== FPO++ additions =====

    use_tanh: bool = True
    """True: action_bound * tanh(x) (holosoma default). False: actor_scale * x (FPO++ linear)."""

    actor_scale: float = 1.0
    """Linear action scaling factor when use_tanh=False (FPO++ style)."""

    action_perturb_std: float = 0.0
    """Gaussian noise std added to actions during training (FPO++ entropy regularizer)."""

    storage_action_noise_std: float = 0.0
    """Gaussian noise std added to stored actions for implicit regularization."""

    cfm_loss_t_inverse_cdf_beta: float = 1.0
    """Beta(1, beta) for timestep sampling. 1.0 = uniform. >1 favors t near 0 (action refinement)."""

    use_learned_time_embed: bool = True
    """True: learned MLP projection (holosoma). False: raw sinusoidal (FPO++ official)."""

    mlp_output_scale: float = 1.0
    """Scaling factor applied to velocity field MLP output."""

    final_layer_weight_scale: float | None = None
    """Scaling for actor final layer init weights. None = default near-zero init."""

    use_ste_clamp: bool = False
    """Use straight-through estimator for log_ratio clamping (FPO++ style)."""

    cfm_diff_clamp_max: float = 10.0
    """Upper bound for one-sided CFM loss difference clamping (used with use_ste_clamp=True)."""

    cfm_loss_clamp_negative_advantages: bool = False
    """Clamp current CFM loss harder when advantage is negative (FPO++ stability)."""

    cfm_loss_clamp_negative_advantages_max: float = 20.0
    """Maximum CFM loss for negative advantage clamping."""

    advantage_clamp: tuple[float, float] | None = None
    """Symmetric advantage clamp (positive_max, negative_max). None = disabled."""

    knn_entropy_coef: float = 0.0
    """Coefficient for kNN entropy exploration bonus. 0 = disabled."""

    knn_entropy_k: int = 1
    """k for kNN entropy estimator."""

    use_clipped_value_loss: bool = True
    """Whether to clip value loss (PPO style). False = simple MSE (FPO++ default)."""

    ema_decay: float = 0.0
    """EMA decay for actor weights. 0 = disabled. FPO++ canonical: 0.95."""

    ema_warmup_steps: int = 500
    """PPO update steps before starting EMA. Prevents noisy early weights contaminating EMA."""

    # Divergence guard parameters
    divergence_guard_enabled: bool = True
    """Enable divergence monitoring and early stopping for FPO training."""

    divergence_warmup_iters: int = 50
    """Number of initial iterations to skip before divergence monitoring activates."""

    div_warn_cfm_ratio: float = 8.0
    """CFM loss ratio threshold for warning (cfm_loss / max(ema_cfm_loss, 1.0))."""

    div_stop_cfm_ratio: float = 20.0
    """CFM loss ratio threshold for stop condition."""

    div_hard_cfm_ratio: float = 50.0
    """CFM loss ratio threshold for immediate stop."""

    div_warn_stage2_rate: float = 0.70
    """stage2 clamp rate threshold for warning."""

    div_stop_stage2_rate: float = 0.85
    """stage2 clamp rate threshold for stop condition."""

    div_hard_stage2_rate: float = 0.95
    """stage2 clamp rate threshold for immediate stop."""

    div_warn_cov_adv_logr: float = -0.002
    """cov(adv, logr) threshold for warning (triggers when below this value)."""

    div_stop_cov_adv_logr: float = -0.005
    """cov(adv, logr) threshold for stop condition (triggers when below this value)."""

    div_stop_consecutive: int = 2
    """Number of consecutive iterations meeting stop criteria before stopping."""


@dataclass(frozen=True)
class FPOAlgoConfig:
    """Configuration for FPO algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FlashSACVendorConfig:
    """Configuration for the holosoma adapter around vendored FlashSAC.

    This dataclass is the holosoma-side mirror of
    :class:`holosoma._vendored.flash_rl.agents.flashSAC.agent.FlashSACConfig`,
    plus a few holosoma-specific fields (``actor_obs_keys`` /
    ``critic_obs_keys`` / ``num_learning_iterations`` / logging cadence) that
    let the adapter integrate with holosoma's training loop and the
    ``LeggedRobotLocomotionManager`` env contract.
    """

    # ---- holosoma loop cadence (NOT in upstream FlashSACConfig) ----------
    num_learning_iterations: int = 50_000
    """Total outer interaction steps to run (1 step ≈ 1 env collection per env)."""

    logging_interval: int = 100
    """Interval (interaction steps) at which to flush metrics to TensorBoard / W&B."""

    save_interval: int = 1000
    """Interval (interaction steps) at which to checkpoint."""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    """Observation manager group keys consumed by the FlashSAC actor."""

    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])
    """Observation manager group keys consumed by the FlashSAC critic (asymmetric obs)."""

    # ---- 1:1 mirror of FlashSACConfig (vendored) -------------------------
    seed: int = 0
    normalize_reward: bool = True
    normalized_G_max: float = 5.0
    asymmetric_observation: bool = False
    device_type: str = "cuda"

    buffer_max_length: int = 10_000_000
    buffer_min_length: int = 100_000
    buffer_device_type: str = "cuda"
    sample_batch_size: int = 2048

    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_rate: float = 1e-6
    learning_rate_decay_rate: float = 1.0

    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2

    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_bins: int = 101
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0
    critic_target_update_tau: float = 0.01

    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.15
    temp_target_entropy: float | None = None

    gamma: float = 0.99
    n_step: int = 3

    use_compile: bool = True
    compile_mode: str = "auto"
    use_amp: bool = True

    load_optimizer: bool = True
    load_reward_normalizer: bool = True

    updates_per_interaction_step: float = 2.0
    """Number of gradient updates per interaction step (can be fractional)."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration (currently unused for FlashSAC)."""


@dataclass(frozen=True)
class FlashSACVendorAlgoConfig:
    """Top-level wrapper for the FlashSAC vendor adapter (mirrors FastSACAlgoConfig)."""

    _target_: str
    """Target algorithm class (``holosoma.agents.flash_sac.flash_sac_agent.FlashSACAgent``)."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FlashSACVendorConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, FastSACConfig, FPOConfig, FlashSACVendorConfig]

AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig, FPOAlgoConfig, FlashSACVendorAlgoConfig]
