from __future__ import annotations

import math

import torch
from holosoma.config_types.algo import ModuleConfig
from torch import Tensor, nn

from .modules import BaseModule


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a linear projection."""

    def __init__(self, embed_dim: int, frequency_embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.frequency_embed_dim = frequency_embed_dim
        # Pre-compute frequency table as a buffer so it follows .to(device) and
        # is treated as a constant by ONNX export (avoids torch.arange inside forward).
        half_dim = frequency_embed_dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer("freq", freq)  # [half_dim]

    def forward(self, t: Tensor) -> Tensor:
        # t: [B, 1] or [B] -> sinusoidal: [B, frequency_embed_dim] -> projection: [B, embed_dim]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] -> [B, 1]
        # t: [B, 1], self.freq: [half_dim] -> args: [B, half_dim]
        args = t.float() * self.freq.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embed_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class AdaLNBlock(nn.Module):
    """A linear layer followed by adaptive layer normalization (adaLN).

    adaLN modulates the post-normalization hidden state using scale/shift
    parameters derived from the timestep embedding.
    """

    def __init__(self, in_features: int, out_features: int, embed_dim: int, activation: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)
        # Project timestep embedding to scale and shift
        self.ada_proj = nn.Linear(embed_dim, out_features * 2)
        self.activation = activation

    def forward(self, x: Tensor, t_embed: Tensor) -> Tensor:
        h = self.linear(x)
        h = self.layer_norm(h)
        # Adaptive modulation
        scale_shift = self.ada_proj(t_embed)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return self.activation(h)


class SimpleTimestepEmbedder(nn.Module):
    """Raw sinusoidal timestep embedding (no learned projection).

    Used by FPO++ official implementation. Lighter than :class:`TimestepEmbedder`.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        freq = 2.0 ** torch.arange(half_dim, dtype=torch.float32)
        self.register_buffer("freq", freq)  # [half_dim]

    def forward(self, t: Tensor) -> Tensor:
        # t: [..., 1] -> [..., embed_dim]
        scaled_t = t * self.freq  # [..., half_dim]
        return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)


class VelocityFieldMLP(nn.Module):
    """Velocity field network for the flow model.

    Predicts the velocity field v_theta(x_t, t; obs) used in the ODE integration.
    Supports optional adaLN conditioning on timestep.

    When ``use_ada_ln=False`` and ``use_learned_time_embed=False``, the architecture
    matches the official FPO++ implementation (simple MLP with raw sinusoidal embed).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        time_embed_dim: int,
        use_ada_ln: bool,
        activation_name: str = "ELU",
        use_learned_time_embed: bool = True,
        mlp_output_scale: float = 1.0,
        final_layer_weight_scale: float | None = None,
    ):
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.mlp_output_scale = mlp_output_scale

        # Timestep embedder: learned projection (holosoma) or raw sinusoidal (FPO++)
        if use_learned_time_embed:
            self.time_embedder = TimestepEmbedder(time_embed_dim)
        else:
            self.time_embedder = SimpleTimestepEmbedder(time_embed_dim)

        # Input: obs + action (x_t) + time_embed (if not adaLN)
        if use_ada_ln:
            input_dim = obs_dim + action_dim
        else:
            input_dim = obs_dim + action_dim + time_embed_dim

        activation = getattr(nn, activation_name)()

        if use_ada_ln:
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(AdaLNBlock(prev_dim, h_dim, time_embed_dim, activation))
                prev_dim = h_dim
            self.hidden_layers = nn.ModuleList(layers)
        else:
            mlp_layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                mlp_layers.append(nn.Linear(prev_dim, h_dim))
                mlp_layers.append(activation)
                prev_dim = h_dim
            self.hidden_net = nn.Sequential(*mlp_layers)

        # Output layer: predicts velocity (same dim as action)
        self.output_layer = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, action_dim)

        # Weight initialization for final layer
        if final_layer_weight_scale is not None:
            with torch.no_grad():
                self.output_layer.weight.data *= final_layer_weight_scale
                if self.output_layer.bias is not None:
                    self.output_layer.bias.data *= final_layer_weight_scale
        else:
            # Default: near-zero init so untrained velocity field produces ~0
            nn.init.normal_(self.output_layer.weight, std=1e-3)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, obs: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict velocity field.

        Parameters
        ----------
        obs : Tensor
            Observations, shape [B, obs_dim] or [B*K, obs_dim]
        x_t : Tensor
            Interpolated state at time t, shape [B, action_dim] or [B*K, action_dim]
        t : Tensor
            Diffusion timestep in [0, 1], shape [B, 1] or [B*K, 1]

        Returns
        -------
        Tensor
            Predicted velocity, shape [B, action_dim] or [B*K, action_dim]
        """
        t_embed = self.time_embedder(t)

        if self.use_ada_ln:
            h = torch.cat([obs, x_t], dim=-1)
            for layer in self.hidden_layers:
                h = layer(h, t_embed)
        else:
            h = torch.cat([obs, x_t, t_embed], dim=-1)
            h = self.hidden_net(h)

        return self.mlp_output_scale * self.output_layer(h)


class FlowPolicy(nn.Module):
    """Flow-based policy using conditional flow matching (CFM).

    The policy generates actions by integrating a learned velocity field
    from noise (t=1) to clean actions (t=0) using Euler steps.

    Interpolation convention: x_t = t * eps + (1 - t) * action
        - t=0: clean action
        - t=1: pure noise (eps)
    """

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        num_actions: int,
        history_length: dict[str, int],
        time_embed_dim: int = 64,
        use_ada_ln: bool = True,
        num_flow_steps: int = 10,
        action_bound: float = 3.0,
        flow_param_mode: str = "velocity",
        # FPO++ additions
        use_tanh: bool = True,
        actor_scale: float = 1.0,
        action_perturb_std: float = 0.0,
        cfm_loss_t_inverse_cdf_beta: float = 1.0,
        use_learned_time_embed: bool = True,
        mlp_output_scale: float = 1.0,
        final_layer_weight_scale: float | None = None,
    ):
        super().__init__()
        if flow_param_mode not in ("velocity", "data"):
            msg = f"flow_param_mode must be 'velocity' or 'data', got '{flow_param_mode}'"
            raise ValueError(msg)
        self.num_actions = num_actions
        self.num_flow_steps = num_flow_steps
        self.action_bound = action_bound
        self.flow_param_mode = flow_param_mode
        # FPO++ parameters
        self.use_tanh = use_tanh
        self.actor_scale = actor_scale
        self.action_perturb_std = action_perturb_std
        self.cfm_loss_t_inverse_cdf_beta = cfm_loss_t_inverse_cdf_beta

        # Process module config to resolve action dim
        module_config = self._process_module_config(module_config, num_actions)

        # Compute observation dimension using BaseModule's pattern
        self._obs_module = BaseModule(obs_dim_dict, module_config, history_length)
        obs_dim = self._obs_module.input_dim
        # Remove the _obs_module's network since we only needed input_dim calculation
        del self._obs_module

        self.obs_dim = obs_dim
        hidden_dims = module_config.layer_config.hidden_dims
        activation_name = module_config.layer_config.activation

        self.velocity_field = VelocityFieldMLP(
            obs_dim=obs_dim,
            action_dim=num_actions,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
            use_ada_ln=use_ada_ln,
            activation_name=activation_name,
            use_learned_time_embed=use_learned_time_embed,
            mlp_output_scale=mlp_output_scale,
            final_layer_weight_scale=final_layer_weight_scale,
        )

    def _process_module_config(self, module_config: ModuleConfig, num_actions: int) -> ModuleConfig:
        for idx, output_dim in enumerate(module_config.output_dim):
            if output_dim == "robot_action_dim":
                module_config.output_dim[idx] = num_actions
        return module_config

    def reset(self, dones=None):
        pass

    def interpolate_xt(self, action: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """Compute interpolated state x_t = t * eps + (1 - t) * action.

        Parameters
        ----------
        action : Tensor
            Clean actions, shape [..., A]
        eps : Tensor
            Noise samples, shape [..., A]
        t : Tensor
            Timestep in [0, 1], shape [..., 1]

        Returns
        -------
        Tensor
            Interpolated state, shape [..., A]
        """
        return t * eps + (1.0 - t) * action

    def sample_mc_noise_and_time(self, batch_size: int, num_mc_samples: int, device: torch.device):
        """Sample noise and time for Monte Carlo CFM loss estimation.

        When ``cfm_loss_t_inverse_cdf_beta != 1.0``, timesteps are sampled from
        a Beta(1, β) distribution via inverse CDF transform and clamped to
        [0.005, 0.995] to avoid boundary instabilities. With β=1.0 (default)
        this reduces to uniform sampling.

        Returns
        -------
        eps : Tensor
            Noise samples, shape [B, K, A]
        t : Tensor
            Timestep samples, shape [B, K, 1]
        """
        eps = torch.randn(batch_size, num_mc_samples, self.num_actions, device=device)
        uniform_t = torch.rand(batch_size, num_mc_samples, 1, device=device)

        beta = self.cfm_loss_t_inverse_cdf_beta
        if beta == 1.0:
            t = uniform_t
        else:
            # Inverse CDF of Beta(1, beta): F^{-1}(u) = 1 - (1-u)^(1/beta)
            # Scale to [0.005, 0.995] to avoid boundary instabilities
            t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))

        return eps, t

    def compute_flow_loss(
        self,
        obs: Tensor,
        action: Tensor,
        eps: Tensor,
        t: Tensor,
        reduction: str = "sum",
        dim_clip: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-sample CFM loss: ||v_theta(x_t, t; obs) - target||^2.

        Parameters
        ----------
        obs : Tensor
            Observations, shape [B, obs_dim]
        action : Tensor
            Clean actions, shape [B, A]
        eps : Tensor
            Noise samples, shape [B, K, A]
        t : Tensor
            Timestep samples, shape [B, K, 1]
        reduction : str
            Reduction over action dim: 'sum', 'mean', or 'sqrt' (variance-preserving).
        dim_clip : float | None
            Per-dimension squared-error clamp (paper Appendix C.2, δ). Applied before reduction.

        Returns
        -------
        loss : Tensor
            Per-sample CFM loss, shape [B, K, 1]
        x1_pred : Tensor
            Predicted noise (x1), shape [B, K, A]
        x0_pred : Tensor
            Predicted clean action (x0), shape [B, K, A]
        """
        b, k, a = eps.shape

        # Expand obs and action for MC samples: [B, obs_dim] -> [B*K, obs_dim]
        obs_expanded = obs.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        action_expanded = action.unsqueeze(1).expand(-1, k, -1).reshape(b * k, a)
        eps_flat = eps.reshape(b * k, a)
        t_flat = t.reshape(b * k, 1)

        # When use_tanh=False (FPO++ linear scaling), learn flow in scaled space
        # During inference we output actor_scale * x_t, so train in same space
        if not self.use_tanh and self.actor_scale != 1.0:
            action_expanded = action_expanded / self.actor_scale

        # Interpolate: x_t = t * eps + (1 - t) * action
        x_t = self.interpolate_xt(action_expanded, eps_flat, t_flat)

        # Target: velocity param -> (action - eps), data param -> action
        if self.flow_param_mode == "velocity":
            target_velocity = action_expanded - eps_flat
        else:
            target_velocity = action_expanded

        # Predicted velocity
        predicted_velocity = self.velocity_field(obs_expanded, x_t, t_flat)

        # Compute x0_pred and x1_pred from velocity (for kNN entropy & KL tracking)
        # x0_pred = x_t - t * velocity, x1_pred = x0_pred + velocity
        x0_pred_flat = x_t - t_flat * predicted_velocity
        x1_pred_flat = x0_pred_flat + predicted_velocity

        # Per-sample loss: [B*K, A] -> [B*K, 1] -> [B, K, 1]
        sq_diff = (predicted_velocity - target_velocity) ** 2
        if dim_clip is not None:
            sq_diff = sq_diff.clamp(max=dim_clip)
        if reduction == "sum":
            loss = sq_diff.sum(dim=-1, keepdim=True)
        elif reduction == "sqrt":
            # Variance-preserving: divide by sqrt(action_dim)
            loss = sq_diff.sum(dim=-1, keepdim=True) / (a**0.5)
        else:
            loss = sq_diff.mean(dim=-1, keepdim=True)

        return (
            loss.reshape(b, k, 1),
            x1_pred_flat.reshape(b, k, a),
            x0_pred_flat.reshape(b, k, a),
        )

    def _scale_actions(self, x: Tensor) -> Tensor:
        """Apply output scaling to raw ODE output.

        When ``use_tanh=True`` (holosoma default): action_bound * tanh(x).
        When ``use_tanh=False`` (FPO++ style): actor_scale * x (linear).
        """
        if self.use_tanh:
            return self.action_bound * torch.tanh(x)
        return self.actor_scale * x

    def act(self, obs_dict: dict[str, Tensor], num_flow_steps: int | None = None) -> Tensor:
        """Generate actions via ODE Euler integration (training mode with noise).

        Integrates from t=1 (noise) to t=0 (clean action) using Euler steps.

        Parameters
        ----------
        obs_dict : dict
            Dictionary with 'actor_obs' key, shape [B, obs_dim]
        num_flow_steps : int | None
            Number of integration steps. Defaults to self.num_flow_steps.

        Returns
        -------
        Tensor
            Generated actions, shape [B, A]
        """
        obs = obs_dict["actor_obs"]
        k = num_flow_steps or self.num_flow_steps
        dt = 1.0 / k

        # Start from pure noise at t=1
        x = torch.randn(obs.shape[0], self.num_actions, device=obs.device)

        for j in range(k, 0, -1):
            t_val = j / k  # t >= 1/K > 0, avoids t=0
            t_tensor = obs.new_full((obs.shape[0], 1), t_val)
            output = self.velocity_field(obs, x, t_tensor)
            if self.flow_param_mode == "velocity":
                x = x + dt * output
            else:
                t_prev = (j - 1) / k
                x = x * (t_prev / t_val) + output * (dt / t_val)

        actions = self._scale_actions(x)

        # Action perturbation (FPO++ entropy regularizer)
        if self.training and self.action_perturb_std > 0:
            actions = actions + self.action_perturb_std * torch.randn_like(actions)

        return actions

    def act_inference(self, obs_dict: dict[str, Tensor], num_flow_steps: int | None = None) -> Tensor:
        """Generate actions via ODE Euler integration (deterministic inference).

        Starts from zeros instead of random noise, making the output fully
        deterministic for a given observation. Used for evaluation, symmetry
        loss computation, and ONNX export.

        Parameters
        ----------
        obs_dict : dict
            Dictionary with 'actor_obs' key, shape [B, obs_dim]
        num_flow_steps : int | None
            Number of integration steps. Defaults to self.num_flow_steps.

        Returns
        -------
        Tensor
            Generated actions, shape [B, A]
        """
        obs = obs_dict["actor_obs"]
        k = num_flow_steps or self.num_flow_steps
        dt = 1.0 / k

        # Start from zeros (deterministic) instead of randn (stochastic)
        x = obs.new_zeros(obs.shape[0], self.num_actions)

        for j in range(k, 0, -1):
            t_val = j / k
            t_tensor = obs.new_full((obs.shape[0], 1), t_val)
            output = self.velocity_field(obs, x, t_tensor)
            if self.flow_param_mode == "velocity":
                x = x + dt * output
            else:
                t_prev = (j - 1) / k
                x = x * (t_prev / t_val) + output * (dt / t_val)

        return self._scale_actions(x)
