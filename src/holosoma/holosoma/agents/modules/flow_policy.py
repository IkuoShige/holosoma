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


class VelocityFieldMLP(nn.Module):
    """Velocity field network for the flow model.

    Predicts the velocity field v_theta(x_t, t; obs) used in the ODE integration.
    Supports optional adaLN conditioning on timestep.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        time_embed_dim: int,
        use_ada_ln: bool,
        activation_name: str = "ELU",
    ):
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.time_embedder = TimestepEmbedder(time_embed_dim)

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
        # Zero-initialize so untrained velocity field produces ~0, keeping
        # initial ODE output close to the starting noise (small after tanh).
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

        return self.output_layer(h)


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
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_flow_steps = num_flow_steps
        self.action_bound = action_bound

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

        Returns
        -------
        eps : Tensor
            Noise samples, shape [B, K, A]
        t : Tensor
            Timestep samples, shape [B, K, 1]
        """
        eps = torch.randn(batch_size, num_mc_samples, self.num_actions, device=device)
        t = torch.rand(batch_size, num_mc_samples, 1, device=device)
        return eps, t

    def compute_flow_loss(self, obs: Tensor, action: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """Compute per-sample CFM loss: ||v_theta(x_t, t; obs) - (action - eps)||^2.

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

        Returns
        -------
        Tensor
            Per-sample CFM loss, shape [B, K, 1]
        """
        b, k, a = eps.shape

        # Expand obs and action for MC samples: [B, obs_dim] -> [B*K, obs_dim]
        obs_expanded = obs.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        action_expanded = action.unsqueeze(1).expand(-1, k, -1).reshape(b * k, a)
        eps_flat = eps.reshape(b * k, a)
        t_flat = t.reshape(b * k, 1)

        # Interpolate
        x_t = self.interpolate_xt(action_expanded, eps_flat, t_flat)

        # Target velocity: action - eps (direction from noise to clean)
        target_velocity = action_expanded - eps_flat

        # Predicted velocity
        predicted_velocity = self.velocity_field(obs_expanded, x_t, t_flat)

        # Per-sample MSE loss: [B*K, A] -> [B*K, 1] -> [B, K, 1]
        loss = ((predicted_velocity - target_velocity) ** 2).mean(dim=-1, keepdim=True)
        return loss.reshape(b, k, 1)

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
        t_val = 1.0

        for _ in range(k):
            # Use obs.new_full to inherit device/dtype from obs (ONNX-safe)
            t_tensor = obs.new_full((obs.shape[0], 1), t_val)
            velocity = self.velocity_field(obs, x, t_tensor)
            # v_theta approximates (action - eps), the CFM target velocity.
            # x_t = t*eps + (1-t)*action => dx/dt = eps - action = -v_theta
            # Stepping from t to t-dt: x_{t-dt} = x_t + (-dt)*(-v_theta) = x_t + dt*v_theta
            x = x + dt * velocity
            t_val -= dt

        # Bound output via tanh to prevent unbounded actions
        return self.action_bound * torch.tanh(x)

    def act_inference(self, obs_dict: dict[str, Tensor], num_flow_steps: int | None = None) -> Tensor:
        """Generate actions via ODE Euler integration (inference mode, no grad).

        Same as act() but wrapped with torch.no_grad().

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
        return self.act(obs_dict, num_flow_steps=num_flow_steps)
