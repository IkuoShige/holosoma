from __future__ import annotations

from holosoma.agents.modules.flow_policy import FlowPolicy
from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCritic, PPOCriticEncoder


def setup_ppo_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict,
    module_config,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    raise ValueError(f"Invalid critic type: {module_type}")


def setup_flow_policy_module(
    obs_dim_dict,
    module_config,
    num_actions,
    device,
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
    return FlowPolicy(
        obs_dim_dict=obs_dim_dict,
        module_config=module_config,
        num_actions=num_actions,
        history_length=history_length,
        time_embed_dim=time_embed_dim,
        use_ada_ln=use_ada_ln,
        num_flow_steps=num_flow_steps,
        action_bound=action_bound,
        flow_param_mode=flow_param_mode,
        use_tanh=use_tanh,
        actor_scale=actor_scale,
        action_perturb_std=action_perturb_std,
        cfm_loss_t_inverse_cdf_beta=cfm_loss_t_inverse_cdf_beta,
        use_learned_time_embed=use_learned_time_embed,
        mlp_output_scale=mlp_output_scale,
        final_layer_weight_scale=final_layer_weight_scale,
    ).to(device)
