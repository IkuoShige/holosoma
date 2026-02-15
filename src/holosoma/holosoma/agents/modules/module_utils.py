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
):
    return FlowPolicy(
        obs_dim_dict=obs_dim_dict,
        module_config=module_config,
        num_actions=num_actions,
        history_length=history_length,
        time_embed_dim=time_embed_dim,
        use_ada_ln=use_ada_ln,
        num_flow_steps=num_flow_steps,
    ).to(device)
