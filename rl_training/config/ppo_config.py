from __future__ import annotations


def build_ppo_config() -> dict:
    """Build PPO training defaults for the RL local planner."""
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "ent_coef": 0.003,
        "vf_coef": 0.5,
        "total_timesteps": 1_500_000,
        "n_envs": 4,
        "eval_freq": 20_000,
        "n_eval_episodes": 8,
        "seed": 7,
    }


PPO_CONFIG = build_ppo_config()
