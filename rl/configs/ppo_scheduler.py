def build_ppo_scheduler_config() -> dict:
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "total_timesteps": 50_000,
        "n_envs": 2,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
        "seed": 7,
    }


PPO_SCHEDULER_CONFIG = build_ppo_scheduler_config()
