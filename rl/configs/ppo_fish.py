def build_ppo_config() -> dict:
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "total_timesteps": 20_000,
        "n_envs": 4,
        "eval_freq": 5_000,
        "n_eval_episodes": 5,
        "seed": 7,
    }


PPO_CONFIG = build_ppo_config()
