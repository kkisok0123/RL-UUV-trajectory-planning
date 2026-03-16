# test_env.py
from underwater_env import UnderwaterEnv
from gymnasium.utils.env_checker import check_env

env = UnderwaterEnv()
check_env(env)

obs, info = env.reset()
print("initial obs:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("reward =", reward, "terminated =", terminated, "truncated =", truncated)
    if terminated or truncated:
        obs, info = env.reset()