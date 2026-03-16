# underwater_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dynamics_simple import dynamics_step


class UnderwaterEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = 0.1
        self.max_steps = 300
        self.step_count = 0

        # 状态: [x, y, z, vx, vy, vz]
        # 观测里再拼上目标相对位置、障碍物相对位置
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(12,), dtype=np.float32
        )

        # 动作: 三个方向的力
        self.action_space = spaces.Box(
            low=np.array([-20, -20, -20], dtype=np.float32),
            high=np.array([20, 20, 20], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.goal = None
        self.obstacle = None
        self.prev_dist = None

    def _get_obs(self):
        pos = self.state[:3]
        goal_rel = self.goal - pos
        obs_rel = self.obstacle - pos
        obs = np.concatenate([self.state, goal_rel, obs_rel]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.state = np.array([
            self.np_random.uniform(-1, 1),
            self.np_random.uniform(-1, 1),
            self.np_random.uniform(-1, 1),
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        self.goal = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        self.obstacle = np.array([
            self.np_random.uniform(1, 4),
            self.np_random.uniform(1, 4),
            self.np_random.uniform(1, 4)
        ], dtype=np.float32)

        self.prev_dist = np.linalg.norm(self.goal - self.state[:3])

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = dynamics_step(self.state, action, self.dt)

        pos = self.state[:3]
        dist_goal = np.linalg.norm(self.goal - pos)
        dist_obs = np.linalg.norm(self.obstacle - pos)

        reached_goal = dist_goal < 0.5
        collided = dist_obs < 0.5
        timeout = self.step_count >= self.max_steps

        # reward
        reward = 0.0
        reward += 10.0 * (self.prev_dist - dist_goal)      # 朝目标前进
        reward += -0.01 * np.sum(action ** 2)              # 控制能量惩罚

        if dist_obs < 1.5:
            reward += -5.0 * (1.5 - dist_obs)              # 靠近障碍物惩罚

        if reached_goal:
            reward += 200.0

        if collided:
            reward += -200.0

        self.prev_dist = dist_goal

        terminated = reached_goal or collided
        truncated = timeout

        obs = self._get_obs()
        info = {
            "dist_goal": dist_goal,
            "dist_obs": dist_obs
        }

        return obs, reward, terminated, truncated, info
