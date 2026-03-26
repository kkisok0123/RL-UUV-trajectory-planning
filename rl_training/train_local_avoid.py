from __future__ import annotations

import atexit
import importlib.util
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_training.callbacks.collision_eval_callback import CollisionAwareEvalCallback
from rl_training.config.ppo_config import build_ppo_config
from rl_training.envs.fish_avoid_env import FishAvoidEnv


def make_env():
    return Monitor(FishAvoidEnv())


def _get_tensorboard_log_dir(log_dir: Path) -> str | None:
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return str(log_dir)


def _start_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen | None:
    if _get_tensorboard_log_dir(log_dir) is None:
        return None

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(port),
        "--reload_interval",
        "5",
    ]
    process = subprocess.Popen(command)
    atexit.register(process.terminate)

    url = f"http://127.0.0.1:{port}"
    time.sleep(1.5)
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass
    print(f"TensorBoard started at {url}")
    return process


def main() -> None:
    cfg = build_ppo_config()
    model_dir = ROOT / "artifacts" / "models"
    log_dir = ROOT / "artifacts" / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = _get_tensorboard_log_dir(log_dir)
    tensorboard_proc = _start_tensorboard(log_dir) if tensorboard_log is not None else None

    train_env = DummyVecEnv([make_env for _ in range(cfg["n_envs"])])
    eval_env = DummyVecEnv([make_env])

    eval_callback = CollisionAwareEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        cfg["policy"],
        train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=cfg["seed"],
    )
    model.learn(total_timesteps=cfg["total_timesteps"], callback=eval_callback)
    model.save(str(model_dir / "latest_model"))
    if tensorboard_proc is not None and tensorboard_proc.poll() is None:
        tensorboard_proc.terminate()


if __name__ == "__main__":
    main()
