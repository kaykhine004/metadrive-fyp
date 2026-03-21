#!/usr/bin/env python
"""
highway_ExpertTrained.py  —  Training script only.
Trains a PPO policy on the same highway map used in highway_expert.py.

Usage
-----
    python highway_ExpertTrained.py                      # 500 000 steps
    python highway_ExpertTrained.py --timesteps 1000000  # longer run

Output
------
    trained_expert/ppo_highway_expert.zip        final weights
    trained_expert/ppo_checkpoint_*.zip          every 50 k steps
    trained_expert/tb_logs/                      TensorBoard

After training is done, highway_expert.py will automatically pick up
the saved weights.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock

# ---------------------------------------------------------------------------
# Output paths  (same folder as this script)
# ---------------------------------------------------------------------------
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(_THIS_DIR, "trained_expert")
MODEL_PATH = os.path.join(TRAIN_DIR, "ppo_highway_expert")

# ---------------------------------------------------------------------------
# Environment config  — identical to highway_expert.py map, plus reward tuning
# ---------------------------------------------------------------------------
ENV_CONFIG = dict(
    map="CSrRSY$yRSCR",
    num_scenarios=10,
    start_seed=5,
    traffic_density=0.05,
    need_inverse_traffic=True,
    random_traffic=True,
    accident_prob=0.25,
    static_traffic_object=True,
    horizon=5000,
    crash_object_done=False,
    out_of_road_done=False,
    random_spawn_lane_index=False,
    # Reward tuning: keep vehicle on the road
    use_lateral_reward=True,       # reward for staying in lane
    out_of_road_penalty=15.0,      # strong penalty for going off-road
    vehicle_config={
        "show_navi_mark": False,
        "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
    },
)

# ---------------------------------------------------------------------------
# Reward wrapper
# ---------------------------------------------------------------------------
class HighwayRewardWrapper(gym.Wrapper):
    """
    Reward = env default (lane-keeping, out-of-road penalty, progress) + custom bonuses.
    The env default already includes: driving_reward, lateral_factor, -out_of_road_penalty,
    -crash_penalties, +success_reward. We add extra crash penalty and arrival bonus.
    """
    CRASH_EXTRA   = -15.0   # on top of env default -5
    ARRIVE_EXTRA  = 90.0    # on top of env default +10

    def __init__(self, env):
        super().__init__(env)
        self._prev_crash = False

    def reset(self, **kwargs):
        kwargs.pop("seed",    None)   # MetaDrive does not accept gymnasium seed kwarg
        kwargs.pop("options", None)
        result = self.env.reset(**kwargs)
        self._prev_crash = False
        return result

    def step(self, action):
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        crash_now = bool(
            info.get("crash_vehicle", False) or
            info.get("crash_object",  False)
        )
        reward = float(default_reward)  # includes out_of_road_penalty, lateral, etc.
        if crash_now and not self._prev_crash:
            reward += self.CRASH_EXTRA
        if info.get("arrive_dest", False):
            reward += self.ARRIVE_EXTRA

        self._prev_crash = crash_now
        return obs, reward, terminated, truncated, info


def make_train_env():
    env = SafeMetaDriveEnv({**ENV_CONFIG, "use_render": False})
    env = HighwayRewardWrapper(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(total_timesteps: int = 500_000):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    env = make_train_env()

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=TRAIN_DIR,
        name_prefix="ppo_checkpoint",
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(TRAIN_DIR, "tb_logs"),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("  PPO Training  -  Highway Expert  (no IDM)")
    print(f"  Map         : {ENV_CONFIG['map']}")
    print(f"  Timesteps   : {total_timesteps:,}")
    print(f"  Save path   : {TRAIN_DIR}")
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    env.close()
    print(f"\nTraining complete. Model saved -> {MODEL_PATH}.zip")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default 500 000)"
    )
    args = parser.parse_args()
    train(args.timesteps)
