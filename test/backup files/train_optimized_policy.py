#!/usr/bin/env python
"""
Train an OptimizedPolicy (PPO) on the custom Roundabout + Intersection map.

Goals:
  - Minimize crashes (large penalty)
  - Minimize time to reach destination (time penalty + speed reward)
  - Stay on road (out-of-road penalty)

Output:
  Saves trained model to: test/Final analysis/Intersection analysis/optimized_policy_model.zip
"""
import os
import numpy as np

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pg_space import Parameter
from metadrive.component.road_network import Road
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils import Config

TARGET_DEST_NODE = Roundabout.node(3, 1, 3)

MODEL_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "optimized_policy_model")


# ---------------------------------------------------------------------------
# Custom map (same as evaluation)
# ---------------------------------------------------------------------------
class RoundaboutIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0

        first_block = FirstPGBlock(
            self.road_network, self.config[self.LANE_WIDTH], self.config[self.LANE_NUM],
            parent_node_path, physics_world, length=length,
        )
        self.blocks.append(first_block)

        InterSection.EXIT_PART_LENGTH = length
        intersection_block = InterSection(
            1, first_block.get_socket(index=0),
            self.road_network, random_seed=1, ignore_intersection_checking=False,
        )
        intersection_block.enable_u_turn(self.config["lane_num"] > 1)
        intersection_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(intersection_block)

        straight_block = Straight(
            2, intersection_block.get_socket(index=0),
            self.road_network, random_seed=1,
        )
        straight_block.construct_from_config(
            {Parameter.length: length}, parent_node_path, physics_world,
        )
        self.blocks.append(straight_block)

        Roundabout.EXIT_PART_LENGTH = length
        roundabout_block = Roundabout(
            3, straight_block.get_socket(index=0),
            self.road_network, random_seed=1, ignore_intersection_checking=False,
        )
        roundabout_block.construct_block(
            parent_node_path, physics_world,
            extra_config={"exit_radius": 10, "inner_radius": 30, "angle": 70},
        )
        self.blocks.append(roundabout_block)


class RoundaboutIntersectionMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(
                RoundaboutIntersectionMap,
                map_config=config["map_config"], random_seed=None,
            )
        else:
            assert len(self.spawned_objects) == 1
            _map = list(self.spawned_objects.values())[0]
        self.load_map(_map)


# ---------------------------------------------------------------------------
# Training environment — single-agent with IDM traffic
# ---------------------------------------------------------------------------
class IntersectionTrainEnv(MetaDriveEnv):
    @staticmethod
    def default_config() -> Config:
        cfg = MetaDriveEnv.default_config()
        cfg.update(dict(
            map_config=dict(exit_length=100, lane_num=2),
            traffic_density=0.15,
            horizon=1500,

            crash_vehicle_done=True,
            crash_object_done=True,
            out_of_road_done=True,

            success_reward=20.0,
            out_of_road_penalty=10.0,
            crash_vehicle_penalty=10.0,
            crash_object_penalty=5.0,
            driving_reward=1.0,
            speed_reward=0.3,
            use_lateral_reward=True,

            agent_configs={
                "default_agent": dict(
                    use_special_color=True,
                    spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
                    destination=TARGET_DEST_NODE,
                )
            },
            vehicle_config=dict(
                show_navi_mark=False,
                show_dest_mark=False,
                show_line_to_dest=False,
                lidar=dict(num_lasers=72, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
            ),

            use_render=False,
            log_level=50,
        ))
        return cfg

    def reward_function(self, vehicle_id):
        reward, step_info = super().reward_function(vehicle_id)
        reward -= 0.02
        return reward, step_info

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", RoundaboutIntersectionMapManager())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 500_000


def train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    print("=" * 60)
    print("  Training OptimizedPolicy (PPO)")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Model save path: {MODEL_PATH}.zip")
    print("=" * 60)

    env = IntersectionTrainEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=os.path.join(MODEL_SAVE_DIR, "tb_logs"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        name_prefix="optimized_policy",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}.zip")
    env.close()


if __name__ == "__main__":
    train()
