#!/usr/bin/env python
"""
Street scenario with ReplayEgoCarPolicy — replays the recorded NuScenes trajectory.

Based on the MetaDrive scenario env example. Compares reactive vs non-reactive traffic.
Supports top-down rendering and GIF generation.

Usage:
  python street_IDMTrajectory.py                    # Run with 3D view, no GIF
  python street_IDMTrajectory.py --top_down         # Top-down view
  python street_IDMTrajectory.py --gif              # Generate comparison GIF (reactive vs non-reactive)
  python street_IDMTrajectory.py --trajectory_idm   # Use TrajectoryIDMPolicy instead of Replay
"""
import argparse

import cv2
import numpy as np

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.constants import HELP_MESSAGE

nuscenes_data = AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)


def run_real_env(reactive, scenario_index=6, num_steps=150, policy_cls=ReplayEgoCarPolicy,
                 use_render=True, top_down=False, screen_record=False):
    """
    Run ScenarioEnv with NuScenes data.

    Args:
        reactive: If True, enable reactive_traffic
        scenario_index: Which scenario to load (default 6)
        num_steps: Number of steps to run
        policy_cls: ReplayEgoCarPolicy (replay) or TrajectoryIDMPolicy (IDM follow)
        use_render: Enable rendering
        top_down: Use top-down 2D view
        screen_record: Record frames for GIF

    Returns:
        frames: List of rendered frames (if screen_record), else None
    """
    cfg = {
        "reactive_traffic": reactive,
        "data_directory": nuscenes_data,
        "start_scenario_index": scenario_index,
        "num_scenarios": 1,
        "crash_vehicle_done": True,
        "log_level": 50,
        "use_render": use_render,
        "agent_policy": policy_cls,
        "no_traffic": policy_cls == TrajectoryIDMPolicy,
    }
    if policy_cls == TrajectoryIDMPolicy:
        TrajectoryIDMPolicy.NORMAL_SPEED = 30

    env = ScenarioEnv(cfg)
    try:
        o, _ = env.reset(seed=scenario_index)
        for i in range(1, num_steps):
            # [steering, throttle_brake]; 0,0 or 1,0 for auto policy (policy controls)
            o, r, tm, tc, info = env.step([0.0, 0.0])
            if top_down or screen_record:
                render_kw = dict(
                    mode="top_down",
                    window=use_render and not screen_record,
                    screen_record=screen_record,
                    camera_position=(0, 0),
                    screen_size=(500, 400),
                )
                env.render(**render_kw)
            elif use_render:
                env.render()
            if tm or tc:
                break
        frames = env.top_down_renderer.screen_frames if screen_record and hasattr(env, "top_down_renderer") and env.top_down_renderer is not None else []
    finally:
        env.close()
    return frames if screen_record else None


def main():
    parser = argparse.ArgumentParser(description="Street scenario with Replay or TrajectoryIDM policy")
    parser.add_argument("--top_down", action="store_true", help="Use 2D top-down view")
    parser.add_argument("--gif", action="store_true", help="Generate comparison GIF (reactive vs non-reactive)")
    parser.add_argument("--trajectory_idm", action="store_true", help="Use TrajectoryIDMPolicy instead of ReplayEgoCarPolicy")
    parser.add_argument("--scenario", type=int, default=6, help="Scenario index (default: 6)")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps (default: 150)")
    args = parser.parse_args()

    policy_cls = TrajectoryIDMPolicy if args.trajectory_idm else ReplayEgoCarPolicy
    policy_name = "TrajectoryIDMPolicy" if args.trajectory_idm else "ReplayEgoCarPolicy"

    if args.gif:
        # Generate comparison GIF: non-reactive vs reactive
        print("Running non-reactive traffic...")
        f_1 = run_real_env(False, scenario_index=args.scenario, num_steps=args.steps,
                           policy_cls=policy_cls, use_render=False, top_down=True, screen_record=True)
        print("Running reactive traffic...")
        f_2 = run_real_env(True, scenario_index=args.scenario, num_steps=args.steps,
                           policy_cls=policy_cls, use_render=False, top_down=True, screen_record=True)

        if f_1 and f_2:
            frames = []
            n = min(len(f_1), len(f_2))
            for i in range(n):
                img1 = np.asarray(f_1[i])
                img2 = np.asarray(f_2[i])
                combined = np.hstack([img1, img2])
                frames.append(combined)

            from metadrive.utils.doc_utils import generate_gif
            generate_gif(frames, gif_name="street_IDMTrajectory_comparison.gif", is_pygame_surface=False)
            print("Saved street_IDMTrajectory_comparison.gif")
        else:
            print("No frames captured. Ensure top_down_renderer is available.")
    else:
        # Interactive run
        print(HELP_MESSAGE)
        print(f"Policy: {policy_name} | Scenario: {args.scenario} | Steps: {args.steps}")
        frames = run_real_env(
            reactive=False,
            scenario_index=args.scenario,
            num_steps=args.steps,
            policy_cls=policy_cls,
            use_render=True,
            top_down=args.top_down,
            screen_record=False,
        )


if __name__ == "__main__":
    main()
