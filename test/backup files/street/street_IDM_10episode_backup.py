#!/usr/bin/env python
"""
Street IDM — 10 continuous episodes with logging (same structure as street_expert_10episode).

- IDMPolicy ego on map SCTSyCP; mid-route pedestrians (2–4 per episode).
- Per episode: crash count (edge-triggered, reset each episode).
- Timeout: >60 s *simulated* time with speed < STALL_SPEED_KMH (consecutive stall).
- Pass = YES if no timeout, else NO.
- pyautogui.press('f') once after startup when rendering (unlimited FPS).
- After N episodes (default 10), writes .xlsx to: .../Street analysis/IDM analysis/

Env (optional):
  METADRIVE_HEADLESS=1       — no window; skips pyautogui F.
  STREET_IDM_EPISODES=N     — default 10.

Requires: pip install openpyxl pyautogui
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime
from pathlib import Path

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.constants import Decoration, HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

OUTPUT_DIR = Path(
    r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main"
    r"\test\Final analysis\Street analysis\IDM analysis"
)

NUM_EPISODES = int(os.environ.get("STREET_IDM_EPISODES", "10"))
STALL_SPEED_KMH = 1.0
STALL_SECONDS = 60.0

SPAWN_ROUTE_PROGRESS_RANGE = (0.38, 0.58)
SPAWN_APPROACH_DISTANCE_RANGE = (18.0, 30.0)
CURB_WAIT_STEPS_RANGE = (20, 55)
PEDESTRIAN_SPEED_RANGE = (0.55, 1.15)
PEDESTRIAN_COUNT_RANGE = (2, 4)
MAX_CROSSING_CANDIDATES = 8
MIN_SPAWN_SEPARATION = 2.5


class MidRouteCrossingController:
    """Mid-route crossing on ego lane; 2–4 pedestrians per episode."""

    def __init__(self):
        self.crossing_candidate = None
        self._awaiting_crossing_site = True
        self.pending_group = []
        self.active_group = []
        self.spawned_ped_ids = []
        self.spawned_this_episode = False
        self.trigger_route_progress = 0.45
        self.trigger_approach_distance = 24.0
        self.last_spawn_step = -10_000

    def cleanup_spawned_pedestrians(self, env):
        """Required before env.reset(): remove dynamic pedestrians from physics world."""
        if not self.spawned_ped_ids:
            self.pending_group = []
            self.active_group = []
            return
        try:
            ids = list(dict.fromkeys(self.spawned_ped_ids))
            env.engine.clear_objects(ids, force_destroy=True, record=False)
        except Exception as exc:
            print(f"[WARN] Pedestrian cleanup failed: {exc}", flush=True)
        self.spawned_ped_ids = []
        self.pending_group = []
        self.active_group = []

    def reset(self, env):
        self.crossing_candidate = None
        self._awaiting_crossing_site = True
        self.pending_group = []
        self.active_group = []
        self.spawned_ped_ids = []
        self.spawned_this_episode = False
        self.trigger_route_progress = random.uniform(*SPAWN_ROUTE_PROGRESS_RANGE)
        self.trigger_approach_distance = random.uniform(*SPAWN_APPROACH_DISTANCE_RANGE)
        self.last_spawn_step = -10_000

    def update(self, env, step, info):
        ego = env.agent
        if ego is None:
            return

        route_progress = float(info.get("route_completion", 0.0))
        if self._awaiting_crossing_site:
            if route_progress >= self.trigger_route_progress:
                self.crossing_candidate = self._resolve_crossing_site(ego, env)
                self._awaiting_crossing_site = False

        if self.crossing_candidate is None:
            return

        crossing_pos = self.crossing_candidate["center"]
        dx = float(ego.position[0]) - crossing_pos[0]
        dy = float(ego.position[1]) - crossing_pos[1]
        ego_distance_to_crossing = (dx * dx + dy * dy) ** 0.5

        if not self.spawned_this_episode:
            can_spawn = (
                route_progress >= self.trigger_route_progress
                and ego_distance_to_crossing <= self.trigger_approach_distance
                and step - self.last_spawn_step > 120
            )
            if can_spawn:
                self._spawn_waiting_group(env, step)

        if self.pending_group:
            self._update_waiting_group(env)
        if self.active_group:
            self._update_crossing_group(env)

    @staticmethod
    def _candidate_from_ego_lane(ego):
        lane = getattr(ego, "lane", None)
        if lane is None:
            return None
        try:
            length = float(lane.length)
        except (TypeError, AttributeError, ValueError):
            return None
        if length < 6.0:
            return None
        try:
            long_p, _ = lane.local_coordinates(ego.position)
        except Exception:
            return None
        long_pos = float(long_p)
        long_pos = min(max(long_pos, 2.0), length - 2.0)
        center = lane.position(long_pos, 0.0)
        return {
            "lane": lane,
            "long_pos": long_pos,
            "center": (float(center[0]), float(center[1])),
            "lane_length": length,
        }

    def _select_crossing_candidate(self, env):
        road_network = env.current_map.road_network
        if not hasattr(road_network, "graph") or not road_network.graph:
            return None

        candidates = []
        for start_node, to_dict in road_network.graph.items():
            if start_node in (Decoration.start, Decoration.end):
                continue
            for end_node, lanes in to_dict.items():
                if end_node in (Decoration.start, Decoration.end):
                    continue
                for lane in lanes:
                    if 8.0 <= lane.length <= 35.0:
                        long_pos = min(max(4.0, lane.length * 0.35), lane.length - 2.0)
                        center = lane.position(long_pos, 0.0)
                        candidates.append(
                            {
                                "lane": lane,
                                "long_pos": long_pos,
                                "center": (float(center[0]), float(center[1])),
                                "lane_length": float(lane.length),
                            }
                        )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["lane_length"])
        usable = candidates[:MAX_CROSSING_CANDIDATES]
        return usable[len(usable) // 2]

    def _resolve_crossing_site(self, ego, env):
        c = self._candidate_from_ego_lane(ego)
        return c if c is not None else self._select_crossing_candidate(env)

    def _spawn_waiting_group(self, env, step):
        lane = self.crossing_candidate["lane"]
        long_pos = self.crossing_candidate["long_pos"]
        center = self.crossing_candidate["center"]

        spawn_side = random.choice([-1.0, 1.0])
        ped_count = random.randint(*PEDESTRIAN_COUNT_RANGE)
        group = []

        curb_lat = spawn_side * (lane.width * 0.72)
        cross_direction = -1.0 if spawn_side > 0 else 1.0

        for idx in range(ped_count):
            longitudinal_jitter = random.uniform(-1.2, 1.2)
            lateral_jitter = random.uniform(0.0, 0.25)
            wait_long = min(max(1.5, long_pos + longitudinal_jitter), lane.length - 1.5)
            wait_lat = curb_lat + spawn_side * lateral_jitter
            wait_pos = lane.position(wait_long, wait_lat)

            if any(
                ((wait_pos[0] - existing["wait_pos"][0]) ** 2 + (wait_pos[1] - existing["wait_pos"][1]) ** 2)
                < MIN_SPAWN_SEPARATION**2
                for existing in group
            ):
                continue

            ped = env.engine.spawn_object(
                Pedestrian,
                position=[float(wait_pos[0]), float(wait_pos[1])],
                heading_theta=lane.heading_theta_at(wait_long) + (1.57 if spawn_side > 0 else -1.57),
                random_seed=env.current_seed + step + idx + 1,
            )
            if ped is None:
                continue

            ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
            self.spawned_ped_ids.append(ped.id)
            group.append(
                {
                    "ped": ped,
                    "wait_pos": [float(wait_pos[0]), float(wait_pos[1])],
                    "target_pos": [
                        float(lane.position(wait_long, -wait_lat)[0]),
                        float(lane.position(wait_long, -wait_lat)[1]),
                    ],
                    "heading": lane.heading_theta_at(wait_long) + (1.57 if cross_direction > 0 else -1.57),
                    "cross_direction": cross_direction,
                    "speed": random.uniform(*PEDESTRIAN_SPEED_RANGE),
                    "wait_until_step": step + random.randint(*CURB_WAIT_STEPS_RANGE),
                }
            )

        if not group:
            return

        self.pending_group = group
        self.spawned_this_episode = True
        self.last_spawn_step = step
        self.crossing_candidate["center"] = center

    def _update_waiting_group(self, env):
        ego = env.agent
        if ego is None:
            return

        crossing_pos = self.crossing_candidate["center"]
        dx = float(ego.position[0]) - crossing_pos[0]
        dy = float(ego.position[1]) - crossing_pos[1]
        ego_distance_to_crossing = (dx * dx + dy * dy) ** 0.5

        remaining = []
        for item in self.pending_group:
            ped = item["ped"]
            ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
            if (
                self._is_valid_object(env, ped)
                and env.episode_step >= item["wait_until_step"]
                and ego_distance_to_crossing <= self.trigger_approach_distance
            ):
                ped.set_heading_theta(item["heading"])
                self.active_group.append(item)
            else:
                remaining.append(item)
        self.pending_group = remaining

    def _update_crossing_group(self, env):
        remaining = []
        for item in self.active_group:
            ped = item["ped"]
            if not self._is_valid_object(env, ped):
                continue

            pos = ped.position
            target = item["target_pos"]
            dx = target[0] - float(pos[0])
            dy = target[1] - float(pos[1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 0.6:
                ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
                continue

            ped.set_heading_theta(item["heading"])
            ped.set_velocity([dx, dy], item["speed"], in_local_frame=False)
            remaining.append(item)

        self.active_group = remaining

    @staticmethod
    def _is_valid_object(env, obj):
        try:
            return obj is not None and obj.id in env.engine.get_objects()
        except Exception:
            return False


def _press_fps_unlock():
    if pyautogui is None:
        print("[WARN] pyautogui not installed; skipping FPS toggle (pip install pyautogui).")
        return
    try:
        time.sleep(2.5)
        pyautogui.press("f")
        print("[INFO] Pressed F to toggle FPS (unlimited if supported).")
    except Exception as exc:
        print(f"[WARN] pyautogui.press('f') failed: {exc}")


def _crash_edge(prev_flags, info):
    v = bool(info.get("crash_vehicle", False))
    o = bool(info.get("crash_object", False))
    h = bool(info.get("crash_human", False))
    now = (v, o, h)
    edge = (not prev_flags[0] and v) or (not prev_flags[1] and o) or (not prev_flags[2] and h)
    return edge, now


def _sim_seconds_per_env_step(env: MetaDriveEnv) -> float:
    """Simulated seconds per env.step(); uses engine.global_config (same as physics)."""
    c = env.engine.global_config
    return float(c["physics_world_step_size"]) * int(c["decision_repeat"])


def _save_xlsx(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "Pass",
        "Timeout",
        "crash_count",
        "time_to_goal_s",
        "episode_sim_time_s",
        "speed_avg_kmh",
        "simulation_datetime",
    ]
    if Workbook is None:
        csv_path = path.with_suffix(".csv")
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fieldnames})
        print(f"[INFO] openpyxl missing; wrote CSV instead: {csv_path}")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "idm_10ep"
    ws.append(fieldnames)
    for r in rows:
        ws.append([r[k] for k in fieldnames])
    wb.save(path)
    print(f"[INFO] Saved: {path}")


def build_config():
    headless = os.environ.get("METADRIVE_HEADLESS", "").lower() in ("1", "true", "yes")
    return {
        "use_render": not headless,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "num_scenarios": 1,
        "start_seed": 0,
        "traffic_density": 0.06,
        "random_traffic": True,
        "need_inverse_traffic": True,
        "accident_prob": 0.0,
        "static_traffic_object": False,
        "horizon": 8000,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "crash_human_done": False,
        "out_of_road_done": False,
        "random_spawn_lane_index": False,
        "show_crosswalk": True,
        "map_config": {
            "type": "block_sequence",
            "config": "SCTSyCP",
            "lane_num": 1,
        },
        "vehicle_config": dict(
            show_navi_mark=True,
            spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
            lidar=dict(num_lasers=120, distance=50, num_others=0),
            lane_line_detector=dict(num_lasers=12, distance=50),
            side_detector=dict(num_lasers=12, distance=50),
        ),
        "interface_panel": ["dashboard"],
    }


def main():
    cfg = build_config()
    crossing_controller = MidRouteCrossingController()
    results: list[dict] = []
    sim_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    env = None
    try:
        env = MetaDriveEnv(cfg)
        env.reset(seed=0)
        crossing_controller.reset(env)

        print(HELP_MESSAGE)
        print(f"Running {NUM_EPISODES} episodes (IDMPolicy). Results -> {OUTPUT_DIR}")
        print("Mid-route: 2–4 pedestrians cross the ego street when you reach ~38–58% route progress.")
        if cfg.get("use_render"):
            _press_fps_unlock()

        for episode in range(1, NUM_EPISODES + 1):
            if episode > 1:
                crossing_controller.cleanup_spawned_pedestrians(env)
                env.reset(seed=0)
                crossing_controller.reset(env)

            sim_dt = _sim_seconds_per_env_step(env)
            horizon_cap = env.config.get("horizon")
            crash_count = 0
            prev_crash = (False, False, False)
            speeds: list[float] = []
            stall_accum_s = 0.0
            timed_out = False
            last_info: dict = {}
            episode_wall_t0 = time.perf_counter()
            num_env_steps = 0

            print(f"\n>>> Episode {episode}/{NUM_EPISODES} start\n", flush=True)

            while True:
                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                last_info = info
                num_env_steps += 1
                episode_sim_time_s = num_env_steps * sim_dt
                ep_step = int(env.episode_step)
                crossing_controller.update(env, ep_step, info)

                spd = float(getattr(env.agent, "speed_km_h", 0.0) or 0.0)
                speeds.append(spd)

                if spd < STALL_SPEED_KMH:
                    stall_accum_s += sim_dt
                else:
                    stall_accum_s = 0.0

                if stall_accum_s >= STALL_SECONDS:
                    timed_out = True
                    break

                edge, prev_crash = _crash_edge(prev_crash, info)
                if edge:
                    crash_count += 1

                route_completion = info.get("route_completion", 0.0) * 100.0
                if cfg.get("use_render"):
                    env.render(
                        text={
                            "Policy": "IDMPolicy",
                            "route_completion": f"{route_completion:.1f}%",
                            "Episode": f"{episode}/{NUM_EPISODES}",
                            "sim_time_s": f"{episode_sim_time_s:.2f}",
                            "stall_sim_s": f"{stall_accum_s:.1f}/{STALL_SECONDS}",
                            "Crashes": crash_count,
                            "Press F to speed up": "",
                        }
                    )

                max_step = bool(info.get("max_step", False))
                if terminated or truncated or max_step:
                    break
                if horizon_cap is not None and ep_step >= int(horizon_cap):
                    break

            episode_sim_time_s = num_env_steps * sim_dt
            if num_env_steps != int(env.episode_step):
                print(
                    f"[WARN] Step count mismatch: local={num_env_steps} engine.episode_step={env.episode_step}",
                    flush=True,
                )

            arrived = bool(last_info.get("arrive_dest", False))
            pass_str = "NO" if timed_out else "YES"
            timeout_str = "YES" if timed_out else "NO"
            wall_to_goal = time.perf_counter() - episode_wall_t0
            time_goal = round(wall_to_goal, 2) if arrived else "N/A"
            speed_avg = round(sum(speeds) / max(len(speeds), 1), 2)

            crossing_controller.cleanup_spawned_pedestrians(env)

            results.append(
                {
                    "episode": episode,
                    "Pass": pass_str,
                    "Timeout": timeout_str,
                    "crash_count": crash_count,
                    "time_to_goal_s": time_goal,
                    "episode_sim_time_s": round(episode_sim_time_s, 2),
                    "speed_avg_kmh": speed_avg,
                    "simulation_datetime": sim_started,
                }
            )
            print(
                f"Episode {episode}/{NUM_EPISODES} done | Pass={pass_str} Timeout={timeout_str} "
                f"crashes={crash_count} time_to_goal={time_goal} "
                f"sim_time={round(episode_sim_time_s, 2)}s (episode) speed_avg={speed_avg} km/h",
                flush=True,
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"idm_10episode_{ts}.xlsx"
        _save_xlsx(results, out_path)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
