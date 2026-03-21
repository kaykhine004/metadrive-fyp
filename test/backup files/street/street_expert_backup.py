#!/usr/bin/env python
"""
Custom street expert run based on the user-defined route layout.

This version keeps the same procedural map and expert-driven ego vehicle, and
adds pedestrian crossing in the middle of the driven street:
- crossing site is fixed when route progress reaches the mid section, on the ego lane at that time
- pedestrians wait at the curb, then cross perpendicular to the lane when the ego is near
- 2–4 pedestrians per episode; graph-based fallback if lane data is missing
"""
import random

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.constants import Decoration, HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

# Tuning parameters
SPAWN_ROUTE_PROGRESS_RANGE = (0.38, 0.58)
SPAWN_APPROACH_DISTANCE_RANGE = (18.0, 30.0)
CURB_WAIT_STEPS_RANGE = (20, 55)
PEDESTRIAN_SPEED_RANGE = (0.55, 1.15)
PEDESTRIAN_COUNT_RANGE = (2, 4)
MAX_CROSSING_CANDIDATES = 8
MIN_SPAWN_SEPARATION = 2.5


class MidRouteCrossingController:
    """
    When the ego reaches the mid-route band, place a crossing on its current lane
    (middle of that street). Pedestrians wait at the curb, then cross when the
    vehicle is within range. Falls back to a short-lane graph site if needed.
    """

    def __init__(self):
        self.crossing_candidate = None
        self._awaiting_crossing_site = True
        self.pending_group = []
        self.active_group = []
        self.spawned_this_episode = False
        self.trigger_route_progress = 0.45
        self.trigger_approach_distance = 24.0
        self.last_spawn_step = -10_000

    def reset(self, env):
        self.crossing_candidate = None
        self._awaiting_crossing_site = True
        self.pending_group = []
        self.active_group = []
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
                    # Short lanes are usually near intersections where crosswalks appear.
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
                < MIN_SPAWN_SEPARATION ** 2
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
            group.append(
                {
                    "ped": ped,
                    "wait_pos": [float(wait_pos[0]), float(wait_pos[1])],
                    "target_pos": [float(lane.position(wait_long, -wait_lat)[0]), float(lane.position(wait_long, -wait_lat)[1])],
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
            if self._is_valid_object(env, ped) and env.episode_step >= item["wait_until_step"] and \
                    ego_distance_to_crossing <= self.trigger_approach_distance:
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

    def debug_status(self):
        return {
            "spawned": "Yes" if self.spawned_this_episode else "No",
            "waiting": len(self.pending_group),
            "crossing": len(self.active_group),
            "trigger_progress": f"{self.trigger_route_progress * 100:.0f}%",
            "trigger_distance": f"{self.trigger_approach_distance:.1f} m",
        }


if __name__ == "__main__":
    cfg = {
        "use_render": True,
        "manual_control": False,
        "agent_policy": ExpertPolicy,
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
        # Same map as before.
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

    env = None
    crossing_controller = MidRouteCrossingController()
    episode = 0
    try:
        env = MetaDriveEnv(cfg)
        env.reset()
        crossing_controller.reset(env)
        episode += 1

        print(HELP_MESSAGE)
        print("Street expert custom route loaded.")
        print("Traffic enabled with about 4-5 cars. Reference map files: route_map.svg and route_map.png.")
        print("Route layout: Zone 1 start -> Zone 2 junction -> Zone 3 bottleneck -> Zone 4 parking branch.")
        print("Crosswalks enabled. Pedestrians cross mid-route on the ego lane (2–4 per episode).")

        for step in range(1, 1_000_000):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            crossing_controller.update(env, step, info)

            route_completion = info.get("route_completion", 0.0) * 100.0
            env.render(
                text={
                    "Policy": "ExpertPolicy",
                    "route_completion": f"{route_completion:.1f}%",
                    "Episode": episode,
                    "Press F to speed up": "",
                }
            )

            if terminated or truncated:
                if info.get("arrive_dest", False):
                    print(f"Episode {episode}: arrived at destination.")
                else:
                    print(
                        f"Episode {episode}: ended early "
                        f"(route={route_completion:.1f}%, "
                        f"out_of_road={info.get('out_of_road', False)}, "
                        f"crash={info.get('crash', False)}, "
                        f"crash_human={info.get('crash_human', False)}, "
                        f"max_step={info.get('max_step', False)})."
                    )
                env.reset()
                crossing_controller.reset(env)
                episode += 1

    finally:
        if env is not None:
            env.close()
