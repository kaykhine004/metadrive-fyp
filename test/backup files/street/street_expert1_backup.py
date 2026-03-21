#!/usr/bin/env python
"""
Custom street expert run based on the user-defined route layout.

This version keeps the same procedural map and expert-driven ego vehicle, and
adds a reliable pedestrian crossing event in the middle of the route:
- crossing site is chosen around the mid-route section
- pedestrians do NOT appear at the start
- pedestrians spawn at the curb when ego approaches the crosswalk
- they wait briefly, then cross the road
- designed to be testable and visibly trigger during the run
"""

import math
import random

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.constants import HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy


# =========================================================
# Tunable parameters
# =========================================================
CROSSWALK_PROGRESS = 0.45
SPAWN_DISTANCE = 30.0
START_CROSS_DISTANCE = 22.0
PED_COUNT = (1, 1)
CURB_WAIT_STEPS = (8, 15)        # wait at curb before crossing
PED_SPEED = (1.0, 1.35)            # walking speed
PED_COUNT = (1, 2)                 # number of pedestrians
CROSSWALK_LONG_JITTER = 0.8        # slight variation along the crosswalk
CROSSWALK_LAT_FACTOR = 0.95        # curb position relative to lane width


class NaturalCrosswalkTestController:
    """
    Reliable and testable mid-route pedestrian crossing event.

    Behavior:
    - A crossing site is selected from the ego lane around mid-route.
    - When the ego gets close enough, 1-2 pedestrians spawn at the curb.
    - They wait briefly, then cross to the opposite curb.
    - This avoids spawning at the beginning and guarantees crossing movement.
    """

    def __init__(self):
        self.crossing_site = None
        self.event_spawned = False
        self.event_finished = False
        self.waiting = []
        self.crossing = []

    def reset(self, env):
        self.crossing_site = None
        self.event_spawned = False
        self.event_finished = False
        self.waiting = []
        self.crossing = []

    def update(self, env, step, info):
        ego = env.agent
        if ego is None or self.event_finished:
            return

        route_progress = float(info.get("route_completion", 0.0))

        # 1) Lock one crossing site around the middle of the route
        if self.crossing_site is None and route_progress >= CROSSWALK_PROGRESS:
            self.crossing_site = self._make_crossing_site_from_ego_lane(ego)

        if self.crossing_site is None:
            return

        cross_center = self.crossing_site["center"]
        ego_dist = self._distance(ego.position, cross_center)

        # 2) Spawn peds when ego approaches the crossing
        if not self.event_spawned and ego_dist <= SPAWN_DISTANCE:
            self._spawn_group(env, step)
            self.event_spawned = True

        # 3) Keep them waiting first
        if self.waiting:
            self._update_waiting(env, step, ego_dist)

        # 4) Move them across
        if self.crossing:
            self._update_crossing(env)

        # 5) Finish event when everybody has crossed
        if self.event_spawned and not self.waiting and not self.crossing:
            self.event_finished = True

    def _make_crossing_site_from_ego_lane(self, ego):
        lane = getattr(ego, "lane", None)
        if lane is None:
            return None

        try:
            ego_long, _ = lane.local_coordinates(ego.position)
            lane_length = float(lane.length)
            lane_heading = float(lane.heading_theta_at(ego_long))
        except Exception:
            return None

        cross_long = min(max(float(ego_long) + 10.0, 6.0), lane_length - 6.0)
        center = lane.position(cross_long, 0.0)

        return {
            "lane": lane,
            "cross_long": cross_long,
            "center": [float(center[0]), float(center[1])],
            "heading": lane_heading,
        }

    def _spawn_group(self, env, step):
        lane = self.crossing_site["lane"]
        cross_long = self.crossing_site["cross_long"]
        heading = self.crossing_site["heading"]

        ped_count = random.randint(*PED_COUNT)
        spawn_side = random.choice([-1.0, 1.0])

        curb_lat = spawn_side * lane.width * CROSSWALK_LAT_FACTOR
        target_lat = -curb_lat

        for i in range(ped_count):
            ped_long = cross_long + random.uniform(-CROSSWALK_LONG_JITTER, CROSSWALK_LONG_JITTER)
            ped_long = min(max(ped_long, 2.0), lane.length - 2.0)

            wait_pos = lane.position(ped_long, curb_lat)
            target_pos = lane.position(ped_long, target_lat)

            ped_heading = heading + (-math.pi / 2 if spawn_side > 0 else math.pi / 2)

            ped = env.engine.spawn_object(
                Pedestrian,
                position=[float(wait_pos[0]), float(wait_pos[1])],
                heading_theta=ped_heading,
                random_seed=env.current_seed + step + i + 100,
            )
            if ped is None:
                continue

            ped.set_heading_theta(ped_heading)
            ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)

            self.waiting.append(
                {
                    "ped": ped,
                    "wait_until": step + random.randint(*CURB_WAIT_STEPS) + i * 8,
                    "target_pos": [float(target_pos[0]), float(target_pos[1])],
                    "heading": ped_heading,
                    "speed": random.uniform(*PED_SPEED),
                }
            )

    def _update_waiting(self, env, step, ego_dist):
        remaining = []

        for item in self.waiting:
            ped = item["ped"]
            if not self._valid(env, ped):
                continue

            ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
            ped.set_heading_theta(item["heading"])

            if step >= item["wait_until"] and ego_dist <= START_CROSS_DISTANCE:
                self.crossing.append(item)
            else:
                remaining.append(item)

        self.waiting = remaining

    def _update_crossing(self, env):
        remaining = []

        for item in self.crossing:
            ped = item["ped"]
            if not self._valid(env, ped):
                continue

            pos = ped.position
            tx, ty = item["target_pos"]
            dx = tx - float(pos[0])
            dy = ty - float(pos[1])
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.4:
                ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
                continue

            vx = dx / max(dist, 1e-6)
            vy = dy / max(dist, 1e-6)

            ped.set_heading_theta(item["heading"])
            ped.set_velocity([vx, vy], item["speed"], in_local_frame=False)
            remaining.append(item)

        self.crossing = remaining

    @staticmethod
    def _distance(a, b):
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _valid(env, obj):
        try:
            return obj is not None and obj.id in env.engine.get_objects()
        except Exception:
            return False

    def debug_text(self):
        return {
            "event_spawned": "Yes" if self.event_spawned else "No",
            "waiting": len(self.waiting),
            "crossing": len(self.crossing),
            "finished": "Yes" if self.event_finished else "No",
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
    crossing_controller = NaturalCrosswalkTestController()
    episode = 0

    try:
        env = MetaDriveEnv(cfg)
        env.reset()
        crossing_controller.reset(env)
        episode += 1

        print(HELP_MESSAGE)
        print("Street expert custom route loaded.")
        print("Reliable mid-route crosswalk test enabled.")
        print("Pedestrians spawn near the curb and cross when ego approaches.")

        for step in range(1, 1_000_000):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            crossing_controller.update(env, step, info)

            route_completion = info.get("route_completion", 0.0) * 100.0
            debug = crossing_controller.debug_text()

            env.render(
                text={
                    "Policy": "ExpertPolicy",
                    "route_completion": f"{route_completion:.1f}%",
                    "Episode": episode,
                    "Ped event spawned": debug["event_spawned"],
                    "Waiting": debug["waiting"],
                    "Crossing": debug["crossing"],
                    "Finished": debug["finished"],
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