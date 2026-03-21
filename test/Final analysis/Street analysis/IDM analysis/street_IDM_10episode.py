#!/usr/bin/env python
"""
Procedural MetaDrive urban scenario: PG block-sequence map (no dataset / ScenarioEnv),
with curves, a merge/split bottleneck, T-junction and 4-way intersection.

Updated version:
- exactly 3 crosswalks
- each crosswalk triggers separately along the route
- pedestrians spawn from BOTH sides of the road
- pedestrians cross in BOTH directions randomly
- once a group finishes crossing, the next crosswalk can trigger
- about 6-7 traffic vehicles added
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np
from openpyxl import Workbook

# One level deeper than street_IDM.py (IDM analysis/); repo root is metadrive-main
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metadrive.component.map.pg_map import PGMap
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.constants import DEFAULT_AGENT, HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.type import MetaDriveType


# =============================================================================
# Tunable parameters
# =============================================================================
BLOCK_SEQUENCE = "SSCSCySYCTSXCS"
EXIT_LENGTH = 72.0
LANE_NUM = 2
LANE_WIDTH = 3.5

# Traffic enabled: about 6-7 vehicles on this map
TRAFFIC_DENSITY = 0.04
NEED_INVERSE_TRAFFIC = True
RANDOM_TRAFFIC = True

# Exactly 3 crosswalks
CROSSWALK_ROUTE_FRACTIONS = (0.34, 0.56, 0.78)

CROSSWALK_HALF_DEPTH_ALONG_ROAD = 1.8
CROSSWALK_HALF_WIDTH_ACROSS_ROAD = 5.5

GATE_CROSS_START_ON_EGO_NEAR = False
TRIGGER_START_CROSS_DISTANCE = 24.0

CURB_LATERAL_FACTOR = 0.68
SPAWN_LONG_JITTER = 0.45

# Smaller and more natural groups
PED_COUNT_MIN = 2
PED_COUNT_MAX = 4

PED_SPACING_ALONG_CROSSWALK = 0.90
BIDIRECTIONAL_LONG_STAGGER_M = 0.35

EGO_MIN_SPEED_KMH = 80.0
EGO_MAX_SPEED_KMH = 120.0
EGO_SPAWN_SPEED_M_S = EGO_MIN_SPEED_KMH / 3.6

SPAWN_TIME_TO_CROSSWALK_MIN_S = 1.35
SPAWN_TIME_TO_CROSSWALK_MAX_S = 2.25
SPAWN_ASSUME_MIN_EGO_SPEED_M_S = EGO_MIN_SPEED_KMH / 3.6 * 0.45
SPAWN_PANIC_DISTANCE_M = 26.0
MEET_BIAS_LONG_M = 2.2

WAIT_STEPS_MIN = 0
WAIT_STEPS_MAX = 6

PED_SPEED_MIN = 0.45
PED_SPEED_MAX = 0.80

PED_SPAWN_PROGRESS_ACROSS_ROAD = 0.12

IDM_BRAKE_DISTANCE_M = 32.0
IDM_BRAKE_BUFFER_M = 4.0
IDM_CROSSWALK_LATERAL_MARGIN_M = 1.0
IDM_EMERGENCY_BRAKE = -6.0

# Batch evaluation
NUM_EPISODES = 10
STUCK_TIMEOUT_S = 20.0
STUCK_SPEED_THRESHOLD_M_S = 0.2

OUTPUT_DIR = (
    r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main\test\Final analysis"
    r"\Street analysis\IDM analysis"
)


def _save_results_xlsx(rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"street_IDM_results_{ts}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "IDM"
    ws.append(
        [
            "Episode",
            "Pass",
            "Timeout",
            "crash count pedestrian",
            "crash count vehicle",
            "time taken to reach goal sim s",
            "speed avg kmh",
            "simulation datetime",
        ]
    )
    for r in rows:
        ws.append(
            [
                r["episode"],
                r["Pass"],
                r["Timeout"],
                r["crash count pedestrian"],
                r["crash count vehicle"],
                r["time taken to reach goal sim s"],
                r["speed avg kmh"],
                r["simulation datetime"],
            ]
        )
    wb.save(path)
    return path


def _excel_row_from_episode_record(r: dict) -> dict:
    timeout = r["outcome"] == "timeout_stuck"
    arrive = bool(r.get("arrive_dest", False))
    pass_ok = arrive and not timeout
    return {
        "episode": r["episode"],
        "Pass": "YES" if pass_ok else "NO",
        "Timeout": "YES" if timeout else "NO",
        "crash count pedestrian": r["pedestrians_hit"],
        "crash count vehicle": r["vehicle_crashes"],
        "time taken to reach goal sim s": r["time_goal_sim_s"],
        "speed avg kmh": r["speed_avg_kmh"],
        "simulation datetime": r["simulation_datetime"],
    }


class CrosswalkPGMap(PGMap):
    def _generate(self):
        super()._generate()
        self.pedestrian_event_sites = []
        self.pedestrian_event_anchor = None
        spawn_lane_index = (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0)
        self._inject_crosswalks_for_spawn_lane(
            spawn_lane_index=spawn_lane_index,
            route_fractions=CROSSWALK_ROUTE_FRACTIONS,
            half_depth=CROSSWALK_HALF_DEPTH_ALONG_ROAD,
            half_width=CROSSWALK_HALF_WIDTH_ACROSS_ROAD,
        )

    @staticmethod
    def _anchor_at_route_fraction(segment_info, total, route_fraction, half_depth, half_width):
        if total < 15.0 or not segment_info:
            return None
        target_dist = float(np.clip(route_fraction, 0.08, 0.92)) * total
        acc = 0.0
        chosen_lane = None
        s_local = None
        for lane, ln in segment_info:
            if acc + ln >= target_dist:
                s_local = target_dist - acc
                s_local = float(np.clip(s_local, 3.0, max(3.0, ln - 3.0)))
                chosen_lane = lane
                break
            acc += ln

        if chosen_lane is None:
            lane, ln = segment_info[-1]
            chosen_lane = lane
            s_local = float(np.clip(ln * 0.5, 3.0, max(3.0, ln - 3.0)))

        center = chosen_lane.position(s_local, 0.0)
        poly = np.array(
            [
                chosen_lane.position(s_local - half_depth, -half_width),
                chosen_lane.position(s_local + half_depth, -half_width),
                chosen_lane.position(s_local + half_depth, half_width),
                chosen_lane.position(s_local - half_depth, half_width),
            ],
            dtype=np.float64,
        )
        return {
            "lane": chosen_lane,
            "cross_long": s_local,
            "center": np.array([float(center[0]), float(center[1])], dtype=np.float64),
            "heading": float(chosen_lane.heading_theta_at(s_local)),
            "polygon": poly,
        }

    def _inject_crosswalks_for_spawn_lane(self, spawn_lane_index, route_fractions, half_depth, half_width):
        rn = self.road_network
        seed = self.engine.global_random_seed
        try:
            dest = NodeNetworkNavigation.auto_assign_task(self, spawn_lane_index, None, seed)
        except Exception:
            return

        path = rn.shortest_path(spawn_lane_index, dest)
        if len(path) < 2:
            return

        spawn_lane_id = spawn_lane_index[2]
        total = 0.0
        segment_info = []
        for c1, c2 in zip(path[:-1], path[1:]):
            lanes = rn.graph[c1][c2]
            li = min(spawn_lane_id, len(lanes) - 1)
            lane = lanes[li]
            segment_info.append((lane, float(lane.length)))
            total += float(lane.length)

        fracs = route_fractions
        if isinstance(fracs, (int, float)):
            fracs = (float(fracs),)

        seen = set()
        sites = []
        zebra_i = 0
        for frac in fracs:
            key = round(float(frac), 4)
            if key in seen:
                continue
            seen.add(key)
            data = self._anchor_at_route_fraction(segment_info, total, float(frac), half_depth, half_width)
            if data is None:
                continue
            poly = data.pop("polygon")
            self.crosswalks[f"proc_zebra_{zebra_i}"] = {
                "type": MetaDriveType.CROSSWALK,
                "polygon": poly,
            }
            zebra_i += 1
            sites.append(data)

        self.pedestrian_event_sites = sites
        if sites:
            self.pedestrian_event_anchor = sites[0]


class CrosswalkPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.maps[current_seed] is None:
            map_config = config["map_config"]
            map_config.update({"seed": current_seed})
            map_config = self.add_random_to_map(map_config)
            m = self.spawn_object(CrosswalkPGMap, map_config=map_config, random_seed=None)
            self.current_map = m
            if self.engine.global_config["store_map"]:
                self.maps[current_seed] = m
        else:
            m = self.maps[current_seed]
            self.load_map(m)


class UrbanCrosswalkEnv(MetaDriveEnv):
    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.object_manager import TrafficObjectManager

        self.engine.register_manager("map_manager", CrosswalkPGMapManager())
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())


class PedestrianAwareIDMPolicy(IDMPolicy):
    """
    IDM with an additional yield rule for pedestrians occupying or about to occupy
    the ego lane at injected crosswalks.
    """

    def act(self, *args, **kwargs):
        action = super().act(*args, **kwargs)
        action[1] = min(action[1], self._pedestrian_crosswalk_brake())
        self.action_info["action"] = action
        return action

    def _pedestrian_crosswalk_brake(self):
        ego = self.control_object
        lane = self.routing_target_lane or ego.lane
        if lane is None:
            return 1.0

        crosswalk_long = self._nearest_crosswalk_long(lane)
        if crosswalk_long is None:
            return 1.0

        ego_long, _ = lane.local_coordinates(ego.position)
        dist_to_crosswalk = crosswalk_long - ego_long
        if dist_to_crosswalk <= 0.0 or dist_to_crosswalk > IDM_BRAKE_DISTANCE_M:
            return 1.0

        hazard = self._pedestrian_hazard_near_crosswalk(lane, crosswalk_long)
        if not hazard:
            return 1.0

        gap = max(dist_to_crosswalk - IDM_BRAKE_BUFFER_M, 0.5)
        required = -(ego.speed ** 2) / (2.0 * gap)
        return min(IDM_EMERGENCY_BRAKE, required)

    def _nearest_crosswalk_long(self, lane):
        current_map = getattr(self.engine.map_manager, "current_map", None)
        sites = list(getattr(current_map, "pedestrian_event_sites", None) or [])
        nearest = None
        for site in sites:
            site_lane = site.get("lane")
            if site_lane is not lane:
                continue
            s = site.get("cross_long")
            if s is None:
                continue
            if nearest is None or s < nearest:
                nearest = float(s)
        return nearest

    def _pedestrian_hazard_near_crosswalk(self, lane, crosswalk_long):
        ego = self.control_object
        objects = self.engine.get_objects().values()
        crosswalk_half_depth = CROSSWALK_HALF_DEPTH_ALONG_ROAD + 1.0
        crosswalk_half_width = lane.width * max(1.0, float(LANE_NUM)) * 0.5 + IDM_CROSSWALK_LATERAL_MARGIN_M

        for obj in objects:
            if not isinstance(obj, Pedestrian):
                continue
            if obj is ego:
                continue
            try:
                ped_long, ped_lat = lane.local_coordinates(obj.position)
            except Exception:
                continue
            if abs(ped_long - crosswalk_long) <= crosswalk_half_depth and abs(ped_lat) <= crosswalk_half_width:
                return True
        return False


class MidRouteCrosswalkPedController:
    """
    3 independent crosswalk events.
    For each crosswalk:
    - pedestrians spawn from BOTH sides
    - crossing directions are mixed randomly
    - next crosswalk only activates after current group is done
    """

    def __init__(self):
        self._sites = []
        self._site_index = 0
        self.event_spawned = False
        self.all_sites_finished = False
        self.waiting = []
        self.crossing = []
        self._spawned_ped_ids = []

    def cleanup(self, env):
        eng = env.engine
        ids = [i for i in self._spawned_ped_ids if i in eng.get_objects()]
        if ids:
            eng.clear_objects(ids, force_destroy=True, record=False)
        self._spawned_ped_ids = []

    def reset(self, env):
        m = env.engine.map_manager.current_map
        self._sites = list(getattr(m, "pedestrian_event_sites", None) or [])
        if not self._sites:
            a = getattr(m, "pedestrian_event_anchor", None)
            if a is not None:
                self._sites = [a]
        self._site_index = 0
        self.event_spawned = False
        self.all_sites_finished = False
        self.waiting = []
        self.crossing = []
        self._spawned_ped_ids = []

    def update(self, env, step, info):
        ego = env.agent
        if ego is None or self.all_sites_finished:
            return
        if not self._sites:
            return
        if self._site_index >= len(self._sites):
            self.all_sites_finished = True
            return

        anchor = self._sites[self._site_index]
        cross_center = anchor["center"]
        ego_dist = self._dist(ego.position, cross_center)

        if not self.event_spawned and self._should_spawn_now(ego, ego_dist):
            self._spawn_pedestrians(env, step, anchor, ego, site_k=self._site_index)
            self.event_spawned = True

        if self.waiting:
            self._update_waiting(env, step, ego_dist)

        if self.crossing:
            self._update_crossing(env)

        if self.event_spawned and not self.waiting and not self.crossing:
            self._site_index += 1
            self.event_spawned = False
            if self._site_index >= len(self._sites):
                self.all_sites_finished = True

    @staticmethod
    def _ego_speed_m_s(ego):
        try:
            return max(float(ego.speed), SPAWN_ASSUME_MIN_EGO_SPEED_M_S)
        except Exception:
            return SPAWN_ASSUME_MIN_EGO_SPEED_M_S

    def _should_spawn_now(self, ego, ego_dist):
        spd = self._ego_speed_m_s(ego)
        t = ego_dist / max(spd, 1e-3)
        if SPAWN_TIME_TO_CROSSWALK_MIN_S <= t <= SPAWN_TIME_TO_CROSSWALK_MAX_S:
            return True
        if ego_dist <= SPAWN_PANIC_DISTANCE_M:
            return True
        if ego_dist <= 38.0 and t <= 1.05:
            return True
        return False

    def _spawn_pedestrians(self, env, step, anchor, ego, site_k=0):
        lane = anchor["lane"]
        s0 = float(anchor["cross_long"])

        try:
            ego_long, _ = lane.local_coordinates(ego.position)
            bias = MEET_BIAS_LONG_M if s0 > ego_long else -MEET_BIAS_LONG_M
            s_lane_anchor = s0 - bias
        except Exception:
            s_lane_anchor = s0

        n = random.randint(PED_COUNT_MIN, PED_COUNT_MAX)

        side_choices = [1.0, -1.0]
        while len(side_choices) < n:
            side_choices.append(random.choice([-1.0, 1.0]))
        random.shuffle(side_choices)

        offsets = [(i - 0.5 * (n - 1)) * PED_SPACING_ALONG_CROSSWALK for i in range(n)]

        for i, (off, spawn_side) in enumerate(zip(offsets, side_choices)):
            curb_lat = spawn_side * lane.width * CURB_LATERAL_FACTOR
            target_lat = -curb_lat * max(1.0, float(LANE_NUM))

            ped_long = (
                s_lane_anchor
                + off
                + random.uniform(-SPAWN_LONG_JITTER, SPAWN_LONG_JITTER)
                + random.uniform(-BIDIRECTIONAL_LONG_STAGGER_M, BIDIRECTIONAL_LONG_STAGGER_M)
            )
            ped_long = float(np.clip(ped_long, 2.0, max(2.0, lane.length - 2.0)))

            wait_pos = lane.position(ped_long, curb_lat)
            target_pos = lane.position(ped_long, target_lat)

            tx, ty = float(target_pos[0]), float(target_pos[1])
            wx, wy = float(wait_pos[0]), float(wait_pos[1])

            alpha_i = float(np.clip(PED_SPAWN_PROGRESS_ACROSS_ROAD + random.uniform(0.0, 0.08), 0.0, 0.92))
            sx = wx * (1.0 - alpha_i) + tx * alpha_i
            sy = wy * (1.0 - alpha_i) + ty * alpha_i

            ped_heading = math.atan2(ty - sy, tx - sx)

            full_dx, full_dy = tx - wx, ty - wy
            fn = math.hypot(full_dx, full_dy)
            if fn < 1e-3:
                continue

            ped = env.engine.spawn_object(
                Pedestrian,
                position=[sx, sy],
                heading_theta=ped_heading,
                random_seed=env.current_seed + step + 701 + site_k * 911 + i * 97,
            )
            if ped is None:
                continue

            self._spawned_ped_ids.append(ped.id)
            ped.set_heading_theta(ped_heading)
            ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)

            wait_until = step + random.randint(WAIT_STEPS_MIN, WAIT_STEPS_MAX) + random.randint(0, 4)

            self.waiting.append(
                {
                    "ped": ped,
                    "wait_until": wait_until,
                    "target_pos": [tx, ty],
                    "heading": ped_heading,
                    "speed": random.uniform(PED_SPEED_MIN, PED_SPEED_MAX),
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
            can_cross = step >= item["wait_until"]
            if GATE_CROSS_START_ON_EGO_NEAR:
                can_cross = can_cross and ego_dist <= TRIGGER_START_CROSS_DISTANCE
            if can_cross:
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
            dx, dy = tx - float(pos[0]), ty - float(pos[1])
            dist = math.hypot(dx, dy)

            if dist < 0.5:
                ped.set_velocity([0.0, 0.0], 0.0, in_local_frame=True)
                continue

            ux, uy = dx / max(dist, 1e-6), dy / max(dist, 1e-6)
            ped.set_heading_theta(math.atan2(uy, ux))
            ped.set_velocity([ux, uy], item["speed"], in_local_frame=False)
            remaining.append(item)

        self.crossing = remaining

    @staticmethod
    def _dist(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    @staticmethod
    def _valid(env, obj):
        try:
            return obj is not None and obj.id in env.engine.get_objects()
        except Exception:
            return False

    def debug_text(self):
        n = max(len(self._sites), 1)
        return {
            "spawned": "yes" if self.event_spawned else "no",
            "site": f"{self._site_index + 1}/{n}",
            "waiting": str(len(self.waiting)),
            "crossing": str(len(self.crossing)),
            "done": "yes" if self.all_sites_finished else "no",
        }


def build_config(use_render: bool):
    return {
        "use_render": use_render,
        "manual_control": False,
        "agent_policy": PedestrianAwareIDMPolicy,
        "num_scenarios": 1,
        "start_seed": 0,
        "traffic_density": TRAFFIC_DENSITY,
        "random_traffic": RANDOM_TRAFFIC,
        "need_inverse_traffic": NEED_INVERSE_TRAFFIC,
        "accident_prob": 0.0,
        "static_traffic_object": False,
        "horizon": 8000,
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "crash_human_done": False,
        "out_of_road_done": False,
        "random_spawn_lane_index": False,
        "show_crosswalk": True,
        "map": 3,
        "map_config": {
            "type": "block_sequence",
            "config": BLOCK_SEQUENCE,
            "lane_num": LANE_NUM,
            "lane_width": LANE_WIDTH,
            "exit_length": EXIT_LENGTH,
        },
        "vehicle_config": dict(
            show_navi_mark=True,
            spawn_lane_index=(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
            lidar=dict(num_lasers=120, distance=50, num_others=0),
            lane_line_detector=dict(num_lasers=12, distance=50),
            side_detector=dict(num_lasers=12, distance=50),
        ),
        "agent_configs": {
            DEFAULT_AGENT: dict(
                max_speed_km_h=EGO_MAX_SPEED_KMH,
                spawn_velocity=[EGO_SPAWN_SPEED_M_S, 0.0],
                spawn_velocity_car_frame=True,
            )
        },
        "interface_panel": ["dashboard"],
    }


def _ego_speed_m_s(env) -> float:
    try:
        return float(env.agent.speed)
    except Exception:
        return 0.0


def main():
    use_render = "--headless" not in sys.argv
    cfg = build_config(use_render=use_render)
    ped_ctl = MidRouteCrosswalkPedController()
    env = None
    horizon = int(cfg.get("horizon", 8000))
    max_steps_per_episode = min(horizon, 1_000_000)
    episode_results: list[dict] = []

    try:
        env = UrbanCrosswalkEnv(cfg)
        sim_step_sec = env.config["physics_world_step_size"] * env.config["decision_repeat"]

        print(HELP_MESSAGE)
        print("Urban procedural map:", BLOCK_SEQUENCE)
        print("Ego policy: IDM Policy")
        print(f"Batch: {NUM_EPISODES} episodes | stuck if speed < {STUCK_SPEED_THRESHOLD_M_S} m/s for {STUCK_TIMEOUT_S} s wall time")
        print("Traffic enabled: about 6-7 vehicles")
        print("Crosswalk route fractions:", CROSSWALK_ROUTE_FRACTIONS)
        print(f"Results (.xlsx) -> {OUTPUT_DIR}")

        for ep in range(1, NUM_EPISODES + 1):
            env.reset()
            ped_ctl.reset(env)
            ns = len(getattr(env.current_map, "pedestrian_event_sites", None) or [])
            if ep == 1:
                print(f"Ped crossing sites along route: {ns}")

            sim_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            episode_wall_start_s = time.perf_counter()
            ped_hit_count = 0
            veh_crash_count = 0
            prev_crash_human = False
            prev_crash_vehicle = False
            stuck_elapsed_s = 0.0
            last_tick_s = time.perf_counter()
            outcome = None
            final_info: dict = {}
            sim_elapsed = 0.0
            speed_sum = 0.0
            speed_steps = 0

            for step in range(1, max_steps_per_episode + 1):
                now = time.perf_counter()
                dt = now - last_tick_s
                last_tick_s = now

                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                ped_ctl.update(env, step, info)
                route_pct = info.get("route_completion", 0.0) * 100.0
                final_info = info
                sim_elapsed += sim_step_sec
                try:
                    speed_sum += float(env.agent.speed_km_h)
                    speed_steps += 1
                except Exception:
                    pass

                ch = bool(info.get("crash_human", False))
                cv = bool(info.get("crash_vehicle", False))
                if ch and not prev_crash_human:
                    ped_hit_count += 1
                if cv and not prev_crash_vehicle:
                    veh_crash_count += 1
                prev_crash_human, prev_crash_vehicle = ch, cv

                spd = _ego_speed_m_s(env)
                if spd < STUCK_SPEED_THRESHOLD_M_S:
                    stuck_elapsed_s += dt
                else:
                    stuck_elapsed_s = 0.0

                wall_s = now - episode_wall_start_s

                if use_render:
                    env.render(
                        text={
                            "0. Episode": f"{ep}/{NUM_EPISODES}",
                            "1. Policy": "IDM Policy",
                            "2. Time": f"{wall_s:.2f} s",
                            "3. Stuck timer": f"{stuck_elapsed_s:.1f} / {STUCK_TIMEOUT_S:.0f} s",
                            "4. Peds hit": str(ped_hit_count),
                            "5. Veh crashes": str(veh_crash_count),
                        }
                    )

                if stuck_elapsed_s >= STUCK_TIMEOUT_S:
                    outcome = "timeout_stuck"
                    wall_s = now - episode_wall_start_s
                    speed_avg = speed_sum / max(speed_steps, 1)
                    episode_results.append(
                        {
                            "episode": ep,
                            "outcome": outcome,
                            "route_completion_pct": route_pct,
                            "wall_time_s": wall_s,
                            "pedestrians_hit": ped_hit_count,
                            "vehicle_crashes": veh_crash_count,
                            "arrive_dest": False,
                            "simulation_datetime": sim_datetime,
                            "time_goal_sim_s": "N/A",
                            "speed_avg_kmh": round(speed_avg, 2),
                        }
                    )
                    print(
                        f"Episode {ep}/{NUM_EPISODES} end: timeout (stuck) - "
                        f"route={route_pct:.1f}%, wall={wall_s:.2f}s, peds_hit={ped_hit_count}, veh_crashes={veh_crash_count}"
                    )
                    ped_ctl.cleanup(env)
                    break

                if terminated or truncated:
                    arrive = bool(info.get("arrive_dest", False))
                    if arrive:
                        outcome = "arrive_dest"
                    elif truncated:
                        outcome = "truncated"
                    else:
                        outcome = "terminated"
                    wall_s = now - episode_wall_start_s
                    speed_avg = speed_sum / max(speed_steps, 1)
                    episode_results.append(
                        {
                            "episode": ep,
                            "outcome": outcome,
                            "route_completion_pct": route_pct,
                            "wall_time_s": wall_s,
                            "pedestrians_hit": ped_hit_count,
                            "vehicle_crashes": veh_crash_count,
                            "arrive_dest": arrive,
                            "simulation_datetime": sim_datetime,
                            "time_goal_sim_s": round(sim_elapsed, 2) if arrive else "N/A",
                            "speed_avg_kmh": round(speed_avg, 2),
                        }
                    )
                    print(
                        f"Episode {ep}/{NUM_EPISODES} end: {outcome} - "
                        f"route={route_pct:.1f}%, wall={wall_s:.2f}s, arrive_dest={arrive}, "
                        f"peds_hit={ped_hit_count}, veh_crashes={veh_crash_count}"
                    )
                    ped_ctl.cleanup(env)
                    break
            else:
                outcome = "max_steps"
                wall_s = time.perf_counter() - episode_wall_start_s
                route_pct = final_info.get("route_completion", 0.0) * 100.0
                arrive_final = bool(final_info.get("arrive_dest", False))
                speed_avg = speed_sum / max(speed_steps, 1)
                episode_results.append(
                    {
                        "episode": ep,
                        "outcome": outcome,
                        "route_completion_pct": route_pct,
                        "wall_time_s": wall_s,
                        "pedestrians_hit": ped_hit_count,
                        "vehicle_crashes": veh_crash_count,
                        "arrive_dest": arrive_final,
                        "simulation_datetime": sim_datetime,
                        "time_goal_sim_s": round(sim_elapsed, 2) if arrive_final else "N/A",
                        "speed_avg_kmh": round(speed_avg, 2),
                    }
                )
                print(
                    f"Episode {ep}/{NUM_EPISODES} end: max_steps - "
                    f"route={route_pct:.1f}%, wall={wall_s:.2f}s, peds_hit={ped_hit_count}, veh_crashes={veh_crash_count}"
                )
                ped_ctl.cleanup(env)

        print("\n=== Summary ({0} episodes) ===".format(NUM_EPISODES))
        for r in episode_results:
            print(
                f"  ep{r['episode']:2d}  {r['outcome']:16s}  route={r['route_completion_pct']:5.1f}%  "
                f"t={r['wall_time_s']:6.2f}s  peds={r['pedestrians_hit']}  veh={r['vehicle_crashes']}"
            )
        c = Counter(r["outcome"] for r in episode_results)
        print("Outcome counts:", dict(c))

        xlsx_rows = [_excel_row_from_episode_record(r) for r in episode_results]
        out_path = _save_results_xlsx(xlsx_rows)
        print(f"\nSaved Excel:\n  {out_path}")
    finally:
        if env is not None:
            try:
                ped_ctl.cleanup(env)
            except Exception:
                pass
            env.close()


if __name__ == "__main__":
    main()
