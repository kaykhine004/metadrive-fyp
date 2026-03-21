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

This script tracks **only the ego / agent vehicle** (ExpertPolicy). Other traffic
vehicles are ignored for pass/fail; only ego stats and crashes are shown.

10-episode batch: logs Pass / Timeout / crashes / sim time / speed to an .xlsx in
``expert analysis`` (see OUTPUT_DIR). Crash counts reset each episode. Pass = YES
if the episode did not time out (see Timeout); time-to-goal is simulation seconds
when ``arrive_dest`` is True, else N/A.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import Texture

# Repo root = metadrive-main (this file lives under test/.../expert analysis/)
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
from metadrive.policy.expert_policy import ExpertPolicy
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

TOTAL_EPISODES = 10
STOP_TIMEOUT_SEC = 60.0
STOP_SPEED_KMH = 0.5
OUTPUT_DIR = (
    r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main\test\Final analysis"
    r"\Street analysis\expert analysis"
)


def _press_fast_forward():
    try:
        import pyautogui

        pyautogui.press("f")
    except Exception:
        pass


def _save_results_xlsx(rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"street_expert_results_{ts}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Expert"
    ws.append(
        [
            "Episode",
            "Pass",
            "Timeout",
            "crash_count_pedestrian",
            "crash_count_vehicle",
            "time_taken_to_reach_goal_sim_s",
            "speed_avg_kmh",
            "simulation_datetime",
        ]
    )
    for r in rows:
        ws.append(
            [
                r["episode"],
                r["Pass"],
                r["Timeout"],
                r["crash_count_pedestrian"],
                r["crash_count_vehicle"],
                r["time_taken_to_reach_goal_sim_s"],
                r["speed_avg_kmh"],
                r["simulation_datetime"],
            ]
        )
    wb.save(path)
    return path


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
        "agent_policy": ExpertPolicy,
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
        "top_down_camera_initial_z": 260,
    }


# --- Top-down minimap (ego-only trail; START / END) ---------------------------------
TD_SCREEN = (420, 420)
TD_SCALING = 1.0
TD_FILM = (800, 800)
TRAIL_MAX_POINTS = 700


def _ego_goal_xy(env) -> tuple[float, float] | None:
    v = env.agent
    if v is None:
        return None
    nav = getattr(v, "navigation", None)
    if nav is None or getattr(nav, "final_lane", None) is None:
        return None
    fl = nav.final_lane
    p = fl.position(fl.length, 0.0)
    return float(p[0]), float(p[1])


def _world_to_screen(wx, wy, renderer):
    canvas = renderer._frame_canvas
    fx, fy = canvas.pos2pix(wx, wy)
    film_w, film_h = canvas.get_size()
    scr_w, scr_h = renderer._screen_canvas.get_size()
    return int(fx - (film_w / 2 - scr_w / 2)), int(fy - (film_h / 2 - scr_h / 2))


def _draw_minimap(
    img,
    renderer,
    vehicle_pos,
    start_pos,
    end_pos,
    trail_xy: deque,
    episode_label: str | None = None,
):
    out = img.copy()
    if episode_label:
        cv2.putText(
            out,
            episode_label,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            out,
            episode_label,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    sx, sy = _world_to_screen(*start_pos, renderer)
    cv2.putText(out, "START", (sx - 22, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "START", (sx - 22, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    ex, ey = _world_to_screen(*end_pos, renderer)
    cv2.putText(out, "END", (ex - 16, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "END", (ex - 16, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1)
    if len(trail_xy) >= 2:
        pts = np.array(
            [_world_to_screen(float(px), float(py), renderer) for px, py in trail_xy],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], isClosed=False, color=(80, 180, 255), thickness=2)
    vx, vy = _world_to_screen(*vehicle_pos, renderer)
    cv2.circle(out, (vx, vy), 5, (0, 0, 255), -1)
    cv2.circle(out, (vx, vy), 6, (0, 0, 0), 1)
    return out


def _numpy_to_panda_texture(img_bgr, tex=None):
    h, w = img_bgr.shape[:2]
    img_rgba = np.concatenate(
        [img_bgr[..., ::-1], np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
    )
    img_flipped = np.flipud(img_rgba).copy()
    if tex is None:
        tex = Texture("topdown_street")
        tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.setRamImage(img_flipped.tobytes())
    return tex


def main():
    use_render = "--headless" not in sys.argv
    cfg = build_config(use_render=use_render)
    ped_ctl = MidRouteCrosswalkPedController()
    env = None
    overlay = None
    td_tex = None
    try:
        env = UrbanCrosswalkEnv(cfg)
        env.reset()
        ped_ctl.reset(env)
        if use_render:
            time.sleep(0.4)
        _press_fast_forward()

        print(HELP_MESSAGE)
        print("Urban procedural map:", BLOCK_SEQUENCE)
        print("Traffic enabled: about 6-7 vehicles")
        print("Crosswalk route fractions:", CROSSWALK_ROUTE_FRACTIONS)
        print("Tracked vehicle: ego agent only (ExpertPolicy); other vehicles not scored.")
        ns = len(getattr(env.current_map, "pedestrian_event_sites", None) or [])
        print(f"Ped crossing sites along route: {ns}")
        print(f"Running {TOTAL_EPISODES} episodes; results -> {OUTPUT_DIR}")

        sim_step_sec = env.config["physics_world_step_size"] * env.config["decision_repeat"]
        results: list[dict] = []

        max_steps = 2500 if not use_render else 1_000_000
        for episode_num in range(1, TOTAL_EPISODES + 1):
            sim_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            ego = env.agent
            start_pos = (float(ego.position[0]), float(ego.position[1]))
            gxy = _ego_goal_xy(env)
            end_pos = gxy if gxy is not None else start_pos
            trail_world: deque[tuple[float, float]] = deque(maxlen=TRAIL_MAX_POINTS)

            episode_wall_start_s = time.perf_counter()
            sim_elapsed = 0.0
            stop_accum = 0.0
            timeout = False
            ped_hit_count = 0
            veh_crash_count = 0
            prev_crash_human = False
            prev_crash_vehicle = False
            speed_sum = 0.0
            speed_steps = 0
            reached_goal = False
            route_pct = 0.0

            for step in range(1, max_steps + 1):
                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                ped_ctl.update(env, step, info)
                route_pct = info.get("route_completion", 0.0) * 100.0
                sim_elapsed += sim_step_sec

                ego = env.agent
                speed_kmh = float(ego.speed_km_h) if ego is not None else 0.0
                speed_sum += speed_kmh
                speed_steps += 1

                ch = bool(info.get("crash_human", False))
                cv = bool(info.get("crash_vehicle", False))
                if ch and not prev_crash_human:
                    ped_hit_count += 1
                if cv and not prev_crash_vehicle:
                    veh_crash_count += 1
                prev_crash_human, prev_crash_vehicle = ch, cv

                if speed_kmh <= STOP_SPEED_KMH:
                    stop_accum += sim_step_sec
                else:
                    stop_accum = 0.0
                if stop_accum >= STOP_TIMEOUT_SEC:
                    timeout = True

                pol = env.engine.get_policy(env.agent.name)
                policy_name = info.get("policy") or (pol.name if pol is not None else ExpertPolicy.__name__)
                wall_s = time.perf_counter() - episode_wall_start_s
                pass_str = "YES" if not timeout else "NO"

                if ego is not None:
                    trail_world.append((float(ego.position[0]), float(ego.position[1])))
                vpos = (
                    (float(ego.position[0]), float(ego.position[1]))
                    if ego is not None
                    else start_pos
                )

                if use_render:
                    env.render(
                        text={
                            "Episode": f"{episode_num} / {TOTAL_EPISODES}",
                            "Policy (ego only)": str(policy_name),
                            "Total time (wall)": f"{wall_s:.2f} s",
                            "Sim time": f"{sim_elapsed:.1f} s",
                            "Pass": pass_str,
                            "Timeout": "YES" if timeout else "NO",
                            "Crashes w/ pedestrians": str(ped_hit_count),
                            "Crashes w/ other vehicles": str(veh_crash_count),
                            "Route % (ego)": f"{route_pct:.1f}%",
                        }
                    )
                    td_img = env.render(
                        mode="top_down",
                        window=False,
                        film_size=TD_FILM,
                        screen_size=TD_SCREEN,
                        scaling=TD_SCALING,
                        draw_contour=True,
                        center_on_map=True,
                    )
                    if td_img is not None and env.top_down_renderer is not None:
                        td_img = _draw_minimap(
                            td_img,
                            env.top_down_renderer,
                            vpos,
                            start_pos,
                            end_pos,
                            trail_world,
                            episode_label=f"Episode {episode_num}/{TOTAL_EPISODES}",
                        )
                        td_tex = _numpy_to_panda_texture(td_img, td_tex)
                        if overlay is None:
                            overlay = OnscreenImage(
                                image=td_tex,
                                pos=(0.72, 0, 0.82),
                                scale=(0.22, 1, 0.22),
                                parent=env.engine.aspect2d,
                            )
                            overlay.setTransparency(True)
                        else:
                            overlay.setTexture(td_tex)

                reached_goal = bool(info.get("arrive_dest", False))
                if timeout or terminated or truncated:
                    if timeout:
                        print(
                            f"Episode {episode_num}/{TOTAL_EPISODES} STOPPED (timeout): "
                            f"stopped >= {STOP_TIMEOUT_SEC:.0f}s sim at speed <= {STOP_SPEED_KMH} km/h"
                        )
                    else:
                        print(
                            f"Episode {episode_num}/{TOTAL_EPISODES} end:",
                            f"route={route_pct:.1f}%",
                            "arrive_dest=",
                            info.get("arrive_dest", False),
                        )
                    break

            speed_avg = speed_sum / max(speed_steps, 1)
            pass_ok = not timeout
            results.append(
                {
                    "episode": episode_num,
                    "Pass": "YES" if pass_ok else "NO",
                    "Timeout": "YES" if timeout else "NO",
                    "crash_count_pedestrian": ped_hit_count,
                    "crash_count_vehicle": veh_crash_count,
                    "time_taken_to_reach_goal_sim_s": round(sim_elapsed, 2) if reached_goal else "N/A",
                    "speed_avg_kmh": round(speed_avg, 2),
                    "simulation_datetime": sim_datetime,
                }
            )
            print(
                f"Episode {episode_num}/{TOTAL_EPISODES} | Pass:{'YES' if pass_ok else 'NO'} | "
                f"Timeout:{timeout} | ped:{ped_hit_count} veh:{veh_crash_count} | "
                f"goal:{reached_goal} | sim_t:{sim_elapsed:.1f}s"
            )

            if episode_num < TOTAL_EPISODES:
                ped_ctl.cleanup(env)
                env.reset()
                ped_ctl.reset(env)
                if use_render:
                    time.sleep(0.4)
                _press_fast_forward()

        out_path = _save_results_xlsx(results)
        print(f"Finished {TOTAL_EPISODES} episodes. Saved:\n  {out_path}")
    finally:
        if env is not None:
            try:
                ped_ctl.cleanup(env)
            except Exception:
                pass
            env.close()


if __name__ == "__main__":
    main()