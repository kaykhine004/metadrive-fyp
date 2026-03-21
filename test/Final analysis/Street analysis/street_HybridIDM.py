#!/usr/bin/env python
"""
Procedural MetaDrive urban scenario: PG block-sequence map (no dataset / ScenarioEnv),
with curves, a merge/split bottleneck, T-junction and 4-way intersection.

Updated version:
- ego uses PedestrianAwareIDMPolicy (vehicles + pedestrians in IDM / emergency brake / steer nudge)
- exactly 3 crosswalks
- each crosswalk triggers separately along the route
- pedestrians spawn from BOTH sides of the road
- pedestrians cross in BOTH directions randomly
- once a group finishes crossing, the next crosswalk can trigger
- about 6-7 traffic vehicles added
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metadrive.component.map.pg_map import PGMap
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.constants import DEFAULT_AGENT, HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
from metadrive.type import MetaDriveType
from metadrive.utils.math import wrap_to_pi


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
EGO_CRUISE_SPEED_KMH = 72.0
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

# Pedestrian-aware IDM (see PedestrianAwareIDMPolicy)
PED_IDM_DETECT_FORWARD_M = 52.0
PED_IDM_LATERAL_EXTRA_M = 2.2
PED_IDM_LONG_BUFFER_M = 3.0
PED_IDM_EMERGENCY_BRAKE_M = 14.0
PED_IDM_STEER_GAIN = 0.42
PED_IDM_CONFLICT_TIME_HORIZON_S = 4.5
PED_IDM_ENTRY_YIELD_BUFFER_S = 0.35
PED_IDM_EXIT_CLEAR_BUFFER_S = 0.55
PED_IDM_MIN_LATERAL_SPEED_M_S = 0.05
PED_IDM_MAX_STOP_BRAKE_M_S2 = 6.8
ROUTE_CURVE_LOOKAHEAD_SAMPLES_M = (8.0, 16.0, 24.0)
ROUTE_CURVE_PREVIEW_TRIGGER_M = 48.0
ROUTE_CURVE_MIN_SPEED_KMH = 25.0
ROUTE_CURVE_LAT_ACCEL_M_S2 = 2.4


class PedestrianAwareIDMPolicy(IDMPolicy):
    """
    Standard IDM only considers vehicles on lane graph edges; pedestrians are invisible to it.
    This policy merges the closest conflicting pedestrian (in ego frame, ahead and in a wide lateral corridor)
    into the IDM front-obstacle slot and adds a small lateral nudge away from nearby peds.
    """

    NORMAL_SPEED = EGO_CRUISE_SPEED_KMH
    MAX_SPEED = EGO_MAX_SPEED_KMH
    MAX_LONG_DIST = 45

    def _lane_width_ego(self):
        ego = self.control_object
        try:
            return float(ego.navigation.get_current_lane_width())
        except Exception:
            return float(LANE_WIDTH)

    def _ego_frame_vectors(self):
        heading = np.asarray(self.control_object.heading, dtype=np.float64)
        right = np.array([heading[1], -heading[0]], dtype=np.float64)
        return heading, right

    def _pedestrian_conflict_half_width(self):
        ego = self.control_object
        lane_w = self._lane_width_ego()
        ego_half_width = 0.5 * float(getattr(ego, "WIDTH", 2.0))
        return max(0.5 * lane_w, ego_half_width + 0.9)

    def _best_pedestrian_obstacle(self, surrounding_objs):
        ego = self.control_object
        heading, right = self._ego_frame_vectors()
        conflict_half_width = self._pedestrian_conflict_half_width()
        ego_speed = max(float(np.dot(np.asarray(ego.velocity, dtype=np.float64), heading)), 0.0)
        ego_half_len = 0.5 * float(getattr(ego, "LENGTH", 4.5))

        best_obj = None
        best_plan = None

        for obj in surrounding_objs:
            if not isinstance(obj, Pedestrian):
                continue
            try:
                rel = ego.convert_to_local_coordinates(obj.position, ego.position)
            except Exception:
                continue
            fwd = float(rel[0])
            lat = float(rel[1])
            if fwd < 0.8 or fwd > PED_IDM_DETECT_FORWARD_M:
                continue

            ped_radius = float(getattr(obj, "RADIUS", 0.35))
            ped_vel = np.asarray(getattr(obj, "velocity", [0.0, 0.0]), dtype=np.float64)
            ped_fwd_speed = float(np.dot(ped_vel, heading))
            ped_lat_speed = float(np.dot(ped_vel, right))
            lat_speed_abs = abs(ped_lat_speed)
            inside_corridor = abs(lat) <= (conflict_half_width + ped_radius)

            if inside_corridor:
                t_enter = 0.0
                if lat_speed_abs < PED_IDM_MIN_LATERAL_SPEED_M_S:
                    t_exit = float("inf")
                else:
                    t_exit = max((conflict_half_width + ped_radius - abs(lat)) / lat_speed_abs, 0.0)
            else:
                if lat * ped_lat_speed >= 0.0 or lat_speed_abs < PED_IDM_MIN_LATERAL_SPEED_M_S:
                    continue
                dist_to_corridor = abs(lat) - conflict_half_width - ped_radius
                t_enter = max(dist_to_corridor / lat_speed_abs, 0.0)
                t_exit = (abs(lat) + conflict_half_width + ped_radius) / lat_speed_abs

            if t_enter > PED_IDM_CONFLICT_TIME_HORIZON_S:
                continue

            long_at_enter = fwd + ped_fwd_speed * t_enter
            long_at_exit = fwd + ped_fwd_speed * t_exit if np.isfinite(t_exit) else fwd
            conflict_long = min(long_at_enter, long_at_exit) if np.isfinite(t_exit) else long_at_enter
            stop_dist = conflict_long - ego_half_len - ped_radius - PED_IDM_LONG_BUFFER_M
            stop_dist = max(stop_dist, 0.25)
            ego_arrival_s = stop_dist / max(ego_speed, 0.1)

            block_start_s = max(t_enter - PED_IDM_ENTRY_YIELD_BUFFER_S, 0.0)
            block_end_s = float("inf") if not np.isfinite(t_exit) else t_exit + PED_IDM_EXIT_CLEAR_BUFFER_S
            should_yield = block_start_s <= ego_arrival_s <= block_end_s
            if not should_yield:
                continue

            if np.isfinite(block_end_s):
                safe_speed_m_s = stop_dist / max(block_end_s, 0.35)
                speed_cap_kmh = float(np.clip(safe_speed_m_s * 3.6, self.CREEP_SPEED, self.NORMAL_SPEED))
            else:
                speed_cap_kmh = float(self.CREEP_SPEED)

            emergency = stop_dist < PED_IDM_EMERGENCY_BRAKE_M or ego_arrival_s <= max(t_enter, 0.35)
            plan = {
                "obj": obj,
                "distance": stop_dist,
                "speed_cap_kmh": speed_cap_kmh,
                "arrival_s": ego_arrival_s,
                "emergency": emergency,
            }

            if best_plan is None or (plan["distance"], plan["arrival_s"]) < (best_plan["distance"], best_plan["arrival_s"]):
                best_plan = plan
                best_obj = obj

        if best_plan is None:
            return None
        return best_plan

    def _pedestrian_steer_nudge(self, steering, surrounding_objs):
        ego = self.control_object
        nudge = 0.0
        for obj in surrounding_objs:
            if not isinstance(obj, Pedestrian):
                continue
            try:
                rel = ego.convert_to_local_coordinates(obj.position, ego.position)
            except Exception:
                continue
            fwd = float(rel[0])
            lat = float(rel[1])
            if fwd < 2.0 or fwd > 38.0 or abs(lat) > 9.0:
                continue
            urgency = (1.0 - min(fwd / 38.0, 1.0)) * (1.0 - min(abs(lat) / 7.0, 1.0))
            nudge += -np.sign(lat) * urgency * PED_IDM_STEER_GAIN
        nudge = float(np.clip(nudge, -0.55, 0.55))
        return float(np.clip(steering + nudge, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

    def _pedestrian_stop_brake(self, stop_dist):
        ego_speed = max(float(self.control_object.speed), 0.0)
        if ego_speed < 0.2:
            return 0.0
        stop_dist = max(float(stop_dist), 0.5)
        required = -(ego_speed * ego_speed) / (2.0 * stop_dist)
        return float(np.clip(required, -PED_IDM_MAX_STOP_BRAKE_M_S2, 0.0))

    def _lane_curve_speed_cap_kmh(self, lane, start_long):
        if lane is None:
            return float(self.NORMAL_SPEED)
        start_long = float(np.clip(start_long, 0.0, max(0.0, lane.length - 1e-3)))
        base_heading = float(lane.heading_theta_at(start_long))
        speed_caps = []
        for ds in ROUTE_CURVE_LOOKAHEAD_SAMPLES_M:
            sample_long = min(start_long + float(ds), float(lane.length) - 1e-3)
            if sample_long <= start_long + 1e-3:
                continue
            sample_heading = float(lane.heading_theta_at(sample_long))
            delta_heading = abs(float(wrap_to_pi(sample_heading - base_heading)))
            if delta_heading < 1e-3:
                continue
            radius = (sample_long - start_long) / delta_heading
            safe_speed = math.sqrt(max(radius, 1.0) * ROUTE_CURVE_LAT_ACCEL_M_S2) * 3.6
            speed_caps.append(safe_speed)
        if not speed_caps:
            return float(self.NORMAL_SPEED)
        return float(np.clip(min(speed_caps), ROUTE_CURVE_MIN_SPEED_KMH, self.NORMAL_SPEED))

    def _route_speed_cap_kmh(self, target_lane):
        ego = self.control_object
        if target_lane is None:
            return float(self.NORMAL_SPEED)

        try:
            current_long, _ = target_lane.local_coordinates(ego.position)
        except Exception:
            return float(self.NORMAL_SPEED)

        current_cap = self._lane_curve_speed_cap_kmh(target_lane, current_long)
        try:
            remaining = float(target_lane.length) - float(current_long)
        except Exception:
            remaining = ROUTE_CURVE_PREVIEW_TRIGGER_M + 1.0

        if remaining > ROUTE_CURVE_PREVIEW_TRIGGER_M:
            return current_cap

        next_lanes = list(getattr(ego.navigation, "next_ref_lanes", None) or [])
        if not next_lanes:
            return current_cap

        lane_idx = int(target_lane.index[-1]) if getattr(target_lane, "index", None) is not None else 0
        lane_idx = int(np.clip(lane_idx, 0, len(next_lanes) - 1))
        next_cap = self._lane_curve_speed_cap_kmh(next_lanes[lane_idx], 0.0)
        return min(current_cap, next_cap)

    def act(self, *args, **kwargs):
        ego = self.control_object
        cap = min(float(getattr(ego, "max_speed_km_h", EGO_MAX_SPEED_KMH)), float(self.NORMAL_SPEED))
        self.target_speed = max(float(self.CREEP_SPEED), cap)

        success = self.move_to_next_road()
        all_objects = ego.lidar.get_surrounding_objects(ego)
        try:
            if success and self.enable_lane_change:
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    ego.position,
                    max_distance=self.MAX_LONG_DIST,
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except Exception:
            acc_front_obj = None
            acc_front_dist = self.MAX_LONG_DIST
            steering_target_lane = self.routing_target_lane

        self.target_speed = min(self.target_speed, self._route_speed_cap_kmh(steering_target_lane))

        ped_plan = self._best_pedestrian_obstacle(all_objects)
        if ped_plan is not None:
            ped_obj = ped_plan["obj"]
            ped_dist = ped_plan["distance"]
            self.target_speed = min(self.target_speed, float(ped_plan["speed_cap_kmh"]))
            use_ped = acc_front_obj is None or ped_dist < float(acc_front_dist)
            if use_ped:
                acc_front_obj, acc_front_dist = ped_obj, ped_dist

        steering = self.steering_control(steering_target_lane)
        steering = self._pedestrian_steer_nudge(steering, all_objects)

        vehicle_front_obj = acc_front_obj
        vehicle_front_dist = acc_front_dist
        if isinstance(acc_front_obj, Pedestrian):
            vehicle_front_obj = None
            vehicle_front_dist = self.MAX_LONG_DIST

        dist = float(vehicle_front_dist) if vehicle_front_dist is not None else self.MAX_LONG_DIST
        acc = self.acceleration(vehicle_front_obj, dist)

        if ped_plan is not None:
            acc = min(acc, self._pedestrian_stop_brake(ped_plan["distance"]))
            if ped_plan["emergency"]:
                acc = min(acc, -3.4 - 2.8 * (1.0 - min(ped_plan["distance"], PED_IDM_EMERGENCY_BRAKE_M) / max(PED_IDM_EMERGENCY_BRAKE_M, 1e-3)))

        action = [steering, acc]
        self.action_info["action"] = action
        return action


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
        "out_of_road_done": False,
        "on_continuous_line_done": False,
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


def main():
    use_render = "--headless" not in sys.argv
    cfg = build_config(use_render=use_render)
    ped_ctl = MidRouteCrosswalkPedController()
    env = None
    try:
        env = UrbanCrosswalkEnv(cfg)
        env.reset()
        ped_ctl.reset(env)

        print(HELP_MESSAGE)
        print("Urban procedural map:", BLOCK_SEQUENCE)
        print("Ego policy: Hybrid IDM Policy")
        print("Traffic enabled: about 6-7 vehicles")
        print("Crosswalk route fractions:", CROSSWALK_ROUTE_FRACTIONS)
        ns = len(getattr(env.current_map, "pedestrian_event_sites", None) or [])
        print(f"Ped crossing sites along route: {ns}")

        episode_wall_start_s = time.perf_counter()
        ped_hit_count = 0
        veh_crash_count = 0
        prev_crash_human = False
        prev_crash_vehicle = False

        max_steps = 2500 if not use_render else 1_000_000
        for step in range(1, max_steps + 1):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            ped_ctl.update(env, step, info)
            route_pct = info.get("route_completion", 0.0) * 100.0

            ch = bool(info.get("crash_human", False))
            cv = bool(info.get("crash_vehicle", False))
            if ch and not prev_crash_human:
                ped_hit_count += 1
            if cv and not prev_crash_vehicle:
                veh_crash_count += 1
            prev_crash_human, prev_crash_vehicle = ch, cv

            wall_s = time.perf_counter() - episode_wall_start_s

            if use_render:
                env.render(
                    text={
                        "1. Policy": "Hybrid IDM Policy",
                        "2. Time taken": f"{wall_s:.2f} s",
                        "3. Pedestrians hit": str(ped_hit_count),
                        "4. Crashes (other vehicles)": str(veh_crash_count),
                    }
                )

            if terminated or truncated:
                print(
                    "Episode end:",
                    f"route={route_pct:.1f}%",
                    "arrive_dest=",
                    info.get("arrive_dest", False),
                )
                ped_ctl.cleanup(env)
                env.reset()
                ped_ctl.reset(env)
                episode_wall_start_s = time.perf_counter()
                ped_hit_count = 0
                veh_crash_count = 0
                prev_crash_human = False
                prev_crash_vehicle = False
    finally:
        if env is not None:
            try:
                ped_ctl.cleanup(env)
            except Exception:
                pass
            env.close()


if __name__ == "__main__":
    main()
