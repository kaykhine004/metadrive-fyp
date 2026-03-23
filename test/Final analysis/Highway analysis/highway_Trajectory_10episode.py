#!/usr/bin/env python

import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pyautogui
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import Texture, TransparencyAttrib

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, HELP_MESSAGE
from metadrive.engine.top_down_renderer import draw_top_down_map_native
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy

SPAWN_LANE_INDEX = (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0)


def _build_trajectory_point_lane(road_network, spawn_lane_index, dest_node, lane_width=2.0):
    """
    Build a PointLane from spawn to destination by concatenating lane polylines along the shortest path.
    Used by TrajectoryIDMPolicy to follow a pre-planned trajectory with IDM speed control.
    """
    path = road_network.shortest_path(spawn_lane_index, dest_node)
    if len(path) < 2:
        return None
    polylines = []
    for i in range(len(path) - 1):
        from_node, to_node = path[i], path[i + 1]
        lanes = road_network.graph.get(from_node, {}).get(to_node, [])
        if lanes:
            poly = lanes[0].get_polyline(interval=2, lateral=0)
            polylines.append(np.array(poly)[..., :2])
    if not polylines:
        return None
    path_points = np.concatenate(polylines, axis=0)
    keep = np.ones(len(path_points), dtype=bool)
    keep[1:] = np.any(path_points[1:] != path_points[:-1], axis=1)
    path_points = path_points[keep]
    if len(path_points) < 2:
        return None
    return PointLane(path_points, lane_width)


class HighwayTrajectoryIDMMapManager(PGMapManager):
    """
    Map manager that builds current_sdc_route (PointLane trajectory) for TrajectoryIDMPolicy.
    The destination is set to the last block's exit (forward direction from spawn).
    """

    def __init__(self):
        super().__init__()
        self.current_sdc_route = None

    def reset(self):
        super().reset()
        if self.current_map is None:
            return

        last_block = self.current_map.blocks[-1]
        sockets = last_block.get_socket_list()
        if not sockets:
            self.current_sdc_route = None
            return

        socket = sockets[0]
        dest_node = socket.positive_road.end_node
        self.current_sdc_route = _build_trajectory_point_lane(
            self.current_map.road_network, SPAWN_LANE_INDEX, dest_node
        )
        if DEFAULT_AGENT in self.engine.global_config.get("agent_configs", {}):
            self.engine.global_config["agent_configs"][DEFAULT_AGENT]["destination"] = dest_node

    def unload_map(self, map):
        self.current_sdc_route = None
        super().unload_map(map)


class ObstacleAwareTrajectoryIDMPolicy(TrajectoryIDMPolicy):
    """
    TrajectoryIDMPolicy extended with lane-change capability.

    Normal mode   : follows the pre-built PointLane trajectory with IDM speed control.
    Avoid mode    : when stuck (speed < STUCK_SPEED_KMH for STUCK_STEPS steps),
                    switches to IDMPolicy lane-change logic to go around the obstacle.
    Resume mode   : returns to trajectory following once the vehicle is moving again
                    (speed > RESUME_SPEED_KMH for RESUME_STEPS steps).
    """

    STUCK_SPEED_KMH = 2.0
    STUCK_STEPS = 30
    RESUME_SPEED_KMH = 8.0
    RESUME_STEPS = 50

    def __init__(self, control_object, random_seed, traj_to_follow, policy_index=None):
        super().__init__(control_object, random_seed, traj_to_follow, policy_index)
        self._stuck_steps = 0
        self._moving_steps = 0
        self._avoid_mode = False

    def act(self, agent_id=None):
        speed_kmh = getattr(self.control_object, "speed_km_h", 0.0)

        if speed_kmh < self.STUCK_SPEED_KMH:
            self._stuck_steps += 1
            self._moving_steps = 0
        else:
            self._moving_steps += 1
            self._stuck_steps = 0

        if not self._avoid_mode and self._stuck_steps >= self.STUCK_STEPS:
            self._avoid_mode = True
            self.routing_target_lane = self.control_object.lane
            self.enable_lane_change = True

        if self._avoid_mode and self._moving_steps >= self.RESUME_STEPS:
            self._avoid_mode = False
            self.routing_target_lane = self.traj_to_follow
            self.enable_lane_change = False
            self._stuck_steps = 0

        if self._avoid_mode:
            return IDMPolicy.act(self)
        return super().act(True)

    def reset(self):
        super().reset()
        self._stuck_steps = 0
        self._moving_steps = 0
        self._avoid_mode = False
        self.routing_target_lane = self.traj_to_follow
        self.enable_lane_change = False

    @property
    def mode_label(self):
        return "IDM (avoid)" if self._avoid_mode else "Trajectory"


class SafeMetaDriveEnvTrajectoryIDM(SafeMetaDriveEnv):
    """SafeMetaDriveEnv with ObstacleAwareTrajectoryIDMPolicy for the agent."""

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", HighwayTrajectoryIDMMapManager())


NUM_EPISODES = 10
FILM_SIZE = (512, 512)
MAP_SCALE = 0.30
STUCK_DIST_M = 5.0
STUCK_WINDOW_SEC = 60.0
AUTO_PRESS_F = True

RESULT_DIR = (
    r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main"
    r"\test\Final analysis\Highway analysis\Trajectory test result"
)

COL_START = (220, 80, 0)
COL_END = (0, 50, 220)
COL_AGENT = (0, 200, 0)


def format_mm_ss(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_vehicle_color_text(agent):
    model_colors = {
        "ferra": "red",
        "beetle": "cream",
        "130": "silver",
        "lada": "silver",
        "truck": "grey",
    }
    try:
        model_path = agent.path[0].lower()
        for key, color_name in model_colors.items():
            if key in model_path:
                return color_name
    except Exception:
        pass
    return "unknown"


def build_map_surface(env):
    surface = draw_top_down_map_native(env.current_map, return_surface=True, film_size=FILM_SIZE)
    base_img = WorldSurface.to_cv2_image(surface)
    return surface, base_img


def get_start_end_positions(agent):
    start_xy = agent.spawn_place
    final_lane = agent.navigation.final_lane
    half_width = final_lane.width / 2.0
    end_xy = final_lane.position(final_lane.length, half_width)
    return start_xy, end_xy


def draw_label(img, px, py, text, bg_color, text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x1, y1 = px - pad, py - th - pad
    x2, y2 = px + tw + pad, py + baseline + pad
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (px, py), font, font_scale, text_color, thickness, cv2.LINE_AA)


def build_static_map_with_markers(base_img, surface, start_xy, end_xy):
    img = cv2.cvtColor(base_img.copy(), cv2.COLOR_RGB2BGR)
    for world_pos, label, color in [
        (start_xy, "START", COL_START),
        (end_xy, "END", COL_END),
    ]:
        px, py = surface.vec2pix(world_pos)
        px = int(np.clip(px, 0, img.shape[1] - 1))
        py = int(np.clip(py, 0, img.shape[0] - 1))
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 9, (50, 50, 50), 1)
        draw_label(img, px + 10, py, label, color)
    return img


def draw_agent_on_map(marker_img_bgr, surface, agent):
    img = marker_img_bgr.copy()
    px, py = surface.vec2pix(agent.position)
    px = int(np.clip(px, 0, img.shape[1] - 1))
    py = int(np.clip(py, 0, img.shape[0] - 1))

    heading = agent.heading_theta
    arrow_len = 16
    ex = int(np.clip(px + arrow_len * math.cos(heading), 0, img.shape[1] - 1))
    ey = int(np.clip(py - arrow_len * math.sin(heading), 0, img.shape[0] - 1))

    cv2.circle(img, (px, py), 8, COL_AGENT, -1)
    cv2.arrowedLine(img, (px, py), (ex, ey), (0, 255, 0), 2, tipLength=0.4)
    return img


def numpy_to_texture(img_bgr, tex=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    if tex is None:
        tex = Texture("road_map")
        tex.setup2dTexture(width, height, Texture.TUnsignedByte, Texture.FRgb)
    img_flipped = np.ascontiguousarray(np.flipud(img_rgb))
    tex.setRamImageAs(img_flipped.tobytes(), "RGB")
    return tex


def create_map_overlay(engine, tex):
    aspect = engine.getAspectRatio()
    pos_x = -aspect + MAP_SCALE + 0.02
    pos_z = 1.0 - MAP_SCALE - 0.02

    overlay = OnscreenImage(
        image=tex,
        pos=(pos_x, 0, pos_z),
        scale=MAP_SCALE,
        parent=engine.aspect2d,
    )
    overlay.setTransparency(TransparencyAttrib.MAlpha)
    return overlay


def save_results_xlsx(results: list, start_dt: datetime):
    os.makedirs(RESULT_DIR, exist_ok=True)
    filename = f"Trajectory_results_{start_dt.strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join(RESULT_DIR, filename)
    df = pd.DataFrame(results, columns=[
        "Episode",
        "Pass",
        "Timeout",
        "Crash Count",
        "Time Taken (adaptive mm:ss)",
        "Avg Speed (km/h)",
        "Simulation Date Time",
    ])
    df.to_excel(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    env = SafeMetaDriveEnvTrajectoryIDM(
        dict(
            use_render=True,
            manual_control=False,
            agent_policy=ObstacleAwareTrajectoryIDMPolicy,
            map="CSrRSY$yRSCR",
            num_scenarios=1,
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
            vehicle_config={
                "show_navi_mark": False,
                "spawn_lane_index": SPAWN_LANE_INDEX,
            },
            interface_panel=["dashboard"],
        )
    )

    map_overlay = None
    all_results = []
    test_start_dt = datetime.now()
    step_dt = env.config.get("physics_world_step_size", 0.02) * env.config.get("decision_repeat", 5)

    try:
        print(HELP_MESSAGE)

        for ep in range(1, NUM_EPISODES + 1):
            print(f"\n{'=' * 50}")
            print(f"  Episode {ep} / {NUM_EPISODES}  started at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'=' * 50}")

            env.reset()
            map_surface, map_base_rgb = build_map_surface(env)
            start_xy, end_xy = get_start_end_positions(env.agent)
            map_marker_bgr = build_static_map_with_markers(map_base_rgb, map_surface, start_xy, end_xy)

            if ep == 1:
                map_tex = numpy_to_texture(map_marker_bgr)
                map_overlay = create_map_overlay(env.engine, map_tex)
                if AUTO_PRESS_F:
                    time.sleep(1.5)
                    pyautogui.press("f")
                    print("[INFO] Pressed F - simulation running at unlimited FPS.")
            else:
                numpy_to_texture(map_marker_bgr, map_tex)

            ep_crash_count = 0
            prev_crash_state = False
            ep_speed_sum = 0.0
            ep_speed_steps = 0
            ep_stuck_ref_pos = None
            ep_stuck_ref_time = 0.0
            ep_timeout = False
            ep_arrive = False
            ep_effective_time = 0.0
            ep_prev_wall = time.perf_counter()
            ep_dt_start = datetime.now()

            while True:
                _, _, tm, tc, info = env.step([0, 0])

                now_wall = time.perf_counter()
                delta_wall = max(0.0, now_wall - ep_prev_wall)
                ep_prev_wall = now_wall

                realtime_mode = bool(getattr(env.engine.force_fps, "real_time_simulation", False))
                ep_effective_time += delta_wall if realtime_mode else step_dt
                speed_kmh = getattr(env.agent, "speed_km_h", 0.0)

                crash_now = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
                if crash_now and not prev_crash_state:
                    ep_crash_count += 1
                prev_crash_state = crash_now

                ep_speed_sum += speed_kmh
                ep_speed_steps += 1

                cur_pos = np.array(env.agent.position[:2])
                if ep_stuck_ref_pos is None:
                    ep_stuck_ref_pos = cur_pos.copy()
                    ep_stuck_ref_time = ep_effective_time

                if ep_effective_time - ep_stuck_ref_time >= STUCK_WINDOW_SEC:
                    dist_moved = float(np.linalg.norm(cur_pos - ep_stuck_ref_pos))
                    if dist_moved < STUCK_DIST_M and not ep_timeout:
                        ep_timeout = True
                        print(
                            f"[Episode {ep}] TIMEOUT - moved only {dist_moved:.1f}m "
                            f"in {STUCK_WINDOW_SEC:.0f}s (stuck)."
                        )
                        tm = True
                    ep_stuck_ref_pos = cur_pos.copy()
                    ep_stuck_ref_time = ep_effective_time

                if info.get("arrive_dest", False):
                    ep_arrive = True

                policy = env.engine.get_policy(env.agent.id)
                mode_label = policy.mode_label if hasattr(policy, "mode_label") else "Trajectory"
                env.render(
                    text={
                        "Episode": f"{ep} / {NUM_EPISODES}",
                        "Policy": "TrajectoryIDM + lane-change",
                        "Mode": mode_label,
                        "Crashes": ep_crash_count,
                        "Timeout": "YES" if ep_timeout else "NO",
                        "Pass": "YES" if not ep_timeout else "NO",
                        "Time taken": format_mm_ss(ep_effective_time),
                        "Tracked vehicle color": get_vehicle_color_text(env.agent),
                        "Tracked vehicle speed": f"{speed_kmh:.1f}",
                        "Speed up": "Press F",
                    }
                )

                frame_bgr = draw_agent_on_map(map_marker_bgr, map_surface, env.agent)
                numpy_to_texture(frame_bgr, map_tex)

                if tm or tc:
                    break

            ep_pass = not ep_timeout
            avg_speed = ep_speed_sum / ep_speed_steps if ep_speed_steps > 0 else 0.0
            time_taken = format_mm_ss(ep_effective_time)

            print(
                f"[Episode {ep}] Pass={ep_pass} | Timeout={ep_timeout} | "
                f"Crashes={ep_crash_count} | Time={time_taken} | "
                f"AvgSpeed={avg_speed:.1f} km/h | Arrived={ep_arrive}"
            )

            all_results.append({
                "Episode": ep,
                "Pass": "YES" if ep_pass else "NO",
                "Timeout": "YES" if ep_timeout else "NO",
                "Crash Count": ep_crash_count,
                "Time Taken (adaptive mm:ss)": time_taken,
                "Avg Speed (km/h)": round(avg_speed, 2),
                "Simulation Date Time": ep_dt_start.strftime("%Y-%m-%d %H:%M:%S"),
            })

        save_results_xlsx(all_results, test_start_dt)

    finally:
        if map_overlay is not None:
            map_overlay.destroy()
        env.close()
