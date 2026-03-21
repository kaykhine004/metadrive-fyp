#!/usr/bin/env python
"""
This script demonstrates how to setup the Safe RL environments with TrajectoryIDMPolicy.

TrajectoryIDMPolicy: The ego vehicle follows a pre-built PointLane trajectory with IDM speed control.
Traffic vehicles use standard IDMPolicy. This allows the ego to drive along logged/planned trajectories
while controlling speed reactively (emergency brakes, maintaining distance from vehicles ahead).

Sometimes we want the replayed traffic vehicles to be reactive, as the behavior of the ego car may change
and be different from the logged one. TrajectoryIDMPolicy allows driving along trajectories but controls
speed according to IDM policy—performing emergency brakes and maintaining distance from the car in front.

The script also displays a 2D top-down road map overlay with START (blue), END (red) markers
and a live green arrow tracking the agent.
"""
import math
import time

import cv2
import numpy as np
import pyautogui
from panda3d.core import Texture, TransparencyAttrib
from direct.gui.OnscreenImage import OnscreenImage

from metadrive.constants import HELP_MESSAGE, DEFAULT_AGENT
from metadrive.engine.top_down_renderer import draw_top_down_map_native
from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.lane.point_lane import PointLane
from metadrive.manager.pg_map_manager import PGMapManager

# Spawn and destination for TrajectoryIDMPolicy (fixed route)
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
    # Remove duplicate consecutive points
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
        if self.current_map is not None:
            # Get destination node: last block's socket 0 exit (forward path from spawn)
            last_block = self.current_map.blocks[-1]
            sockets = last_block.get_socket_list()
            if sockets:
                socket = sockets[0]
                dest_node = socket.positive_road.end_node
                self.current_sdc_route = _build_trajectory_point_lane(
                    self.current_map.road_network, SPAWN_LANE_INDEX, dest_node
                )
                # Ensure agent_configs uses same destination for navigation (aligned with trajectory)
                if DEFAULT_AGENT in self.engine.global_config.get("agent_configs", {}):
                    self.engine.global_config["agent_configs"][DEFAULT_AGENT]["destination"] = dest_node
            else:
                self.current_sdc_route = None

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
    STUCK_SPEED_KMH  = 2.0   # below this → may be stuck
    STUCK_STEPS      = 30    # consecutive low-speed steps before switching to IDM
    RESUME_SPEED_KMH = 8.0   # above this → try to return to trajectory
    RESUME_STEPS     = 50    # consecutive good-speed steps before leaving IDM mode

    def __init__(self, control_object, random_seed, traj_to_follow, policy_index=None):
        super().__init__(control_object, random_seed, traj_to_follow, policy_index)
        self._stuck_steps    = 0
        self._moving_steps   = 0
        self._avoid_mode     = False

    def act(self, agent_id=None):
        speed_kmh = getattr(self.control_object, "speed_km_h", 0.0)

        # Track stuck / moving counters
        if speed_kmh < self.STUCK_SPEED_KMH:
            self._stuck_steps  += 1
            self._moving_steps  = 0
        else:
            self._moving_steps += 1
            self._stuck_steps   = 0

        # Enter obstacle-avoidance (IDM lane-change) mode when stuck
        if not self._avoid_mode and self._stuck_steps >= self.STUCK_STEPS:
            self._avoid_mode = True
            # Hand routing over to the road-network lane so IDM can lane-change
            self.routing_target_lane = self.control_object.lane
            self.enable_lane_change  = True

        # Return to trajectory mode once moving well again
        if self._avoid_mode and self._moving_steps >= self.RESUME_STEPS:
            self._avoid_mode = False
            self.routing_target_lane = self.traj_to_follow
            self.enable_lane_change  = False
            self._stuck_steps        = 0

        if self._avoid_mode:
            # Use IDMPolicy's full lane-change act()
            return IDMPolicy.act(self)
        else:
            # Use TrajectoryIDMPolicy's trajectory-following act()
            return super().act(True)   # True = do_speed_control

    def reset(self):
        super().reset()
        self._stuck_steps  = 0
        self._moving_steps = 0
        self._avoid_mode   = False
        # Always start in trajectory mode
        self.routing_target_lane = self.traj_to_follow
        self.enable_lane_change  = False

    @property
    def mode_label(self):
        return "IDM (avoid)" if self._avoid_mode else "Trajectory"


class SafeMetaDriveEnvTrajectoryIDM(SafeMetaDriveEnv):
    """SafeMetaDriveEnv with ObstacleAwareTrajectoryIDMPolicy for the agent."""
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", HighwayTrajectoryIDMMapManager())

FILM_SIZE = (512, 512)   # resolution of the 2D map image
MAP_SCALE = 0.30         # size of the overlay on screen (aspect2d units)

# Colours (BGR for OpenCV)
COL_START  = (220, 80,  0)    # blue  — START marker
COL_END    = (0,   50, 220)   # red   — END marker
COL_AGENT  = (0,   200, 0)    # green — live agent arrow


def format_mm_ss(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_vehicle_color_text(agent):
    """
    Return the visual 3D color of the vehicle by reading its model path.
    panda_color is NOT used — it is the minimap identifier color (always green
    for the tracked agent) and does not reflect the actual 3D appearance.
    """
    MODEL_COLORS = {
        "ferra":  "red",
        "beetle": "cream",
        "130":    "silver",
        "lada":   "silver",
        "truck":  "grey",
    }
    try:
        model_path = agent.path[0].lower()
        for key, color_name in MODEL_COLORS.items():
            if key in model_path:
                return color_name
    except Exception:
        pass
    return "unknown"


def build_map_surface(env):
    """Generate the 2D top-down schematic of the current map."""
    surface = draw_top_down_map_native(env.current_map, return_surface=True, film_size=FILM_SIZE)
    base_img = WorldSurface.to_cv2_image(surface)  # RGB numpy array (H, W, 3)
    return surface, base_img


def get_start_end_positions(agent):
    """
    Return (start_xy, end_xy) world coordinates for the current episode.
    start_xy : agent spawn position
    end_xy   : end of the navigation's final lane (centre)
    """
    start_xy = agent.spawn_place  # (x, y)

    nav = agent.navigation
    fl = nav.final_lane
    # mid-lateral of final lane = halfway across all lanes
    half_width = fl.width / 2.0
    end_xy = fl.position(fl.length, half_width)
    return start_xy, end_xy


def draw_label(img, px, py, text, bg_color, text_color=(255, 255, 255)):
    """Draw a filled rounded rectangle label at (px, py)."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x1, y1 = px - pad,        py - th - pad
    x2, y2 = px + tw + pad,   py + baseline + pad
    # clamp inside image
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (px, py), font, font_scale, text_color, thickness, cv2.LINE_AA)


def build_static_map_with_markers(base_img, surface, start_xy, end_xy):
    """
    Draw START and END markers on a copy of the base map.
    This image is stored once per episode — only the agent arrow is redrawn each step.
    """
    img = base_img.copy()

    # Convert BGR base map to BGR (base_img is RGB from WorldSurface)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for world_pos, label, color in [
        (start_xy, "START", COL_START),
        (end_xy,   "END",   COL_END),
    ]:
        px, py = surface.vec2pix(world_pos)
        px = int(np.clip(px, 0, img.shape[1] - 1))
        py = int(np.clip(py, 0, img.shape[0] - 1))
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 9, (50, 50, 50), 1)    # thin dark outline
        draw_label(img, px + 10, py, label, color)

    return img   # BGR numpy array


def draw_agent_on_map(marker_img_bgr, surface, agent):
    """
    Draw a live green arrow at the agent's current position on top of the
    pre-rendered marker image (BGR).  Returns a BGR image.
    """
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
    return img   # BGR


def numpy_to_texture(img_bgr, tex=None):
    """Load a BGR numpy image into a Panda3D Texture (convert to RGB, flip rows)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if tex is None:
        tex = Texture("road_map")
        tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgb)
    img_flipped = np.ascontiguousarray(np.flipud(img_rgb))
    tex.setRamImageAs(img_flipped.tobytes(), "RGB")
    return tex


def create_map_overlay(engine, tex):
    """
    Pin the map texture as an OnscreenImage in the top-left of the render window.
    aspect2d coords: x = -aspect..+aspect, z = -1..+1
    """
    aspect = engine.getAspectRatio()
    pos_x  = -aspect + MAP_SCALE + 0.02
    pos_z  =  1.0    - MAP_SCALE - 0.02

    osi = OnscreenImage(
        image=tex,
        pos=(pos_x, 0, pos_z),
        scale=MAP_SCALE,
        parent=engine.aspect2d,
    )
    osi.setTransparency(TransparencyAttrib.MAlpha)
    return osi


if __name__ == "__main__":
    env = SafeMetaDriveEnvTrajectoryIDM(
        dict(
            use_render=True,
            manual_control=False,
            agent_policy=ObstacleAwareTrajectoryIDMPolicy,
            # Stable route: ramps + split/merge + tollgate (f/F fork blocks are broken in this version).
            map="CSrRSY$yRSCR",
            num_scenarios=1,
            start_seed=5,
            # Light mixed traffic (includes trucks via built-in vehicle pool).
            traffic_density=0.05,
            need_inverse_traffic=True,
            random_traffic=True,
            # Construction/accident events at some map blocks.
            accident_prob=0.25,
            static_traffic_object=True,
            horizon=5000,
            crash_object_done=False,
            out_of_road_done=False,
            random_spawn_lane_index=False,
            vehicle_config={
                "show_navi_mark": False,
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
            },
            interface_panel=["dashboard"],
        )
    )

    map_surface     = None
    map_marker_bgr  = None   # base image with START/END drawn (updated each reset)
    map_tex         = None
    map_overlay     = None
    tracked_crashes = 0
    prev_crash_state = False

    try:
        env.reset()
        print(HELP_MESSAGE)

        # Build map and draw START / END markers
        map_surface, map_base_rgb = build_map_surface(env)
        start_xy, end_xy = get_start_end_positions(env.agent)
        map_marker_bgr = build_static_map_with_markers(map_base_rgb, map_surface, start_xy, end_xy)

        # Create the Panda3D in-window overlay
        map_tex     = numpy_to_texture(map_marker_bgr)
        map_overlay = create_map_overlay(env.engine, map_tex)

        # Auto-press F to enable unlimited FPS (speed up simulation)
        time.sleep(1.5)
        pyautogui.press('f')
        print("[INFO] Pressed F — simulation running at unlimited FPS.")

        for i in range(1, 1_000_000_000):
            o, r, tm, tc, info = env.step([0, 0])
            crash_now = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
            if crash_now and not prev_crash_state:
                tracked_crashes += 1
            prev_crash_state = crash_now

            sim_time_sec = env.episode_step * env.config["physics_world_step_size"]
            tracked_color = get_vehicle_color_text(env.agent)
            speed_kmh = getattr(env.agent, "speed_km_h", 0.0)
            policy = env.engine.get_policy(env.agent.id)
            mode_label = policy.mode_label if hasattr(policy, "mode_label") else "Trajectory"

            env.render(
                text={
                    "Policy": "TrajectoryIDM + lane-change",
                    "Mode": mode_label,
                    "Crashes": tracked_crashes,
                    "Simulation time": format_mm_ss(sim_time_sec),
                    "Tracked vehicle color": tracked_color,
                    "Tracked vehicle speed": f"{speed_kmh:.1f}",
                    "Speed up": "Press F",
                }
            )

            # Redraw only the agent arrow each step (markers stay fixed)
            frame_bgr = draw_agent_on_map(map_marker_bgr, map_surface, env.agent)
            numpy_to_texture(frame_bgr, map_tex)

            if tm or tc:
                if info["arrive_dest"]:
                    print("Arrived at destination!")
                else:
                    print("Episode ended early — resetting...")
                env.reset()
                tracked_crashes = 0
                prev_crash_state = False
                # Rebuild map + markers for the new scenario
                map_surface, map_base_rgb = build_map_surface(env)
                start_xy, end_xy = get_start_end_positions(env.agent)
                map_marker_bgr = build_static_map_with_markers(map_base_rgb, map_surface, start_xy, end_xy)
                numpy_to_texture(map_marker_bgr, map_tex)

    finally:
        if map_overlay is not None:
            map_overlay.destroy()
        env.close()
