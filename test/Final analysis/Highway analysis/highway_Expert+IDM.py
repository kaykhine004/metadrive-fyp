#!/usr/bin/env python

import math

import cv2
import numpy as np
from panda3d.core import Texture, TransparencyAttrib
from direct.gui.OnscreenImage import OnscreenImage

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.top_down_renderer import draw_top_down_map_native
from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.engine.logger import get_logger

logger = get_logger()


class StuckAwareExpertPolicy(ExpertPolicy):

    # Fast tier — triggers ~1 s after vehicle starts slowing sharply
    SLOW_SPEED_KMH       = 8.0   # km/h — below this = approaching/stopped
    SLOW_CONSECUTIVE_STEPS = 10  # ~1 s at 0.1 s/step

    # Slow tier — position-based fallback
    STUCK_DIST_M         = 3.0   # metres — min movement per window
    STUCK_WINDOW_STEPS   = 30    # ~3 s at 0.1 s/step

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self._slow_count     = 0        # consecutive slow steps
        self._stuck_ref_pos  = None
        self._stuck_ref_step = 0

    def _get_fallback_policy(self):
        if self._fallback_policy is None:
            self._fallback_policy = IDMPolicy(
                self.control_object, random_seed=self.random_seed
            )
            logger.warning("StuckAwareExpertPolicy: obstacle detected — IDM taking over.")
        return self._fallback_policy

    def _trigger_idm(self, reason):
        self._use_fallback = True
        logger.warning("StuckAwareExpertPolicy: %s — switching to IDM.", reason)
        return self._get_fallback_policy().act()

    def act(self, agent_id=None):
        if self._use_fallback:
            return self._get_fallback_policy().act(agent_id)

        try:
            speed_kmh = getattr(self.control_object, "speed_km_h", 0.0)
            cur_pos   = np.array(self.control_object.position[:2])
            step      = self.episode_step

            # ── Tier 1: fast speed-drop check ──────────────────────────────
            if speed_kmh < self.SLOW_SPEED_KMH:
                self._slow_count += 1
            else:
                self._slow_count = 0

            if self._slow_count >= self.SLOW_CONSECUTIVE_STEPS:
                return self._trigger_idm(
                    f"speed {speed_kmh:.1f} km/h < {self.SLOW_SPEED_KMH} for "
                    f"{self._slow_count} steps"
                )

            # ── Tier 2: slow position-based check ──────────────────────────
            if self._stuck_ref_pos is None:
                self._stuck_ref_pos  = cur_pos.copy()
                self._stuck_ref_step = step

            if step - self._stuck_ref_step >= self.STUCK_WINDOW_STEPS:
                dist = float(np.linalg.norm(cur_pos - self._stuck_ref_pos))
                if dist < self.STUCK_DIST_M:
                    return self._trigger_idm(
                        f"moved only {dist:.1f} m in {self.STUCK_WINDOW_STEPS} steps"
                    )
                self._stuck_ref_pos  = cur_pos.copy()
                self._stuck_ref_step = step

        except Exception:
            pass

        return super().act(agent_id)

    def reset(self):
        super().reset()
        self._slow_count     = 0
        self._stuck_ref_pos  = None
        self._stuck_ref_step = 0

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
    env = SafeMetaDriveEnv(
        dict(
            use_render=True,
            manual_control=False,
            agent_policy=StuckAwareExpertPolicy,
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

        for i in range(1, 1_000_000_000):
            o, r, tm, tc, info = env.step([0, 0])
            crash_now = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
            if crash_now and not prev_crash_state:
                tracked_crashes += 1
            prev_crash_state = crash_now

            sim_time_sec = env.episode_step * env.config["physics_world_step_size"] * env.config.get("decision_repeat", 5)
            tracked_color = get_vehicle_color_text(env.agent)
            speed_kmh = getattr(env.agent, "speed_km_h", 0.0)

            env.render(
                text={
                    "Policy": "Expert (PPO+IDM fallback)",
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
