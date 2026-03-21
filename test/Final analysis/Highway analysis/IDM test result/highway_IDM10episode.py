#!/usr/bin/env python
"""
highway_IDM10episode.py
Runs 10 consecutive episodes with IDMPolicy on the highway scenario.

Per-episode tracking:
  - Crash count        : increments each time a new collision event begins.
  - Timeout            : YES if the agent is continuously stopped (speed < 1 km/h)
                         for more than 60 seconds during the episode.
  - Pass               : YES if no timeout occurred, NO otherwise.
  - Time Taken         : adaptive timer.
                        - normal mode (F not pressed): real-world elapsed time
                        - unlimited mode (F pressed): simulation elapsed time
  - Avg Speed (km/h)   : mean of per-step speeds recorded during the episode.

Results are saved to an XLSX file in the IDM test result folder.
"""
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pyautogui
from panda3d.core import Texture, TransparencyAttrib
from direct.gui.OnscreenImage import OnscreenImage

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.top_down_renderer import draw_top_down_map_native
from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.manager.traffic_manager import PGTrafficManager
from metadrive.component.vehicle.vehicle_type import random_vehicle_type as _rvt

# Raise truck (XL) proportion to ~40% of all traffic vehicles.
# Probabilities order: [S, M, L, XL, default]
def _truck_heavy_vehicle_type(self):
    return _rvt(self.np_random, [0.15, 0.20, 0.25, 0.40, 0.0])

PGTrafficManager.random_vehicle_type = _truck_heavy_vehicle_type

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EPISODES         = 10
FILM_SIZE            = (512, 512)
MAP_SCALE            = 0.30

# Stuck detection: if the vehicle hasn't moved more than STUCK_DIST_M metres
# within the last STUCK_WINDOW_SEC seconds it is counted as stuck → timeout.
# This is more robust than a speed threshold because a spinning/oscillating
# vehicle can have non-zero speed while going nowhere.
STUCK_DIST_M         = 5.0    # metres — minimum displacement to NOT be stuck
STUCK_WINDOW_SEC     = 60.0   # seconds — rolling window for displacement check
AUTO_PRESS_F         = True   # keep True to auto-enable unlimited FPS on episode 1

RESULT_DIR = (
    r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main"
    r"\test\Final analysis\Highway analysis\IDM test result"
)

# Colours (BGR for OpenCV)
COL_START  = (220, 80,  0)
COL_END    = (0,   50, 220)
COL_AGENT  = (0,   200, 0)


def format_mm_ss(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_vehicle_color_text(agent):
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
    surface = draw_top_down_map_native(env.current_map, return_surface=True, film_size=FILM_SIZE)
    base_img = WorldSurface.to_cv2_image(surface)
    return surface, base_img


def get_start_end_positions(agent):
    start_xy = agent.spawn_place
    nav      = agent.navigation
    fl       = nav.final_lane
    end_xy   = fl.position(fl.length, fl.width / 2.0)
    return start_xy, end_xy


def draw_label(img, px, py, text, bg_color, text_color=(255, 255, 255)):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x1, y1 = px - pad,      py - th - pad
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
        (end_xy,   "END",   COL_END),
    ]:
        px, py = surface.vec2pix(world_pos)
        px = int(np.clip(px, 0, img.shape[1] - 1))
        py = int(np.clip(py, 0, img.shape[0] - 1))
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 9, (50, 50, 50), 1)
        draw_label(img, px + 10, py, label, color)
    return img


def draw_agent_on_map(marker_img_bgr, surface, agent):
    img     = marker_img_bgr.copy()
    px, py  = surface.vec2pix(agent.position)
    px = int(np.clip(px, 0, img.shape[1] - 1))
    py = int(np.clip(py, 0, img.shape[0] - 1))
    heading   = agent.heading_theta
    arrow_len = 16
    ex = int(np.clip(px + arrow_len * math.cos(heading), 0, img.shape[1] - 1))
    ey = int(np.clip(py - arrow_len * math.sin(heading), 0, img.shape[0] - 1))
    cv2.circle(img, (px, py), 8, COL_AGENT, -1)
    cv2.arrowedLine(img, (px, py), (ex, ey), (0, 255, 0), 2, tipLength=0.4)
    return img


def numpy_to_texture(img_bgr, tex=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    if tex is None:
        tex = Texture("road_map")
        tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgb)
    img_flipped = np.ascontiguousarray(np.flipud(img_rgb))
    tex.setRamImageAs(img_flipped.tobytes(), "RGB")
    return tex


def create_map_overlay(engine, tex):
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


def save_results_xlsx(results: list, start_dt: datetime):
    """Save the episode results list to a timestamped XLSX file."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    filename = f"IDM_results_{start_dt.strftime('%Y%m%d_%H%M%S')}.xlsx"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        dict(
            use_render=True,
            manual_control=False,
            agent_policy=IDMPolicy,
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
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0),
            },
            interface_panel=["dashboard"],
        )
    )

    map_surface    = None
    map_marker_bgr = None
    map_tex        = None
    map_overlay    = None

    all_results    = []
    test_start_dt  = datetime.now()

    # Each env.step() repeats the physics 'decision_repeat' times at 'physics_world_step_size' each.
    # Real simulation seconds per env.step() = physics_world_step_size × decision_repeat (= 0.02 × 5 = 0.1 s).
    step_dt = (env.config.get("physics_world_step_size", 0.02)
               * env.config.get("decision_repeat", 5))

    try:
        for ep in range(1, NUM_EPISODES + 1):
            print(f"\n{'='*50}")
            print(f"  Episode {ep} / {NUM_EPISODES}  started at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*50}")

            env.reset()

            # Rebuild 2D map overlay for this episode
            map_surface, map_base_rgb = build_map_surface(env)
            start_xy, end_xy          = get_start_end_positions(env.agent)
            map_marker_bgr            = build_static_map_with_markers(
                                            map_base_rgb, map_surface, start_xy, end_xy)

            if ep == 1:
                map_tex     = numpy_to_texture(map_marker_bgr)
                map_overlay = create_map_overlay(env.engine, map_tex)
                if AUTO_PRESS_F:
                    # Speed up simulation to unlimited FPS once the window is open
                    time.sleep(1.5)
                    pyautogui.press('f')
                    print("[INFO] Pressed F — simulation running at unlimited FPS.")
                else:
                    print("[INFO] Auto speed-up disabled. Running in normal (real-time) mode.")
            else:
                numpy_to_texture(map_marker_bgr, map_tex)

            # --- Per-episode accumulators ---
            ep_crash_count   = 0
            prev_crash_state = False
            ep_speed_sum     = 0.0
            ep_speed_steps   = 0
            ep_stuck_ref_pos  = None  # position snapshot for stuck detection
            ep_stuck_ref_time = 0.0   # ep_effective_time when snapshot was taken
            ep_timeout       = False
            ep_arrive        = False
            ep_sim_time      = 0.0
            ep_dt_start      = datetime.now()

            # Reset the clock as late as possible — right before the first step —
            # so map-rebuild / overlay / F-press overhead is never counted.
            ep_effective_time = 0.0
            ep_prev_wall      = time.perf_counter()

            # --- Step loop ---
            while True:
                o, r, tm, tc, info = env.step([0, 0])

                now_wall   = time.perf_counter()
                delta_wall = max(0.0, now_wall - ep_prev_wall)
                ep_prev_wall = now_wall

                ep_sim_time = env.episode_step * step_dt
                realtime_mode = bool(getattr(env.engine.force_fps, "real_time_simulation", False))
                delta_time_for_count = delta_wall if realtime_mode else step_dt
                ep_effective_time += delta_time_for_count
                speed_kmh   = getattr(env.agent, "speed_km_h", 0.0)

                # Crash detection (rising-edge only)
                crash_now = bool(
                    info.get("crash_vehicle", False) or
                    info.get("crash_object",  False)
                )
                if crash_now and not prev_crash_state:
                    ep_crash_count += 1
                prev_crash_state = crash_now

                # Speed accumulation for average
                ep_speed_sum   += speed_kmh
                ep_speed_steps += 1

                # Stuck detection: position-based rolling window.
                # Take a position snapshot once per window; if the vehicle has
                # not moved STUCK_DIST_M metres by the end of the window → stuck.
                cur_pos = np.array(env.agent.position[:2])
                if ep_stuck_ref_pos is None:
                    ep_stuck_ref_pos  = cur_pos.copy()
                    ep_stuck_ref_time = ep_effective_time

                time_in_window = ep_effective_time - ep_stuck_ref_time
                if time_in_window >= STUCK_WINDOW_SEC:
                    dist_moved = float(np.linalg.norm(cur_pos - ep_stuck_ref_pos))
                    if dist_moved < STUCK_DIST_M and not ep_timeout:
                        ep_timeout = True
                        print(f"[Episode {ep}] TIMEOUT — moved only {dist_moved:.1f}m "
                              f"in {STUCK_WINDOW_SEC:.0f}s (stuck).")
                        tm = True
                    # Slide the window forward regardless
                    ep_stuck_ref_pos  = cur_pos.copy()
                    ep_stuck_ref_time = ep_effective_time

                # Arrival
                if info.get("arrive_dest", False):
                    ep_arrive = True

                # HUD
                env.render(
                    text={
                        "Episode":          f"{ep} / {NUM_EPISODES}",
                        "Policy":           "IDM",
                        "Crashes":          ep_crash_count,
                        "Timeout":          "YES" if ep_timeout else "NO",
                        "Pass":             "YES" if not ep_timeout else "NO",
                        "Time taken":       format_mm_ss(ep_effective_time),
                        "Tracked vehicle color":  get_vehicle_color_text(env.agent),
                        "Tracked vehicle speed":  f"{speed_kmh:.1f}",
                        "Speed up":         "Press F",
                    }
                )

                # Redraw agent arrow on minimap
                frame_bgr = draw_agent_on_map(map_marker_bgr, map_surface, env.agent)
                numpy_to_texture(frame_bgr, map_tex)

                if tm or tc:
                    break

            # --- Episode summary ---
            ep_pass     = not ep_timeout
            avg_speed   = ep_speed_sum / ep_speed_steps if ep_speed_steps > 0 else 0.0
            time_taken  = format_mm_ss(ep_effective_time)

            print(f"[Episode {ep}] Pass={ep_pass} | Timeout={ep_timeout} | "
                  f"Crashes={ep_crash_count} | Time={time_taken} | "
                  f"AvgSpeed={avg_speed:.1f} km/h | Arrived={ep_arrive}")

            all_results.append({
                "Episode":               ep,
                "Pass":                  "YES" if ep_pass  else "NO",
                "Timeout":               "YES" if ep_timeout else "NO",
                "Crash Count":           ep_crash_count,
                "Time Taken (adaptive mm:ss)": time_taken,
                "Avg Speed (km/h)":      round(avg_speed, 2),
                "Simulation Date Time":  ep_dt_start.strftime("%Y-%m-%d %H:%M:%S"),
            })

        # --- Save results after all 10 episodes ---
        save_results_xlsx(all_results, test_start_dt)

    finally:
        if map_overlay is not None:
            map_overlay.destroy()
        env.close()
