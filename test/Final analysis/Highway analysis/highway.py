#!/usr/bin/env python
"""
This script demonstrates how to setup the Safe RL environments with a 2D top-down
road map overlay displayed inside the same 3D render window (top-left corner).
START (blue) and END (red) markers are drawn on the map alongside a live green arrow
tracking the agent.

Please feel free to run this script to enjoy a journey by keyboard! Remember to press H to see help message!

Auto-Drive mode may fail to solve some scenarios due to distribution mismatch.
"""
import logging
import math

import cv2
import numpy as np
from panda3d.core import Texture, TransparencyAttrib
from direct.gui.OnscreenImage import OnscreenImage

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.top_down_renderer import draw_top_down_map_native
from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.tests.test_functionality.test_object_collision_detection import ComplexEnv

FILM_SIZE = (512, 512)   # resolution of the 2D map image
MAP_SCALE = 0.30         # size of the overlay on screen (aspect2d units)

# Colours (BGR for OpenCV)
COL_START  = (220, 80,  0)    # blue  — START marker
COL_END    = (0,   50, 220)   # red   — END marker
COL_AGENT  = (0,   200, 0)    # green — live agent arrow


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
    env = ComplexEnv(
        dict(
            use_render=True,
            manual_control=True,
            vehicle_config={"show_navi_mark": False},
            interface_panel=["dashboard"],
        )
    )

    map_surface     = None
    map_marker_bgr  = None   # base image with START/END drawn (updated each reset)
    map_tex         = None
    map_overlay     = None

    try:
        env.reset()
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True

        # Build map and draw START / END markers
        map_surface, map_base_rgb = build_map_surface(env)
        start_xy, end_xy = get_start_end_positions(env.agent)
        map_marker_bgr = build_static_map_with_markers(map_base_rgb, map_surface, start_xy, end_xy)

        # Create the Panda3D in-window overlay
        map_tex     = numpy_to_texture(map_marker_bgr)
        map_overlay = create_map_overlay(env.engine, map_tex)

        for i in range(1, 1_000_000_000):
            previous_takeover = env.current_track_agent.expert_takeover
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Total episode cost": env.episode_cost,
                    "Keyboard Control": "W,A,S,D",
                }
            )

            # Redraw only the agent arrow each step (markers stay fixed)
            frame_bgr = draw_agent_on_map(map_marker_bgr, map_surface, env.agent)
            numpy_to_texture(frame_bgr, map_tex)

            if not previous_takeover and env.current_track_agent.expert_takeover:
                logging.warning("Auto-Drive mode may fail to solve some scenarios due to distribution mismatch")

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
                # Rebuild map + markers for the new scenario
                map_surface, map_base_rgb = build_map_surface(env)
                start_xy, end_xy = get_start_end_positions(env.agent)
                map_marker_bgr = build_static_map_with_markers(map_base_rgb, map_surface, start_xy, end_xy)
                numpy_to_texture(map_marker_bgr, map_tex)

    finally:
        if map_overlay is not None:
            map_overlay.destroy()
        env.close()
