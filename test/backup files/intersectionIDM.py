#!/usr/bin/env python
"""
Roundabout + Intersection single-route simulation driven by IDMPolicy.

Fixed route (one direction only):
    [Start] ──► [Intersection] ──► [Straight] ──► [Roundabout] ──► [End]

All vehicles spawn at the intersection entry and always drive to the same
roundabout exit — no random route switching.

Block chain:
    Block 0 : FirstPGBlock  (entry straight — spawn point)
    Block 1 : InterSection  (4-way junction)
    Block 2 : Straight      (connector road between the two junctions)
    Block 3 : Roundabout    (circular junction — destination)
"""
import time
import os

import cv2
import numpy as np
import pyautogui
from direct.gui.OnscreenImage import OnscreenImage
from openpyxl import Workbook
from panda3d.core import Texture

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pg_space import Parameter
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config

# ---------------------------------------------------------------------------
# Single-route config
#   spawn_roads  : only the intersection entry — all vehicles start here
#   end_node     : fixed roundabout exit — all vehicles always go here
# ---------------------------------------------------------------------------
MA_CONFIG = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
    ],
    end_node=Roundabout.node(3, 1, 3),
    num_agents=10,
    horizon=10000,
    out_of_road_done=False,
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=120,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=220,
    crash_done=False,
    allow_respawn=False,
    agent_policy=IDMPolicy,
    agent_configs={},
    vehicle_config=dict(
        random_color=True,
        show_navi_mark=True,
        show_dest_mark=True,
        show_line_to_dest=True,
    ),
    interface_panel=["dashboard"],
)


# ---------------------------------------------------------------------------
# Custom map: FirstPGBlock → InterSection → Straight → Roundabout
# ---------------------------------------------------------------------------
class RoundaboutIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, \
            "Map is not empty — create a new map instead of reusing this one."

        # Block 0: entry straight (provides FirstPGBlock.NODE_2 → NODE_3 spawn road)
        first_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length,
        )
        self.blocks.append(first_block)

        # Block 1: 4-way intersection
        InterSection.EXIT_PART_LENGTH = length
        intersection_block = InterSection(
            1,
            first_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
        )
        if self.config["lane_num"] > 1:
            intersection_block.enable_u_turn(True)
        else:
            intersection_block.enable_u_turn(False)
        intersection_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(intersection_block)

        # Block 2: straight connector between intersection and roundabout
        straight_block = Straight(
            2,
            intersection_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
        )
        straight_block.construct_from_config(
            {Parameter.length: length},
            parent_node_path,
            physics_world,
        )
        self.blocks.append(straight_block)

        # Block 3: roundabout
        Roundabout.EXIT_PART_LENGTH = length
        roundabout_block = Roundabout(
            3,
            straight_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
        )
        roundabout_block.construct_block(
            parent_node_path,
            physics_world,
            extra_config={
                "exit_radius": 10,
                "inner_radius": 30,
                "angle": 70,
            },
        )
        self.blocks.append(roundabout_block)


# ---------------------------------------------------------------------------
# Map manager
# ---------------------------------------------------------------------------
class RoundaboutIntersectionMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(
                RoundaboutIntersectionMap,
                map_config=config["map_config"],
                random_seed=None,
            )
        else:
            assert len(self.spawned_objects) == 1, \
                "Expected exactly one map in this manager."
            _map = list(self.spawned_objects.values())[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


# ---------------------------------------------------------------------------
# Spawn manager — always sends every vehicle to the same fixed end node
# ---------------------------------------------------------------------------
class RoundaboutIntersectionSpawnManager(SpawnManager):
    def update_destination_for(self, agent_id, vehicle_config):
        # All vehicles share the same fixed destination (roundabout exit arm)
        vehicle_config["destination"] = self.engine.global_config["end_node"]
        return vehicle_config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MultiAgentRoundaboutIntersectionEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MA_CONFIG, allow_add_new_key=True)

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", RoundaboutIntersectionMapManager())
        self.engine.update_manager("spawn_manager", RoundaboutIntersectionSpawnManager())


# ---------------------------------------------------------------------------
# Main — track only agent0, show stats HUD
# ---------------------------------------------------------------------------
TOTAL_EPISODES = 10
STOP_TIMEOUT_SEC = 60.0
STOP_SPEED_KMH = 0.5

TD_SCREEN = (450, 450)
TD_SCALING = 1.0
TD_FILM = (800, 800)


def _world_to_screen(wx, wy, renderer):
    """Convert world coordinates (x, y) to screen pixel coordinates."""
    canvas = renderer._frame_canvas
    # world -> film pixel
    fx, fy = canvas.pos2pix(wx, wy)
    # film center crop offset (center_on_map mode)
    film_w, film_h = canvas.get_size()
    scr_w, scr_h = renderer._screen_canvas.get_size()
    off_x = film_w / 2 - scr_w / 2
    off_y = film_h / 2 - scr_h / 2
    return int(fx - off_x), int(fy - off_y)


def _draw_markers(img, renderer, vehicle_pos, start_pos, end_pos):
    """Draw START, END, and current-vehicle markers on the top-down image."""
    out = img.copy()

    # Start label (small text, no big dot)
    sx, sy = _world_to_screen(*start_pos, renderer)
    cv2.putText(out, "START", (sx - 20, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "START", (sx - 20, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    # End label (small text, no big dot)
    ex, ey = _world_to_screen(*end_pos, renderer)
    cv2.putText(out, "END", (ex - 14, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "END", (ex - 14, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    # Current vehicle marker (small red dot)
    vx, vy = _world_to_screen(*vehicle_pos, renderer)
    cv2.circle(out, (vx, vy), 5, (0, 0, 255), -1)
    cv2.circle(out, (vx, vy), 6, (0, 0, 0), 1)

    return out


def _numpy_to_panda_texture(img_bgr, tex=None):
    """Convert a BGR numpy image (from TopDownRenderer) into a Panda3D Texture."""
    h, w = img_bgr.shape[:2]
    img_rgba = np.concatenate(
        [img_bgr[..., ::-1], np.full((h, w, 1), 255, dtype=np.uint8)], axis=2
    )
    img_flipped = np.flipud(img_rgba).copy()
    if tex is None:
        tex = Texture("topdown")
        tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.setRamImage(img_flipped.tobytes())
    return tex


def _end_pos_from_map(env, fallback):
    """Get destination marker world position from destination lane."""
    dest_road = env.current_map.road_network.graph.get(
        Roundabout.node(3, 1, 2), {}
    ).get(Roundabout.node(3, 1, 3), None)
    if dest_road:
        lane = dest_road[0]
        return tuple(lane.position(lane.length, 0))
    return fallback


def _save_results_xlsx(rows):
    """Save episode summary rows to an xlsx file."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main\test\Intersection analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"intersectionIDM_results_{ts}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "intersectionIDM"
    ws.append([
        "episode",
        "Pass",
        "Timeout",
        "crash_count",
        "time_taken_to_reach_goal_sec",
        "speed_avg_kmh",
        "simulation_datetime",
    ])
    for row in rows:
        ws.append([
            row["episode"],
            row["Pass"],
            row["Timeout"],
            row["crash_count"],
            row["time_taken_to_reach_goal_sec"],
            row["speed_avg_kmh"],
            row["simulation_datetime"],
        ])
    wb.save(output_path)
    return output_path


def _set_vehicle_visuals_and_track(env, target_id):
    """Give target a unique red color and track it with main camera."""
    for agent_id, vehicle in env.agents.items():
        if agent_id == target_id:
            vehicle.origin.setColorScale(1.0, 0.05, 0.05, 1.0)  # strong red
        else:
            # Keep other cars in their own random colors.
            vehicle.origin.setColorScale(1.0, 1.0, 1.0, 1.0)
    if target_id in env.agents and env.main_camera is not None:
        env.main_camera.track(env.agents[target_id])


if __name__ == "__main__":
    env = MultiAgentRoundaboutIntersectionEnv({"use_render": True})
    try:
        overlay = None
        td_tex = None
        results = []
        fast_mode_enabled = False
        sim_step_sec = env.config["physics_world_step_size"] * env.config["decision_repeat"]

        # Use 3rd start position by default; change to "agent1" for 2nd.
        target_agent = "agent2"

        for episode in range(1, TOTAL_EPISODES + 1):
            env.reset()
            # Press 'f' once and keep FAST mode for all episodes.
            if episode == 1:
                pyautogui.press("f")
                fast_mode_enabled = True

            _set_vehicle_visuals_and_track(env, target_agent)

            sim_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            sim_elapsed_sec = 0.0
            stop_accum_sec = 0.0
            timeout = False
            reached_goal = False
            crash_count = 0
            prev_crash_state = False
            speed_sum = 0.0
            speed_steps = 0

            start_pos = tuple(env.agents[target_agent].position) if target_agent in env.agents else (0.0, 0.0)
            end_pos = _end_pos_from_map(env, start_pos)

            for i in range(1, 10_000_000):
                o, r, tm, tc, info = env.step(
                    {agent_id: [0, 0] for agent_id in env.agents.keys()}
                )
                # Simulation time increment (independent of real-time speedup with key 'f')
                sim_elapsed_sec += sim_step_sec

                target_vehicle = env.agents.get(target_agent, None)
                speed_kmh = target_vehicle.speed_km_h if target_vehicle is not None else 0.0
                speed_sum += speed_kmh
                speed_steps += 1

                # Count crash events on rising edge (avoid counting same persistent flag every frame)
                crash_state = False
                if target_agent in info:
                    crash_state = bool(
                        info[target_agent].get("crash_vehicle", False) or info[target_agent].get("crash_object", False)
                    )
                if crash_state and (not prev_crash_state):
                    crash_count += 1
                prev_crash_state = crash_state

                # Timeout when target speed stays near zero for > 1 minute
                if speed_kmh <= STOP_SPEED_KMH:
                    stop_accum_sec += sim_step_sec
                else:
                    stop_accum_sec = 0.0
                if stop_accum_sec >= STOP_TIMEOUT_SEC:
                    timeout = True

                hud = {
                    "Quit": "ESC",
                    "Episode": f"{episode}/{TOTAL_EPISODES}",
                    "Tracked Vehicle": f"RED UNIQUE ({target_agent})",
                    "Policy": "IDMPolicy",
                    "Speedup": "FAST (F pressed)" if fast_mode_enabled else "NORMAL",
                    "Sim Time (s)": f"{sim_elapsed_sec:.1f}",
                    "Timeout": "YES" if timeout else "NO",
                    "Crashes (target)": crash_count,
                    "Agents alive": len(env.agents),
                }
                env.render(text=hud)

                td_img = env.render(
                    mode="top_down",
                    window=False,
                    film_size=TD_FILM,
                    screen_size=TD_SCREEN,
                    scaling=TD_SCALING,
                    draw_contour=True,
                    center_on_map=True,
                )
                if td_img is not None:
                    tracked_vehicle = env.agents.get(target_agent, None)
                    vehicle_pos = tuple(tracked_vehicle.position) if tracked_vehicle is not None else start_pos
                    td_img = _draw_markers(
                        td_img,
                        env.top_down_renderer,
                        vehicle_pos,
                        start_pos,
                        end_pos,
                    )
                    td_tex = _numpy_to_panda_texture(td_img, td_tex)
                    if overlay is None:
                        overlay = OnscreenImage(
                            image=td_tex,
                            pos=(-0.62, 0, 0.55),
                            scale=(0.40, 1, 0.40),
                            parent=env.engine.aspect2d,
                        )
                        overlay.setTransparency(True)
                    else:
                        overlay.setTexture(td_tex)

                target_done = tm.get(target_agent, False) or tc.get(target_agent, False)
                reached_goal = bool(info.get(target_agent, {}).get("arrive_dest", False))
                if timeout or target_done:
                    break

            episode_elapsed = sim_elapsed_sec
            speed_avg = speed_sum / max(speed_steps, 1)
            results.append(
                {
                    "episode": episode,
                    "Pass": "YES" if not timeout else "NO",
                    "Timeout": "YES" if timeout else "NO",
                    "crash_count": crash_count,
                    "time_taken_to_reach_goal_sec": round(episode_elapsed, 2) if reached_goal else "N/A",
                    "speed_avg_kmh": round(speed_avg, 2),
                    "simulation_datetime": sim_datetime,
                }
            )
            print(
                f"Episode {episode} finished | Goal:{reached_goal} Timeout:{timeout} "
                f"Crashes:{crash_count} SimTime:{episode_elapsed:.1f}s Speedup:{'FAST' if fast_mode_enabled else 'NORMAL'}"
            )

        output_file = _save_results_xlsx(results)
        print(f"\nResults saved to: {output_file}\n")
    finally:
        env.close()
