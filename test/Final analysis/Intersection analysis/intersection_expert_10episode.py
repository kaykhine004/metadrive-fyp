#!/usr/bin/env python
"""
Roundabout + Intersection — 10-episode evaluation with interruption traffic.
Policy: ExpertPolicy
Route: [Start] ──► [Intersection] ──► [Straight] ──► [Roundabout] ──► [End]
"""
import os
import time
import copy

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
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_AGENT    = "agent0"
TARGET_SPAWN_ROAD = Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)
TARGET_DEST_NODE  = Roundabout.node(3, 1, 3)

INTERRUPTION_SPAWN_ROADS = [
    -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
    -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
    -Road(Roundabout.node(3, 2, 2),   Roundabout.node(3, 2, 3)),
]

TOTAL_EPISODES        = 10
STOP_TIMEOUT_SEC      = 60.0   # sim seconds of zero speed → timeout
STOP_SPEED_KMH        = 0.5    # km/h threshold to count as stopped
POST_CRASH_STUCK_SEC  = 10.0   # sim seconds stuck after a crash → end episode

OUTPUT_DIR = r"C:\Users\User\Desktop\FYP\metadrive0305\metadrive-main\test\Final analysis\Intersection analysis"

TD_SCREEN  = (450, 450)
TD_SCALING = 1.0
TD_FILM    = (800, 800)


# ---------------------------------------------------------------------------
# Agent config: target is pinned; interruption agents spawn naturally
# ---------------------------------------------------------------------------
def _build_agent_configs():
    return {
        TARGET_AGENT: {
            "use_special_color": False,   # we set color manually via setColorScale
            "random_color": True,
            "spawn_lane_index": (TARGET_SPAWN_ROAD.start_node, TARGET_SPAWN_ROAD.end_node, 0),
            "destination": TARGET_DEST_NODE,
        }
    }


def _apply_vehicle_colors(env):
    """Force target = strong red; all others keep random color but no red tint."""
    for agent_id, vehicle in env.agents.items():
        if agent_id == TARGET_AGENT:
            # Strong pure red — unmistakable
            vehicle.origin.setColorScale(1.0, 0.04, 0.04, 1.0)
        else:
            # Reset any leftover tint so random color shows naturally
            vehicle.origin.setColorScale(1.0, 1.0, 1.0, 1.0)


MA_CONFIG = dict(
    spawn_roads=[TARGET_SPAWN_ROAD, *INTERRUPTION_SPAWN_ROADS],
    end_node=TARGET_DEST_NODE,
    num_agents=10,
    horizon=10000,
    out_of_road_done=False,
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=120,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=220,
    crash_done=False,
    allow_respawn=False,
    agent_policy=ExpertPolicy,
    agent_configs=_build_agent_configs(),
    vehicle_config=dict(
        random_color=True,
        show_navi_mark=True,
        show_dest_mark=True,
        show_line_to_dest=True,
    ),
    interface_panel=["dashboard"],
)


# ---------------------------------------------------------------------------
# Custom map
# ---------------------------------------------------------------------------
class RoundaboutIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "Map not empty."

        first_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path, physics_world, length=length,
        )
        self.blocks.append(first_block)

        InterSection.EXIT_PART_LENGTH = length
        intersection_block = InterSection(
            1, first_block.get_socket(index=0),
            self.road_network, random_seed=1, ignore_intersection_checking=False,
        )
        intersection_block.enable_u_turn(self.config["lane_num"] > 1)
        intersection_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(intersection_block)

        straight_block = Straight(
            2, intersection_block.get_socket(index=0),
            self.road_network, random_seed=1,
        )
        straight_block.construct_from_config(
            {Parameter.length: length}, parent_node_path, physics_world,
        )
        self.blocks.append(straight_block)

        Roundabout.EXIT_PART_LENGTH = length
        roundabout_block = Roundabout(
            3, straight_block.get_socket(index=0),
            self.road_network, random_seed=1, ignore_intersection_checking=False,
        )
        roundabout_block.construct_block(
            parent_node_path, physics_world,
            extra_config={"exit_radius": 10, "inner_radius": 30, "angle": 70},
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
            assert len(self.spawned_objects) == 1
            _map = list(self.spawned_objects.values())[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


# ---------------------------------------------------------------------------
# Spawn manager
# ---------------------------------------------------------------------------
class RoundaboutIntersectionSpawnManager(SpawnManager):
    def update_destination_for(self, agent_id, vehicle_config):
        if agent_id == TARGET_AGENT:
            vehicle_config["destination"] = self.engine.global_config["end_node"]
            return vehicle_config
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        spawn_road = Road(*vehicle_config["spawn_lane_index"][:2])
        end_roads = [r for r in end_roads if r != spawn_road] or end_roads
        end_road = -self.np_random.choice(end_roads)
        vehicle_config["destination"] = end_road.end_node
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
# Helpers
# ---------------------------------------------------------------------------
def _world_to_screen(wx, wy, renderer):
    canvas = renderer._frame_canvas
    fx, fy = canvas.pos2pix(wx, wy)
    film_w, film_h = canvas.get_size()
    scr_w, scr_h = renderer._screen_canvas.get_size()
    return int(fx - (film_w / 2 - scr_w / 2)), int(fy - (film_h / 2 - scr_h / 2))


def _draw_markers(img, renderer, vehicle_pos, start_pos, end_pos):
    out = img.copy()
    sx, sy = _world_to_screen(*start_pos, renderer)
    cv2.putText(out, "START", (sx - 20, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "START", (sx - 20, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
    ex, ey = _world_to_screen(*end_pos, renderer)
    cv2.putText(out, "END", (ex - 14, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(out, "END", (ex - 14, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
    vx, vy = _world_to_screen(*vehicle_pos, renderer)
    cv2.circle(out, (vx, vy), 5, (0, 0, 255), -1)   # red in BGR
    cv2.circle(out, (vx, vy), 6, (0, 0, 0), 1)
    return out


def _numpy_to_panda_texture(img_bgr, tex=None):
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


def _save_xlsx(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"intersection_expert_results_{ts}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append([
        "Episode", "Pass", "Timeout", "crash_count",
        "time_taken_to_reach_goal_sec", "speed_avg_kmh", "simulation_datetime",
    ])
    for row in results:
        ws.append([
            row["episode"],
            row["Pass"],
            row["Timeout"],
            row["crash_count"],
            row["time_taken_to_reach_goal_sec"],
            row["speed_avg_kmh"],
            row["simulation_datetime"],
        ])
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Main — 10-episode loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = MultiAgentRoundaboutIntersectionEnv({"use_render": True})
    try:
        overlay    = None
        td_tex     = None
        results    = []
        fast_mode  = False
        # Simulation time per env.step() call (independent of real-time speedup)
        sim_step_sec = env.config["physics_world_step_size"] * env.config["decision_repeat"]

        # Capture fixed map landmarks once (map never changes)
        env.reset()
        _apply_vehicle_colors(env)
        start_pos = tuple(env.agents[TARGET_AGENT].position) \
            if TARGET_AGENT in env.agents else (0.0, 0.0)
        dest_lane_road = env.current_map.road_network.graph.get(
            Roundabout.node(3, 1, 2), {}
        ).get(Roundabout.node(3, 1, 3), None)
        end_pos = tuple(dest_lane_road[0].position(dest_lane_road[0].length, 0)) \
            if dest_lane_road else start_pos

        # ── Press F once ─ stays fast for all 10 episodes ──────────────────
        pyautogui.press("f")
        fast_mode = True

        # ── 10-episode loop ─────────────────────────────────────────────────
        for episode in range(1, TOTAL_EPISODES + 1):
            if episode > 1:
                env.reset()
                _apply_vehicle_colors(env)

            sim_datetime        = time.strftime("%Y-%m-%d %H:%M:%S")
            sim_elapsed         = 0.0
            stop_accum          = 0.0
            timeout             = False
            crash_stuck         = False   # stuck/spinning after a crash
            post_crash_stuck_accum = 0.0  # how long stuck since last crash
            reached_goal        = False
            crash_count         = 0
            prev_crash          = False
            speed_sum           = 0.0
            speed_steps         = 0

            for step in range(1, 10_000_000):
                o, r, tm, tc, info = env.step(
                    {aid: [0, 0] for aid in env.agents.keys()}
                )
                sim_elapsed += sim_step_sec

                # ── Speed / timeout tracking ─────────────────────────────
                target_v = env.agents.get(TARGET_AGENT, None)
                speed_kmh = target_v.speed_km_h if target_v is not None else 0.0
                speed_sum   += speed_kmh
                speed_steps += 1

                if speed_kmh <= STOP_SPEED_KMH:
                    stop_accum += sim_step_sec
                else:
                    stop_accum = 0.0
                if stop_accum >= STOP_TIMEOUT_SEC:
                    timeout = True

                # ── Crash counting (rising-edge only) ────────────────────
                # Only count when the target tracked vehicle is crashed — ignore crashes between other agents
                crash_now = bool(
                    info.get(TARGET_AGENT, {}).get("crash_vehicle", False) or
                    info.get(TARGET_AGENT, {}).get("crash_object",  False)
                )
                if crash_now and not prev_crash:
                    crash_count += 1
                    post_crash_stuck_accum = 0.0   # restart stuck timer on each new crash
                prev_crash = crash_now

                # ── Post-crash stuck detection ────────────────────────────
                # If vehicle has ever crashed and is now barely moving, accumulate.
                if crash_count > 0 and speed_kmh <= STOP_SPEED_KMH:
                    post_crash_stuck_accum += sim_step_sec
                elif speed_kmh > STOP_SPEED_KMH:
                    post_crash_stuck_accum = 0.0   # reset when moving again
                if post_crash_stuck_accum >= POST_CRASH_STUCK_SEC:
                    crash_stuck = True

                # ── HUD ──────────────────────────────────────────────────
                hud = {
                    "Quit":             "ESC",
                    "Episode":          f"{episode}/{TOTAL_EPISODES}",
                    "Tracked Vehicle":  f"RED ({TARGET_AGENT})",
                    "Policy":           "ExpertPolicy",
                    "Speedup":          "FAST (F pressed)" if fast_mode else "NORMAL",
                    "Sim Time (s)":     f"{sim_elapsed:.1f}",
                    "Timeout":          "YES" if timeout else "NO",
                    "Crash Stuck":      "YES - Ending episode" if crash_stuck else "NO",
                    "Crashes (target)": crash_count,
                    "Agents alive":     len(env.agents),
                }
                env.render(text=hud)

                # ── Top-down overlay ─────────────────────────────────────
                td_img = env.render(
                    mode="top_down", window=False,
                    film_size=TD_FILM, screen_size=TD_SCREEN,
                    scaling=TD_SCALING, draw_contour=True, center_on_map=True,
                )
                if td_img is not None:
                    vpos = tuple(target_v.position) if target_v is not None else start_pos
                    td_img = _draw_markers(
                        td_img, env.top_down_renderer, vpos, start_pos, end_pos,
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

                # ── Episode end conditions ───────────────────────────────
                reached_goal = bool(info.get(TARGET_AGENT, {}).get("arrive_dest", False))
                target_done  = tm.get(TARGET_AGENT, False) or tc.get(TARGET_AGENT, False)
                if timeout or crash_stuck or target_done:
                    break

            # ── Record episode result ────────────────────────────────────
            speed_avg = speed_sum / max(speed_steps, 1)
            passed    = (not timeout) and (not crash_stuck)
            results.append({
                "episode":                     episode,
                "Pass":                        "YES" if passed else "NO",
                "Timeout":                     "YES" if timeout else "NO",
                "crash_count":                 crash_count,
                "time_taken_to_reach_goal_sec":
                    round(sim_elapsed, 2) if reached_goal else "N/A",
                "speed_avg_kmh":               round(speed_avg, 2),
                "simulation_datetime":         sim_datetime,
            })
            reason = "CRASH_STUCK" if crash_stuck else ("TIMEOUT" if timeout else "COMPLETED")
            print(
                f"Episode {episode:2d} | Pass:{'YES' if passed else 'NO ':3s} "
                f"| Reason:{reason} | Crashes:{crash_count} "
                f"| SimTime:{sim_elapsed:.1f}s | SpeedAvg:{speed_avg:.1f} km/h"
            )

        # ── Save xlsx after all 10 episodes ─────────────────────────────────
        out_path = _save_xlsx(results)
        print(f"\nResults saved to:\n  {out_path}\n")

    finally:
        env.close()
