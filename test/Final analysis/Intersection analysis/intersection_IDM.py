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
import copy

import cv2
import numpy as np
from direct.gui.OnscreenImage import OnscreenImage
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
TARGET_AGENT = "agent0"
TARGET_SPAWN_ROAD = Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)
TARGET_DEST_NODE = Roundabout.node(3, 1, 3)

# 3 interruption routes from the sketch:
#   (1) top incoming route to intersection
#   (2) side route at intersection
#   (3) lower route from roundabout toward end
INTERRUPTION_SPAWN_ROADS = [
    -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
    -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
    -Road(Roundabout.node(3, 2, 2), Roundabout.node(3, 2, 3)),
]


def _build_agent_configs():
    # Keep only target fixed. All other agents use natural spawn sampling
    # from `spawn_roads` handled by SpawnManager.
    return {
        TARGET_AGENT: {
            "use_special_color": True,
            "spawn_lane_index": (TARGET_SPAWN_ROAD.start_node, TARGET_SPAWN_ROAD.end_node, 0),
            "destination": TARGET_DEST_NODE,
        }
    }


MA_CONFIG = dict(
    spawn_roads=[
        TARGET_SPAWN_ROAD,
        *INTERRUPTION_SPAWN_ROADS,
    ],
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
    agent_policy=IDMPolicy,
    agent_configs=_build_agent_configs(),
    vehicle_config=dict(
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
        # Keep target route fixed exactly as requested.
        if agent_id == TARGET_AGENT:
            vehicle_config["destination"] = self.engine.global_config["end_node"]
            return vehicle_config

        # Interruption vehicles choose random destinations (excluding current spawn road)
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        spawn_road = Road(*vehicle_config["spawn_lane_index"][:2])
        end_roads = [r for r in end_roads if r != spawn_road]
        if len(end_roads) == 0:
            end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
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
# Main — track only agent0, show stats HUD
# ---------------------------------------------------------------------------
TARGET = TARGET_AGENT

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

    # Current vehicle marker (small bright dot)
    vx, vy = _world_to_screen(*vehicle_pos, renderer)
    cv2.circle(out, (vx, vy), 5, (255, 255, 0), -1)
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


if __name__ == "__main__":
    env = MultiAgentRoundaboutIntersectionEnv({"use_render": True})
    try:
        env.reset()
        start_time = time.time()
        crash_count = 0
        overlay = None
        td_tex = None

        # Capture the spawn position of agent0 as the fixed START location
        start_pos = tuple(env.agents[TARGET].position)

        # The END position is the midpoint of the destination lane
        dest_road = env.current_map.road_network.graph.get(
            Roundabout.node(3, 1, 2), {}
        ).get(Roundabout.node(3, 1, 3), None)
        if dest_road:
            lane = dest_road[0]
            end_pos = tuple(lane.position(lane.length, 0))
        else:
            end_pos = start_pos

        for i in range(1, 10_000_000):
            o, r, tm, tc, info = env.step(
                {agent_id: [0, 0] for agent_id in env.agents.keys()}
            )

            if TARGET in info:
                if info[TARGET].get("crash_vehicle", False):
                    crash_count += 1
                if info[TARGET].get("crash_object", False):
                    crash_count += 1

            elapsed = time.time() - start_time

            hud = {
                "Quit": "ESC",
                "Tracked Vehicle": f"GREEN ({TARGET})",
                "Policy": "IDMPolicy",
                "Time (s)": f"{elapsed:.1f}",
                "Crashes (target)": crash_count,
                "Agents alive": len(env.agents),
            }

            # 3D main view with HUD text
            env.render(text=hud)

            # Render 2D top-down off-screen
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
                # Get current vehicle world position
                vehicle_pos = (0.0, 0.0)
                if TARGET in env.agents:
                    vehicle_pos = tuple(env.agents[TARGET].position)

                # Draw START / END / vehicle markers on the image
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

            target_done = tm.get(TARGET, False) or tc.get(TARGET, False)
            if target_done:
                elapsed = time.time() - start_time
                arrived = info.get(TARGET, {}).get("arrive_dest", False)
                print(f"\n{'='*50}")
                print(f"  Target vehicle '{TARGET}' finished!")
                print(f"  Reached destination: {arrived}")
                print(f"  Policy: IDMPolicy")
                print(f"  Total time: {elapsed:.1f}s  |  Steps: {i}")
                print(f"  Crashes: {crash_count}")
                print(f"{'='*50}\n")
                break

            if tm["__all__"]:
                env.reset()
                start_time = time.time()
                crash_count = 0
    finally:
        env.close()
