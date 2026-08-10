"""
environment.py
================================================================================
MULTI-AGENT SUMO/TraCI TRAFFIC ENVIRONMENT (3 independent RL agents)
================================================================================

This module replaces the original single-agent `TrafficEnvironment` with a
true multi-agent environment that controls THREE independent RL vehicles
("agent1", "agent2", "agent3") inside a single, shared SUMO simulation.

--------------------------------------------------------------------------------
WHY A CUSTOM DESIGN INSTEAD OF PLAIN PettingZoo ParallelEnv?
--------------------------------------------------------------------------------
SUMO/TraCI exposes exactly ONE simulation clock. All vehicles - RL controlled
or background traffic - live inside that single clock, and `traci.simulationStep()`
must be called exactly once per joint timestep for every agent to observe a
consistent world state. This class therefore behaves like a PettingZoo
`ParallelEnv` (dict-in / dict-out API: every agent submits an action, the
whole world advances once, every agent gets back its own observation /
reward / termination / truncation / info) while being TraCI-safe.

If the `pettingzoo` package is installed, `MultiAgentTrafficEnv` inherits
from `pettingzoo.ParallelEnv` for full API compliance (so it can be dropped
into PettingZoo/SuperSuit tooling). If PettingZoo is not installed, the class
still works standalone - it simply behaves as a plain Python object that
implements the same reset()/step() dict contract, so the rest of this project
(train.py / test.py / evaluate.py) never has a hard dependency on PettingZoo.

--------------------------------------------------------------------------------
KEY DESIGN DECISION: PER-AGENT RESPAWNING INSTEAD OF PER-AGENT EARLY EXIT
--------------------------------------------------------------------------------
Each agent has its OWN termination condition (collision, or arriving at its
destination) as required. However, ending only ONE agent's episode early
while the other two keep running would desynchronize a single shared,
single-threaded TraCI simulation (there would be no well-defined "next
action" to submit for the finished agent while the others keep stepping).

Instead, when agent_i's vehicle disappears (collision or arrival):
    1. `terminations[agent_i] = True` is reported for that ONE step, together
       with the correct terminal reward (destination bonus / collision
       penalty) - so the caller sees a clean, independent episode boundary
       for that agent.
    2. The vehicle is then immediately re-inserted ("respawned") into the
       simulation via `traci.vehicle.add(...)`, so agent_i keeps producing a
       fresh episode on the very next joint step, independently of agent_j
       and agent_k.
    3. The WHOLE joint simulation truncates together after `max_steps`
       (`truncations[*] = True` for all agents simultaneously) - this is the
       point at which the shared SUMO process is actually restarted.

This keeps the environment fully synchronous and TraCI-safe (no threads, no
races) while still giving every agent genuinely independent termination
events, rewards, and observations, satisfying the "independent agents"
requirement without deadlocking the shared simulation clock.
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import traci.exceptions

# PettingZoo is optional. We degrade gracefully if it is not installed.
try:
    from pettingzoo import ParallelEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    ParallelEnv = object
    PETTINGZOO_AVAILABLE = False


# ==============================================================================
# GLOBAL CONSTANTS (shared by environment.py, train.py, test.py, evaluate.py)
# ==============================================================================

# The three independent RL-controlled vehicles. Each MUST exist as a vehicle
# (or at least a valid <route>) in the SUMO network / route file, OR the
# environment will fall back to spawning them on the first available route.
AGENT_IDS = ["agent1", "agent2", "agent3"]

# Human-readable action meanings (Discrete(8), identical for every agent).
ACTION_MEANINGS = {
    0: "ACCELERATE",
    1: "BRAKE",
    2: "MAINTAIN_SPEED",
    3: "SLOW",
    4: "FAST",
    5: "CHANGE_LANE_LEFT",
    6: "CHANGE_LANE_RIGHT",
    7: "OVERTAKE",
}
NUM_ACTIONS = len(ACTION_MEANINGS)

# --------------------------------------------------------------------------
# Observation vector (12 features), identical layout for every agent:
#   0  current speed                (m/s)
#   1  leader speed                 (m/s)
#   2  gap to leader                (m)
#   3  waiting time                 (s)
#   4  acceleration                 (m/s^2)
#   5  lane index                   (-)
#   6  number of lanes on the edge  (-)
#   7  local traffic density        (vehicles on current edge)
#   8  traffic light state          (0=red, 1=yellow, 2=green/none)
#   9  distance to next traffic light (m)
#   10 distance to junction / end of lane (m)
#   11 route progress               (%)
# --------------------------------------------------------------------------
OBS_LOW = np.array(
    [0, 0, 0, 0, -10, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32
)
OBS_HIGH = np.array(
    [40, 40, 500, 1000, 10, 10, 10, 200, 2, 5000, 5000, 100], dtype=np.float32
)
OBS_DIM = len(OBS_LOW)


class MultiAgentTrafficEnv(ParallelEnv):
    """
    A PettingZoo-Parallel-style multi-agent wrapper around a single shared
    SUMO/TraCI simulation controlling three independent RL vehicles.

    Public API (mirrors PettingZoo's ParallelEnv):
        env.possible_agents        -> ["agent1", "agent2", "agent3"]
        env.agents                 -> currently active agents (empty until reset)
        env.observation_space(id)  -> gymnasium.spaces.Box  (per-agent, all identical)
        env.action_space(id)       -> gymnasium.spaces.Discrete(8)
        obs, infos = env.reset()
        obs, rewards, terminations, truncations, infos = env.step(actions_dict)
        env.close()
    """

    metadata = {"name": "multi_agent_traffic_v0", "render_modes": ["human", "none"]}

    def __init__(
        self,
        sumo_config="simulation/config.sumocfg",
        use_gui=False,
        max_steps=3000,
        max_speed=20.0,
        agent_route_map=None,
        spawn_timeout_steps=2000,
        seed=None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # SUMO / TraCI configuration
        # ------------------------------------------------------------------
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config = sumo_config
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_config,
            "--collision.action", "warn",       # keep the sim running after a crash
            "--collision.check-junctions", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--start",                          # auto-start when using sumo-gui
            "--quit-on-end", "true",
        ]
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.spawn_timeout_steps = spawn_timeout_steps
        self._rng = random.Random(seed)

        # ------------------------------------------------------------------
        # Multi-agent bookkeeping (PettingZoo convention)
        # ------------------------------------------------------------------
        self.possible_agents = list(AGENT_IDS)
        self.agents = []  # populated on reset()

        # Maps each agent id -> the SUMO <route> id used when (re)spawning it.
        # Falls back automatically at runtime if the id is not found.
        self.agent_route_map = agent_route_map or {
            aid: f"route_{aid}" for aid in AGENT_IDS
        }

        self._observation_spaces = {
            aid: spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
            for aid in AGENT_IDS
        }
        self._action_spaces = {aid: spaces.Discrete(NUM_ACTIONS) for aid in AGENT_IDS}
        # Also expose as plain dict attributes (some MARL tooling expects this).
        self.observation_spaces = self._observation_spaces
        self.action_spaces = self._action_spaces

        # ------------------------------------------------------------------
        # Per-agent episode state
        # ------------------------------------------------------------------
        self.step_count = 0
        self.sumo_running = False
        self._prev_route_progress = {aid: 0.0 for aid in AGENT_IDS}
        self._route_length_cache = {}
        self._pending_penalty = {aid: 0.0 for aid in AGENT_IDS}
        self._episode_reward = {aid: 0.0 for aid in AGENT_IDS}
        self._stats = {aid: {"arrivals": 0, "collisions": 0} for aid in AGENT_IDS}

    # ==========================================================================
    # PettingZoo-style space accessors
    # ==========================================================================
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ==========================================================================
    # RESET
    # ==========================================================================
    def reset(self, seed=None, options=None):
        """Starts (or restarts) the shared SUMO simulation and returns the
        first observation/info dict for every agent."""
        if seed is not None:
            self._rng.seed(seed)

        self._safe_close_traci()

        try:
            traci.start(self.sumo_cmd)
            self.sumo_running = True
        except traci.exceptions.TraCIException as exc:
            raise RuntimeError(f"Failed to start SUMO: {exc}") from exc

        self.agents = list(self.possible_agents)
        self.step_count = 0
        self._prev_route_progress = {aid: 0.0 for aid in self.agents}
        self._route_length_cache = {}
        self._pending_penalty = {aid: 0.0 for aid in self.agents}
        self._episode_reward = {aid: 0.0 for aid in self.agents}
        self._stats = {aid: {"arrivals": 0, "collisions": 0} for aid in self.agents}

        # Warm up the simulation until all three agent vehicles exist (either
        # they spawn naturally from the .rou.xml, or we insert them manually).
        for _ in range(self.spawn_timeout_steps):
            traci.simulationStep()
            present = set(traci.vehicle.getIDList())
            if all(aid in present for aid in self.agents):
                break
        for aid in self.agents:
            if aid not in traci.vehicle.getIDList():
                self._respawn_agent(aid)
        traci.simulationStep()

        for aid in self.agents:
            self._configure_vehicle(aid)

        observations = {aid: self._get_observation(aid) for aid in self.agents}
        infos = {aid: {"episode_start": True} for aid in self.agents}
        return observations, infos

    # ==========================================================================
    # STEP
    # ==========================================================================
    def step(self, actions):
        """
        Parameters
        ----------
        actions : dict[str, int]
            Maps agent id -> Discrete(8) action. Missing agents default to
            action 2 (MAINTAIN_SPEED) so a partially-specified dict never
            crashes the simulation.

        Returns
        -------
        observations, rewards, terminations, truncations, infos : dict[str, ...]
        """
        if not self.sumo_running:
            raise RuntimeError("Environment not initialized - call reset() first.")

        active_before = set(traci.vehicle.getIDList())

        # ---- 1) Apply every agent's action BEFORE advancing the clock -------
        for aid in self.agents:
            action = actions.get(aid, 2)
            if aid in active_before:
                self._apply_action(aid, action)

        # ---- 2) Advance the shared simulation EXACTLY ONCE -------------------
        try:
            traci.simulationStep()
        except traci.exceptions.TraCIException as exc:
            # Catastrophic simulation-level failure: end the episode safely
            # for every agent rather than propagating a crash.
            rewards = {aid: 0.0 for aid in self.agents}
            terminations = {aid: False for aid in self.agents}
            truncations = {aid: True for aid in self.agents}
            infos = {aid: {"error": str(exc)} for aid in self.agents}
            observations = {aid: np.zeros(OBS_DIM, dtype=np.float32) for aid in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        self.step_count += 1
        present_after = set(traci.vehicle.getIDList())
        colliding = set(self._safe_get_collisions())
        joint_truncated = self.step_count >= self.max_steps

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for aid in self.agents:
            was_present = aid in active_before
            now_present = aid in present_after
            crashed = aid in colliding

            reward, term_flag, info = self._compute_reward_and_termination(
                aid, was_present, now_present, crashed
            )

            self._episode_reward[aid] += reward
            info["cumulative_reward"] = self._episode_reward[aid]

            rewards[aid] = reward
            terminations[aid] = term_flag
            truncations[aid] = joint_truncated
            infos[aid] = info
            observations[aid] = (
                self._get_observation(aid) if now_present else np.zeros(OBS_DIM, dtype=np.float32)
            )

            # Independent respawn: give this agent a fresh episode right away
            # without waiting for its siblings, unless the whole run is over.
            if term_flag and not joint_truncated:
                self._episode_reward[aid] = 0.0
                self._respawn_agent(aid)
                infos[aid]["episode_start"] = True

        if joint_truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ==========================================================================
    # ACTION APPLICATION  (with defensive error handling, requirement #10)
    # ==========================================================================
    def _apply_action(self, aid, action):
        if aid not in traci.vehicle.getIDList():
            return
        try:
            current_speed = traci.vehicle.getSpeed(aid)
        except traci.exceptions.TraCIException:
            return  # vehicle vanished between the presence check and now

        try:
            if action == 0:      # Accelerate
                traci.vehicle.setSpeed(aid, min(current_speed + 2.0, self.max_speed))
            elif action == 1:    # Brake
                traci.vehicle.setSpeed(aid, max(current_speed - 3.0, 0.0))
            elif action == 2:    # Maintain speed
                traci.vehicle.setSpeed(aid, current_speed)
            elif action == 3:    # Slow
                traci.vehicle.setSpeed(aid, max(current_speed - 1.0, 0.0))
            elif action == 4:    # Fast
                traci.vehicle.setSpeed(aid, min(current_speed + 1.0, self.max_speed))
            elif action == 5:    # Change lane left
                self._safe_lane_change(aid, direction=+1)
            elif action == 6:    # Change lane right
                self._safe_lane_change(aid, direction=-1)
            elif action == 7:    # Overtake
                self._attempt_overtake(aid)
        except traci.exceptions.TraCIException:
            # Any low level SUMO/TraCI error is swallowed: one bad command
            # must never crash the whole multi-agent episode.
            pass

    def _safe_lane_change(self, aid, direction):
        """Handles single-lane roads, invalid target lanes, and vehicles
        currently inside a junction (internal edges) without ever raising."""
        try:
            edge_id = traci.vehicle.getRoadID(aid)
            if not edge_id or edge_id.startswith(":"):
                return  # inside a junction - lane changes are not meaningful here
            lane_count = traci.edge.getLaneNumber(edge_id)
            if lane_count <= 1:
                return  # single-lane road: nothing to change to
            current_lane = traci.vehicle.getLaneIndex(aid)
            target_lane = current_lane + direction
            if target_lane < 0 or target_lane >= lane_count:
                return  # would be an invalid lane index - ignore safely
            traci.vehicle.changeLane(aid, target_lane, 5.0)
            self._pending_penalty[aid] = self._pending_penalty.get(aid, 0.0) - 0.5
        except traci.exceptions.TraCIException:
            pass

    def _attempt_overtake(self, aid):
        """Overtake = change to the left lane and briefly accelerate, but
        only when there genuinely is a slower leader worth passing."""
        try:
            leader = traci.vehicle.getLeader(aid)
            if leader is None:
                return  # no leader -> nothing to overtake
            leader_id, gap = leader
            leader_speed = traci.vehicle.getSpeed(leader_id)
            current_speed = traci.vehicle.getSpeed(aid)
            if leader_speed >= current_speed or gap > 50:
                return
            edge_id = traci.vehicle.getRoadID(aid)
            if not edge_id or edge_id.startswith(":"):
                return
            lane_count = traci.edge.getLaneNumber(edge_id)
            if lane_count <= 1:
                return
            current_lane = traci.vehicle.getLaneIndex(aid)
            target_lane = current_lane + 1
            if target_lane >= lane_count:
                return
            traci.vehicle.changeLane(aid, target_lane, 5.0)
            traci.vehicle.setSpeed(aid, min(current_speed + 3.0, self.max_speed))
            self._pending_penalty[aid] = self._pending_penalty.get(aid, 0.0) - 0.3
        except traci.exceptions.TraCIException:
            pass

    # ==========================================================================
    # OBSERVATION  (requirement #11)
    # ==========================================================================
    def _get_observation(self, aid):
        if aid not in traci.vehicle.getIDList():
            return np.zeros(OBS_DIM, dtype=np.float32)

        speed = self._safe(lambda: traci.vehicle.getSpeed(aid), 0.0)
        waiting_time = self._safe(lambda: traci.vehicle.getWaitingTime(aid), 0.0)
        acceleration = self._safe(lambda: traci.vehicle.getAcceleration(aid), 0.0)
        lane_index = self._safe(lambda: traci.vehicle.getLaneIndex(aid), 0)
        road_id = self._safe(lambda: traci.vehicle.getRoadID(aid), "")

        num_lanes = self._safe(
            lambda: traci.edge.getLaneNumber(road_id) if road_id and not road_id.startswith(":") else 1,
            1,
        )

        # ---- Leader vehicle: handle "missing leader" gracefully ----
        leader = self._safe(lambda: traci.vehicle.getLeader(aid), None)
        if leader is None:
            gap_to_leader, leader_speed = 500.0, self.max_speed
        else:
            leader_id, gap_to_leader = leader
            leader_speed = self._safe(lambda: traci.vehicle.getSpeed(leader_id), self.max_speed)

        local_density = self._safe(
            lambda: traci.edge.getLastStepVehicleNumber(road_id) if road_id and not road_id.startswith(":") else 0,
            0,
        )

        # ---- Traffic light state + distance ("invalid traffic light" safe) ----
        tls_state_val, tls_distance = 0.0, 1000.0
        try:
            next_tls = traci.vehicle.getNextTLS(aid)
            if len(next_tls) > 0:
                _tls_id, _link_idx, tls_distance, tls_state = next_tls[0]
                if tls_state in ("G", "g"):
                    tls_state_val = 2.0
                elif tls_state in ("y", "Y"):
                    tls_state_val = 1.0
                else:
                    tls_state_val = 0.0
        except traci.exceptions.TraCIException:
            pass

        # ---- Distance to the end of the current lane / next junction ----
        distance_to_junction = self._safe(
            lambda: max(
                traci.lane.getLength(traci.vehicle.getLaneID(aid))
                - traci.vehicle.getLanePosition(aid),
                0.0,
            ),
            5000.0,
        )

        route_progress = self._get_route_progress(aid)

        observation = np.array(
            [
                speed, leader_speed, gap_to_leader, waiting_time, acceleration,
                lane_index, num_lanes, local_density, tls_state_val, tls_distance,
                distance_to_junction, route_progress,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, OBS_LOW, OBS_HIGH)

    def _get_route_progress(self, aid):
        """Approximate % of the assigned route the vehicle has completed."""
        try:
            route = traci.vehicle.getRoute(aid)
            route_index = traci.vehicle.getRouteIndex(aid)
            lane_position = traci.vehicle.getLanePosition(aid)

            if aid not in self._route_length_cache:
                total_length = 0.0
                for edge in route:
                    total_length += self._safe(lambda e=edge: traci.lane.getLength(e + "_0"), 0.0)
                self._route_length_cache[aid] = max(total_length, 1.0)
            total_length = self._route_length_cache[aid]

            traveled = 0.0
            for edge in route[:route_index]:
                traveled += self._safe(lambda e=edge: traci.lane.getLength(e + "_0"), 0.0)
            traveled += lane_position

            return float(np.clip((traveled / total_length) * 100.0, 0.0, 100.0))
        except traci.exceptions.TraCIException:
            return self._prev_route_progress.get(aid, 0.0)

    @staticmethod
    def _safe(fn, default):
        """Runs `fn`, swallowing TraCIException and returning `default`
        instead - centralizes the 'missing leader / invalid edge / invalid
        traffic light / vehicle removed' robustness requirement."""
        try:
            result = fn()
            return default if result is None else result
        except traci.exceptions.TraCIException:
            return default
        except Exception:
            return default

    def _safe_get_collisions(self):
        try:
            return traci.simulation.getCollidingVehiclesIDList()
        except traci.exceptions.TraCIException:
            return []

    # ==========================================================================
    # REWARD  (requirement #12)
    # ==========================================================================
    def _compute_reward_and_termination(self, aid, was_present, now_present, crashed):
        info = {}

        if not was_present:
            # The vehicle hadn't spawned yet this tick (e.g. mid warm-up).
            return 0.0, False, info

        if crashed:
            info["event"] = "collision"
            self._stats[aid]["collisions"] += 1
            return -100.0, True, info

        if was_present and not now_present:
            # Disappeared with no recorded collision -> reached its destination.
            info["event"] = "arrived"
            self._stats[aid]["arrivals"] += 1
            return 100.0, True, info

        return self._shaped_reward(aid), False, info

    def _shaped_reward(self, aid):
        reward = 0.0
        speed = self._safe(lambda: traci.vehicle.getSpeed(aid), 0.0)

        # --- Speed reward: encourages efficient, non-stationary travel ---
        reward += (speed / max(self.max_speed, 1e-3)) * 2.0

        # --- Route progress reward: only rewards forward progress ---
        progress = self._get_route_progress(aid)
        delta_progress = progress - self._prev_route_progress.get(aid, progress)
        self._prev_route_progress[aid] = progress
        reward += max(delta_progress, 0.0) * 5.0

        # --- Waiting penalty ---
        waiting_time = self._safe(lambda: traci.vehicle.getWaitingTime(aid), 0.0)
        reward -= waiting_time * 0.1

        # --- Emergency braking penalty ---
        accel = self._safe(lambda: traci.vehicle.getAcceleration(aid), 0.0)
        if accel < -4.5:
            reward -= 5.0

        # --- Safe following distance reward / unsafe gap penalty ---
        leader = self._safe(lambda: traci.vehicle.getLeader(aid), None)
        if leader is not None:
            gap = leader[1]
            if 10.0 <= gap <= 50.0:
                reward += 2.0
            elif gap < 5.0:
                reward -= 8.0

        # --- Standing-still penalty (excludes legitimate red-light stops) ---
        if speed < 0.3 and waiting_time < 1.0:
            reward -= 1.0

        # --- Lane-change penalty (accumulated by _safe_lane_change/_attempt_overtake) ---
        reward += self._pending_penalty.get(aid, 0.0)
        self._pending_penalty[aid] = 0.0

        return reward

    # ==========================================================================
    # VEHICLE (RE)SPAWNING  (requirement #10: "vehicle removed from simulation")
    # ==========================================================================
    def _respawn_agent(self, aid):
        try:
            if aid in traci.vehicle.getIDList():
                return

            route_id = self.agent_route_map.get(aid)
            available_routes = traci.route.getIDList()
            if route_id not in available_routes:
                # The .rou.xml doesn't define a per-agent route: fall back to
                # any valid route so the project never hard-crashes on setup.
                route_id = available_routes[0] if available_routes else None
            if route_id is None:
                return  # nothing we can do - no routes exist in this network

            available_types = self._safe(lambda: traci.vehicletype.getIDList(), ())
            type_id = "vType_rl" if "vType_rl" in available_types else "DEFAULT_VEHTYPE"

            traci.vehicle.add(
                vehID=aid,
                routeID=route_id,
                typeID=type_id,
                departLane="best",
                departSpeed="0",
            )
            self._configure_vehicle(aid)
            self._prev_route_progress[aid] = 0.0
            self._route_length_cache.pop(aid, None)
        except traci.exceptions.TraCIException:
            pass  # respawn failed this tick - we'll simply retry next step

    def _configure_vehicle(self, aid):
        try:
            # SpeedMode 32: obey safety constraints (no collisions) but allow
            # the RL policy to freely set speed via setSpeed().
            traci.vehicle.setSpeedMode(aid, 32)
            # LaneChangeMode 256: disable SUMO's automatic lane changes so
            # only our RL-issued changeLane() calls move the vehicle sideways.
            traci.vehicle.setLaneChangeMode(aid, 256)
            traci.vehicle.setColor(aid, self._agent_color(aid))
        except traci.exceptions.TraCIException:
            pass

    @staticmethod
    def _agent_color(aid):
        palette = {
            "agent1": (255, 0, 0, 255),    # red
            "agent2": (0, 120, 255, 255),  # blue
            "agent3": (0, 200, 0, 255),    # green
        }
        return palette.get(aid, (255, 255, 0, 255))

    # ==========================================================================
    # RENDER / CLOSE
    # ==========================================================================
    def render(self):
        # sumo-gui renders itself; nothing additional required here.
        pass

    def _safe_close_traci(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

    def close(self):
        self._safe_close_traci()
        self.sumo_running = False
        self.agents = []


# ==============================================================================
# Lightweight placeholder Gym env (used ONLY to construct SB3 PPO objects)
# ==============================================================================
class SingleAgentSpaceEnv(gym.Env):
    """
    A minimal Gymnasium environment that performs NO real simulation.

    Stable-Baselines3's `PPO(...)` constructor needs *some* `gym.Env` to read
    `observation_space` / `action_space` from, and to build its internal
    `DummyVecEnv` + `RolloutBuffer`. Because our real multi-agent world lives
    inside `MultiAgentTrafficEnv` (one shared TraCI simulation for all three
    agents), we give each agent's PPO model this tiny placeholder purely for
    object construction. All genuine experience used for learning comes from
    `MultiAgentTrafficEnv` via the custom training loop in train.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


# ==============================================================================
# Manual smoke test (only runs if SUMO/TraCI + a real config are available)
# ==============================================================================
if __name__ == "__main__":
    print(f"PettingZoo available: {PETTINGZOO_AVAILABLE}")
    env = MultiAgentTrafficEnv(use_gui=False, max_steps=200)
    try:
        obs, infos = env.reset()
        print("Reset OK. Agents:", env.agents)
        for _ in range(50):
            actions = {aid: env.action_space(aid).sample() for aid in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            print(rewards)
            if not env.agents:
                break
    finally:
        env.close()
