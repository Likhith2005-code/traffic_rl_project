import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import time


class TrafficEnvironment(gym.Env):

    def __init__(self):
        super(TrafficEnvironment, self).__init__()

        # -----------------------------
        # SUMO Configuration
        # -----------------------------
        self.sumoBinary = "sumo-gui"
        self.sumoConfig = "simulation/config.sumocfg"

        self.sumoCmd = [
            self.sumoBinary,
            "-c",
            self.sumoConfig
        ]

        # -----------------------------
        # RL Agent
        # -----------------------------
        self.agent_id = "agent"
        self.max_speed = 20.0

        # -----------------------------
        # Observation Space
        # speed
        # lane position
        # distance to leader
        # waiting time
        # traffic light state
        # -----------------------------
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([30, 10000, 500, 1000, 2]),
            dtype=np.float32
        )

        # -----------------------------
        # Action Space
        # 0 Accelerate
        # 1 Brake
        # 2 Maintain
        # 3 Slow
        # 4 Fast
        # -----------------------------
        self.action_space = spaces.Discrete(5)

    # ==========================================
    # RESET
    # ==========================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        traci.start(self.sumoCmd)

        # Wait for agent to spawn
        for _ in range(1000):

            traci.simulationStep()

            if "agent" in traci.vehicle.getIDList():

                self.agent_id = "agent"

                print("Agent Found!")
                if self.sumoBinary=="sumo-gui":
                   traci.vehicle.setColor(self.agent_id, (255, 0, 0, 255))
                   traci.gui.trackVehicle("View #0", self.agent_id)

                observation = self.get_observation()

                return observation, {}
    
        raise Exception("Agent vehicle never appeared.")

    # ==========================================
    # STEP
    # ==========================================

    def step(self, action):
        # Agent already disappeared
        if self.agent_id not in traci.vehicle.getIDList():

            return (
                np.zeros(5, dtype=np.float32),
                -100,
                True,
                False,
                {}
            )

        current_speed = traci.vehicle.getSpeed(self.agent_id)

        if action == 0:

            traci.vehicle.setSpeed(
                self.agent_id,
                min(current_speed + 2, self.max_speed)
            )

        elif action == 1:

            traci.vehicle.setSpeed(
                self.agent_id,
                max(current_speed - 2, 0)
            )

        elif action == 2:

            pass

        elif action == 3:

            traci.vehicle.setSpeed(
                self.agent_id,
                max(current_speed - 1, 0)
            )

        elif action == 4:

            traci.vehicle.setSpeed(
                self.agent_id,
                min(current_speed + 1, self.max_speed)
            )

        traci.simulationStep()

        time.sleep(0.1)
        # Agent reached destination
        if self.agent_id not in traci.vehicle.getIDList():

            return (
                np.zeros(5, dtype=np.float32),
                100,
                True,
                False,
                {}
            )

        observation = self.get_observation()

        reward = self.calculate_reward()

        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, {}

    # ==========================================
    # OBSERVATION
    # ==========================================

    def get_observation(self):

        if self.agent_id not in traci.vehicle.getIDList():

            return np.zeros(5, dtype=np.float32)

        speed = traci.vehicle.getSpeed(self.agent_id)

        lane_position = traci.vehicle.getLanePosition(self.agent_id)

        waiting_time = traci.vehicle.getWaitingTime(self.agent_id)

        leader = traci.vehicle.getLeader(self.agent_id)

        if leader is None:

            distance = 500

        else:

            distance = leader[1]

        traffic_light = 0

        observation = np.array(

            [
                speed,
                lane_position,
                distance,
                waiting_time,
                traffic_light
            ],

            dtype=np.float32

        )

        return observation

    # ==========================================
    # REWARD
    # ==========================================

    def calculate_reward(self):

        if self.agent_id not in traci.vehicle.getIDList():

            return -100

        reward = 0

        speed = traci.vehicle.getSpeed(self.agent_id)

        waiting = traci.vehicle.getWaitingTime(self.agent_id)

        # Reward for moving
        reward += speed

        # Penalty for waiting
        reward -= waiting * 0.1

        # Penalty for stopping
        if speed < 1:

            reward -= 5

        # Collision penalty
        collisions = traci.simulation.getCollidingVehiclesIDList()

        if self.agent_id in collisions:

            reward -= 100

        return reward

    # ==========================================
    # CLOSE
    # ==========================================

    def close(self):

        if traci.isLoaded():

            traci.close()