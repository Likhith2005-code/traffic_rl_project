"""
test.py
================================================================================
QUICK DEMO / SANITY-CHECK SCRIPT for the 3-agent SUMO MARL system.

Loads the three independently-trained PPO models and drives them through ONE
joint SUMO episode (optionally with sumo-gui so you can visually watch the
three color-coded RL vehicles: agent1=red, agent2=blue, agent3=green).

Because `MultiAgentTrafficEnv` respawns an agent immediately after it
finishes a sub-episode (arrival or collision), a single joint run can contain
several completed sub-episodes per agent - this script reports both
per-step and per-sub-episode statistics.

For a statistically rigorous, multi-episode evaluation with saved plots, use
`evaluate.py` instead. This script is meant for a fast, human-readable check.

Usage
-----
    python test.py --gui
    python test.py --models-dir ./models --model-suffix _final
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from environment import MultiAgentTrafficEnv, AGENT_IDS


def load_models(models_dir, suffix):
    models = {}
    for agent_id in AGENT_IDS:
        path = f"{models_dir}/ppo_{agent_id}{suffix}"
        models[agent_id] = PPO.load(path)
        print(f"Loaded model for {agent_id}: {path}.zip")
    return models


def run_episode(env, models, deterministic=True, verbose=True):
    obs, infos = env.reset()

    stats = {
        aid: {
            "total_reward": 0.0,
            "steps_alive": 0,
            "speed_sum": 0.0,
            "arrivals": 0,
            "collisions": 0,
        }
        for aid in AGENT_IDS
    }

    start_time = time.time()
    step_count = 0
    done_all = False

    while not done_all:
        actions = {}
        for agent_id in env.agents:
            action, _states = models[agent_id].predict(obs[agent_id], deterministic=deterministic)
            actions[agent_id] = int(action)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        for agent_id in AGENT_IDS:
            if agent_id not in rewards:
                continue
            s = stats[agent_id]
            s["total_reward"] += rewards[agent_id]
            s["steps_alive"] += 1
            s["speed_sum"] += float(obs[agent_id][0]) if agent_id in obs else 0.0

            event = infos.get(agent_id, {}).get("event")
            if event == "arrived":
                s["arrivals"] += 1
            elif event == "collision":
                s["collisions"] += 1

        done_all = len(env.agents) == 0
        if verbose and step_count % 200 == 0:
            print(f"  step {step_count} ...")

    wall_clock_time = time.time() - start_time
    return stats, step_count, wall_clock_time


def print_report(stats, step_count, wall_clock_time):
    print("\n===================== MULTI-AGENT TEST RESULTS =====================")
    print(f"Simulation steps: {step_count}   Wall-clock time: {wall_clock_time:.2f}s")
    print("----------------------------------------------------------------------")
    header = f"{'Agent':<10}{'TotalReward':>14}{'AvgSpeed':>12}{'Arrivals':>10}{'Collisions':>12}"
    print(header)
    for agent_id, s in stats.items():
        avg_speed = s["speed_sum"] / max(s["steps_alive"], 1)
        print(
            f"{agent_id:<10}{s['total_reward']:>14.2f}{avg_speed:>12.2f}"
            f"{s['arrivals']:>10d}{s['collisions']:>12d}"
        )
    print("========================================================================\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run one demo episode of the trained 3-agent policy.")
    parser.add_argument("--sumo-config", type=str, default="simulation/config.sumocfg")
    parser.add_argument("--gui", action="store_true", help="Watch the episode in sumo-gui.")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--model-suffix", type=str, default="_final")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of arg-max.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    models = load_models(args.models_dir, args.model_suffix)
    env = MultiAgentTrafficEnv(
        sumo_config=args.sumo_config,
        use_gui=args.gui,
        max_steps=args.max_steps,
    )

    try:
        stats, step_count, wall_clock_time = run_episode(
            env, models, deterministic=not args.stochastic
        )
        print_report(stats, step_count, wall_clock_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
