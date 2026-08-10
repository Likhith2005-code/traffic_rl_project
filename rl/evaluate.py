"""
evaluate.py
================================================================================
STATISTICAL EVALUATION SCRIPT for the 3-agent SUMO MARL system.

Runs the trained policies for N full joint episodes (headless, deterministic),
aggregates per-agent statistics across all episodes, prints a summary table,
saves a CSV log of every episode, and saves comparison plots (reward,
average speed, arrivals, collisions) to disk - useful for a final-year
project report / presentation.

Usage
-----
    python evaluate.py --episodes 20
    python evaluate.py --episodes 10 --models-dir ./models --model-suffix _final
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs without a display
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from environment import MultiAgentTrafficEnv, AGENT_IDS


# ==============================================================================
# Model loading
# ==============================================================================
def load_models(models_dir, suffix):
    models = {}
    for agent_id in AGENT_IDS:
        path = f"{models_dir}/ppo_{agent_id}{suffix}"
        models[agent_id] = PPO.load(path)
        print(f"Loaded model for {agent_id}: {path}.zip")
    return models


# ==============================================================================
# Single-episode rollout (deterministic policy, no exploration noise)
# ==============================================================================
def run_one_episode(env, models):
    obs, infos = env.reset()

    per_agent = {
        aid: {"reward": 0.0, "steps": 0, "speed_sum": 0.0, "arrivals": 0, "collisions": 0}
        for aid in AGENT_IDS
    }

    step_count = 0
    while env.agents:
        actions = {
            agent_id: int(models[agent_id].predict(obs[agent_id], deterministic=True)[0])
            for agent_id in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        for agent_id in AGENT_IDS:
            if agent_id not in rewards:
                continue
            a = per_agent[agent_id]
            a["reward"] += rewards[agent_id]
            a["steps"] += 1
            a["speed_sum"] += float(obs[agent_id][0]) if agent_id in obs else 0.0
            event = infos.get(agent_id, {}).get("event")
            if event == "arrived":
                a["arrivals"] += 1
            elif event == "collision":
                a["collisions"] += 1

    return per_agent, step_count


# ==============================================================================
# Multi-episode evaluation loop
# ==============================================================================
def evaluate(env, models, n_episodes):
    episode_records = []  # list of dicts, one per (episode, agent)

    for ep in range(1, n_episodes + 1):
        per_agent, step_count = run_one_episode(env, models)
        for agent_id, a in per_agent.items():
            episode_records.append(
                {
                    "episode": ep,
                    "agent": agent_id,
                    "reward": a["reward"],
                    "avg_speed": a["speed_sum"] / max(a["steps"], 1),
                    "arrivals": a["arrivals"],
                    "collisions": a["collisions"],
                }
            )
        print(f"[Episode {ep}/{n_episodes}] steps={step_count}  "
              + "  ".join(f"{aid}: r={per_agent[aid]['reward']:.1f}" for aid in AGENT_IDS))

    return episode_records


# ==============================================================================
# Reporting: console summary + CSV + plots
# ==============================================================================
def summarize(records):
    summary = {}
    for agent_id in AGENT_IDS:
        rows = [r for r in records if r["agent"] == agent_id]
        summary[agent_id] = {
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "std_reward": float(np.std([r["reward"] for r in rows])),
            "mean_speed": float(np.mean([r["avg_speed"] for r in rows])),
            "total_arrivals": int(np.sum([r["arrivals"] for r in rows])),
            "total_collisions": int(np.sum([r["collisions"] for r in rows])),
        }
    return summary


def print_summary(summary, n_episodes):
    print("\n===================== EVALUATION SUMMARY (%d episodes) =====================" % n_episodes)
    header = f"{'Agent':<10}{'MeanReward':>14}{'StdReward':>12}{'MeanSpeed':>12}{'Arrivals':>10}{'Collisions':>12}"
    print(header)
    for agent_id, s in summary.items():
        print(
            f"{agent_id:<10}{s['mean_reward']:>14.2f}{s['std_reward']:>12.2f}"
            f"{s['mean_speed']:>12.2f}{s['total_arrivals']:>10d}{s['total_collisions']:>12d}"
        )
    print("================================================================================\n")


def save_csv(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["episode", "agent", "reward", "avg_speed", "arrivals", "collisions"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Per-episode CSV log saved to: {out_path}")


def save_plots(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("reward", "Episode Reward"),
        ("avg_speed", "Average Speed (m/s)"),
        ("arrivals", "Arrivals"),
        ("collisions", "Collisions"),
    ]

    for metric_key, metric_label in metrics:
        plt.figure(figsize=(8, 5))
        for agent_id in AGENT_IDS:
            rows = sorted((r for r in records if r["agent"] == agent_id), key=lambda r: r["episode"])
            episodes = [r["episode"] for r in rows]
            values = [r[metric_key] for r in rows]
            plt.plot(episodes, values, marker="o", label=agent_id)
        plt.xlabel("Episode")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label} per Episode (per agent)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(out_dir, f"{metric_key}_per_episode.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_path}")


# ==============================================================================
# CLI entry point
# ==============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Statistically evaluate the trained 3-agent policy.")
    parser.add_argument("--sumo-config", type=str, default="simulation/config.sumocfg")
    parser.add_argument("--gui", action="store_true", help="Evaluate with sumo-gui (slower; usually leave off).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--model-suffix", type=str, default="_final")
    parser.add_argument("--results-dir", type=str, default="./eval_results")
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
        records = evaluate(env, models, args.episodes)
    finally:
        env.close()

    summary = summarize(records)
    print_summary(summary, args.episodes)
    save_csv(records, os.path.join(args.results_dir, "episode_log.csv"))
    save_plots(records, args.results_dir)


if __name__ == "__main__":
    main()
