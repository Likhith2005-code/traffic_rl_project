"""
train.py
================================================================================
MULTI-AGENT TRAINING SCRIPT - Independent PPO (IPPO) for 3 SUMO agents
================================================================================

ARCHITECTURAL NOTE (read this first)
--------------------------------------------------------------------------------
Stable-Baselines3 (SB3) does not natively support multi-agent environments:
`PPO.learn()` expects one `gym.Env` (or a `VecEnv` of IDENTICAL, INDEPENDENT
copies of the same single-agent env). Our world is the opposite: THREE
different agents share ONE SUMO/TraCI simulation clock, so their experience
streams cannot be produced by three separate, independently-stepped envs.

Two standard ways exist to reconcile this:
  (A) Parameter sharing: convert the PettingZoo env with SuperSuit into a
      VecEnv and train ONE shared policy for all agents. Simple, but the
      three agents would NOT have independent policies/weights.
  (B) Independent PPO (IPPO): give every agent its OWN PPO model (own
      network, own optimizer, own rollout buffer, own hyperparameters), but
      drive all three through a single shared-simulation timestep loop so
      the underlying SUMO clock only ever advances once per joint step.

This project explicitly asked for three INDEPENDENT agents, so we implement
(B): a manual training loop built directly on top of SB3's own building
blocks (`ActorCriticPolicy`, `RolloutBuffer`, `PPO.train()`). This is
equivalent to what `OnPolicyAlgorithm.collect_rollouts()` does internally,
just orchestrated across three models against one shared environment instead
of one model against one VecEnv.

Every agent still gets:
  - its own PPO neural network (policy + value function)
  - its own rollout buffer / advantage estimates (GAE)
  - its own reward stream and independent gradient updates
  - its own saved model file and its own TensorBoard run

Usage
-----
    python train.py --timesteps 200000 --n-steps 1024 --gui
"""

import os
import argparse

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from environment import MultiAgentTrafficEnv, SingleAgentSpaceEnv, AGENT_IDS


# ==============================================================================
# Independent PPO (IPPO) trainer
# ==============================================================================
class IndependentMultiAgentPPOTrainer:
    """
    Owns one `PPO` model PER agent and one shared `MultiAgentTrafficEnv`, and
    runs a manual rollout-collection + update loop that keeps every agent's
    learning completely independent while sharing a single SUMO simulation.
    """

    def __init__(
        self,
        sumo_config: str,
        use_gui: bool,
        max_steps_per_episode: int,
        n_steps: int,
        learning_rate: float,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        tensorboard_dir: str,
        models_dir: str,
        seed: int,
    ):
        self.n_steps = n_steps
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # ---- The ONE shared multi-agent SUMO environment ----
        self.env = MultiAgentTrafficEnv(
            sumo_config=sumo_config,
            use_gui=use_gui,
            max_steps=max_steps_per_episode,
            seed=seed,
        )

        # ---- One fully independent PPO model per agent ----
        self.models = {}
        for i, agent_id in enumerate(AGENT_IDS):
            placeholder_env = SingleAgentSpaceEnv(agent_id)
            self.models[agent_id] = PPO(
                policy="MlpPolicy",
                env=placeholder_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                tensorboard_log=tensorboard_dir,
                verbose=2,
                seed=seed + i,  # distinct seeds -> distinct exploration per agent
            )

    # --------------------------------------------------------------------
    def _setup_all_models(self, total_timesteps):
        """Prepares each PPO model's logger / internal bookkeeping via SB3's
        own `_setup_learn`, then overwrites the (meaningless) placeholder
        observation with the REAL first observation from the shared env."""
        obs, _infos = self.env.reset()
        callbacks = {}

        for agent_id, model in self.models.items():
            # SB3's `_setup_learn` configures the TensorBoard logger, resets
            # `num_timesteps`, and initializes `_last_obs` / `_last_episode_starts`.
            # The `progress_bar` kwarg was added in later SB3 versions, so we
            # fall back gracefully on older installs.
            try:
                _total, callback = model._setup_learn(
                    total_timesteps=total_timesteps,
                    callback=None,
                    reset_num_timesteps=True,
                    tb_log_name=agent_id,
                    progress_bar=False,
                )
            except TypeError:
                _total, callback = model._setup_learn(
                    total_timesteps, None, True, agent_id
                )
            callback.on_training_start(locals(), globals())
            callbacks[agent_id] = callback

            # Overwrite the placeholder reset observation with the REAL one.
            model._last_obs = np.expand_dims(obs[agent_id].astype(np.float32), axis=0)
            model._last_episode_starts = np.array([True])

        return callbacks

    # --------------------------------------------------------------------
    def train(self, total_timesteps, log_every_updates=1, checkpoint_every_updates=10):
        callbacks = self._setup_all_models(total_timesteps)
        n_updates = max(total_timesteps // self.n_steps, 1)
        global_timesteps = 0

        print(f"\n=== Starting Independent PPO training for {AGENT_IDS} ===")
        print(f"Updates: {n_updates}  |  n_steps/update: {self.n_steps}  |  "
              f"total_timesteps: {total_timesteps}\n")

        for update in range(1, n_updates + 1):
            for model in self.models.values():
                model.rollout_buffer.reset()
                model.policy.set_training_mode(False)

            rollout_rewards = {aid: [] for aid in AGENT_IDS}
            rollout_events = {aid: {"arrived": 0, "collision": 0} for aid in AGENT_IDS}

            for _ in range(self.n_steps):
                actions_np = {}
                actions_t, values_t, log_probs_t = {}, {}, {}

                # ---- Every agent chooses its action from ITS OWN policy ----
                for agent_id, model in self.models.items():
                    obs_tensor = obs_as_tensor(model._last_obs, model.device)
                    with th.no_grad():
                        acts, vals, log_prob = model.policy(obs_tensor)
                    actions_t[agent_id] = acts
                    values_t[agent_id] = vals
                    log_probs_t[agent_id] = log_prob
                    actions_np[agent_id] = int(acts.cpu().numpy().reshape(-1)[0])

                # ---- Advance the SHARED SUMO simulation exactly once ----
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions_np)

                joint_truncated = all(truncations.get(aid, False) for aid in AGENT_IDS)

                for agent_id, model in self.models.items():
                    done = bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))

                    model.rollout_buffer.add(
                        model._last_obs,
                        actions_t[agent_id].cpu().numpy(),
                        np.array([rewards[agent_id]], dtype=np.float32),
                        model._last_episode_starts,
                        values_t[agent_id],
                        log_probs_t[agent_id],
                    )

                    model._last_obs = np.expand_dims(
                        next_obs[agent_id].astype(np.float32), axis=0
                    )
                    model._last_episode_starts = np.array([done])
                    model.num_timesteps += 1

                    rollout_rewards[agent_id].append(rewards[agent_id])
                    event = infos.get(agent_id, {}).get("event")
                    if event in rollout_events[agent_id]:
                        rollout_events[agent_id][event] += 1

                global_timesteps += 1

                if joint_truncated:
                    # The whole shared SUMO run finished its fixed horizon:
                    # restart the simulation for all three agents together.
                    reset_obs, _ = self.env.reset()
                    for agent_id, model in self.models.items():
                        model._last_obs = np.expand_dims(
                            reset_obs[agent_id].astype(np.float32), axis=0
                        )
                        model._last_episode_starts = np.array([True])

            # ---- End of rollout: bootstrap value + independent PPO update ----
            for agent_id, model in self.models.items():
                with th.no_grad():
                    obs_tensor = obs_as_tensor(model._last_obs, model.device)
                    last_values = model.policy.predict_values(obs_tensor)
                model.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_values, dones=model._last_episode_starts
                )

                model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
                model.policy.set_training_mode(True)
                model.train()

                mean_reward = float(np.mean(rollout_rewards[agent_id]))
                model.logger.record("rollout/mean_reward_per_step", mean_reward)
                model.logger.record("rollout/arrivals", rollout_events[agent_id]["arrived"])
                model.logger.record("rollout/collisions", rollout_events[agent_id]["collision"])
                model.logger.dump(step=model.num_timesteps)

            if update % log_every_updates == 0:
                summary = "  ".join(
                    f"{aid}: mean_r={np.mean(rollout_rewards[aid]):.2f} "
                    f"arrivals={rollout_events[aid]['arrived']} "
                    f"collisions={rollout_events[aid]['collision']}"
                    for aid in AGENT_IDS
                )
                print(f"[Update {update}/{n_updates}] timesteps={global_timesteps}  {summary}")

            if update % checkpoint_every_updates == 0:
                self.save(suffix=f"_checkpoint_{global_timesteps}")

        for agent_id, model in self.models.items():
            callbacks[agent_id].on_training_end()

        self.env.close()
        print("\n=== Training complete ===")

    # --------------------------------------------------------------------
    def save(self, suffix=""):
        for agent_id, model in self.models.items():
            path = os.path.join(self.models_dir, f"ppo_{agent_id}{suffix}")
            model.save(path)
        print(f"Models saved to '{self.models_dir}' (suffix='{suffix}')")


# ==============================================================================
# CLI entry point
# ==============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train 3 independent PPO agents in shared SUMO traffic.")
    parser.add_argument("--sumo-config", type=str, default="simulation/config.sumocfg")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui instead of headless sumo.")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Total env timesteps per agent.")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length before each PPO update.")
    parser.add_argument("--max-steps-per-episode", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--tensorboard-dir", type=str, default="./tensorboard_logs")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()

    trainer = IndependentMultiAgentPPOTrainer(
        sumo_config=args.sumo_config,
        use_gui=args.gui,
        max_steps_per_episode=args.max_steps_per_episode,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_dir=args.tensorboard_dir,
        models_dir=args.models_dir,
        seed=args.seed,
    )

    try:
        trainer.train(
            total_timesteps=args.timesteps,
            checkpoint_every_updates=args.checkpoint_every,
        )
    finally:
        trainer.save(suffix="_final")


if __name__ == "__main__":
    main()
