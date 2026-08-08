from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import TrafficEnvironment

# Create Environment
env = TrafficEnvironment()

# Monitor training
env = Monitor(env)

# Create PPO Model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

# Train
model.learn(
    total_timesteps=20000
)

# Save Model
model.save("models/traffic_agent")

print("Training Completed!")

env.close()