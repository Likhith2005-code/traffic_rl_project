from stable_baselines3 import PPO
from environment import TrafficEnvironment

# Create environment
env = TrafficEnvironment()

# Load trained model
model = PPO.load("models/traffic_agent")

# Start environment
obs, info = env.reset()

done = False

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

print("Episode Finished!")

env.close()