import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file 
csv_file = "./data/carla_training_metrics_fix_7.csv"


df = pd.read_csv(csv_file)


if "Episode" not in df.columns or "Reward" not in df.columns:
    raise ValueError("CSV file must contain 'episode' and 'reward' columns.")


window_size = 100  
df["avg_reward"] = df["Reward"].rolling(window=window_size).mean()
df["avg_loss"] = df["Loss"].rolling(window=window_size).mean()
# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Reward"], label="Episode Reward", alpha=0.3)
plt.plot(df["Episode"], df["avg_reward"], label=f"Moving Average ({window_size} episodes)", color="red")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Performance: Reward per Episode")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Loss"], label="Episode Loss", alpha=0.3)
plt.plot(df["Episode"], df["avg_loss"], label=f"Moving Average ({window_size} episodes)", color="red")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Performance: Loss per Episode")
plt.legend()
plt.grid()
plt.show()
