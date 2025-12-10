import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------
# EDIT THESE VALUES ONLY — everything else stays the same
# --------------------------------------------------------
shapes = ["Sphere", "Cube", "Cylinder", "Cone"]

# Success counts (change these only)
manual_success = [100, 100, 5, 8]      # out of N trials
ppo_success    = [46, 13, 15, 20]      # out of N trials

N = 100  # total trials per shape (update if needed)

# Convert to rates
manual = np.array(manual_success) / N
ppo    = np.array(ppo_success) / N
# --------------------------------------------------------

x = np.arange(len(shapes))
width = 0.35

plt.figure(figsize=(8, 4))

bars1 = plt.bar(x - width/2, manual, width, label="Manual", color="#4C72B0")
bars2 = plt.bar(x + width/2, ppo, width, label="PPO", color="#DD8452")

# Add text labels
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
             ha='center', va='bottom', fontsize=10)

for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Success Rate")
plt.title("Multi-Object Search Success Rate per Shape")
plt.xticks(x, shapes)
plt.ylim(0, 1.15)
plt.legend()
plt.tight_layout()

# Save figure
save_path = "rl_success_rates.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved bar chart to: {save_path}")
