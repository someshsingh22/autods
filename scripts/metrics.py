import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

# Path to the MCTS log file
MCTS_LOG_PATH = os.path.expanduser("./nls_incarceration_big/mcts_nodes.json")

with open(MCTS_LOG_PATH, 'r') as f:
    nodes = json.load(f)

# Sort nodes by creation_index
nodes_sorted = sorted(nodes, key=lambda x: x['creation_index'])

# Remove the node with level 0 (if any)
nodes_sorted = [n for n in nodes_sorted if n.get('level', None) != 0]

# Calculate time deltas (in seconds) between consecutive nodes
prev_time = datetime.fromisoformat(nodes_sorted[0]['timestamp'])
time_deltas = []
for node in nodes_sorted:
    node_time = datetime.fromisoformat(node['timestamp'])
    delta = (node_time - prev_time).total_seconds()
    time_deltas.append(delta)
    prev_time = node_time

# Exclude the first delta (should be zero)
time_deltas = time_deltas[1:]

# Uncomment to  remove outliers (set a threshold)
#time_deltas = [delta for delta in time_deltas if delta <= 5000]

# Print statistics
mean = np.mean(time_deltas)
median = np.median(time_deltas)
q1 = np.percentile(time_deltas, 25)
q3 = np.percentile(time_deltas, 75)
variance = np.var(time_deltas)
std = np.std(time_deltas)
min_val = np.min(time_deltas)
max_val = np.max(time_deltas)
count = len(time_deltas)
print(f"Summary statistics for node execution times (seconds):")
print(f"  Count:    {count}")
print(f"  Mean:     {mean:.3f}")
print(f"  Median:   {median:.3f}")
print(f"  Min:      {min_val:.3f}")
print(f"  Q1:       {q1:.3f}")
print(f"  Q3:       {q3:.3f}")
print(f"  Max:      {max_val:.3f}")
print(f"  Variance: {variance:.3f}")
print(f"  Std Dev:  {std:.3f}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(time_deltas, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Time (seconds)')
plt.ylabel('Count')
plt.title('Node Execution Times')
plt.tight_layout()
plt.show()
