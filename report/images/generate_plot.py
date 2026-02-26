import json
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/Users/Edoardo/NNDL-project/report/images/pianogan_working_test_version_19.json', 'r') as f:
    data = json.load(f)

# Extract steps and values
steps = [item[1] for item in data]
values = [item[2] for item in data]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(steps, values, marker='o', linestyle='-', color='b', label='Val Wasserstein Distance')
plt.title('Validation Wasserstein Distance over Steps')
plt.xlabel('Training Steps')
plt.ylabel('Wasserstein Distance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the plot
plt.savefig('/Users/Edoardo/NNDL-project/report/images/val_w_distance.png', dpi=300, bbox_inches='tight')
print("Plot saved to /Users/Edoardo/NNDL-project/report/images/val_w_distance.png")
