import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.noise import NormalActionNoise

# OU-Rauschen konfigurieren
mean = np.array([0.0])
std_dev = np.array([1.0])
ou = OrnsteinUhlenbeckActionNoise(mean, std_dev)

# Rauschwerte sammeln
values = []
for _ in range(1000):
    value = ou()
    values.append(value[0])

# Plot
plt.figure(figsize=(10, 4))
plt.plot(values, label='OU Noise')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Ornstein-Uhlenbeck Noise über Zeit")
plt.xlabel("Zeitschritt")
plt.ylabel("Rauschwert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
