import numpy as np
from framework.post_proc import envelope
import matplotlib.pyplot as plt

has_impedance_matching = True
mode = "NN"

aux_filename = "with_imp" if has_impedance_matching else "no_imp"
sscan_file = f"{mode}_{aux_filename}.npy"
sscan = np.load(sscan_file)

alpha_max_scan = np.pi / 4.0
alpha_min_scan = -np.pi / 4.0
scan_angles = np.linspace(alpha_min_scan, alpha_max_scan, 50)

sscan_env = envelope(sscan, axis=0)

max_values = []
for i in range(50):
    max_values.append(np.amax(sscan_env[:, :, i]))

max_values = np.array(max_values)

plt.figure()
plt.plot(scan_angles, max_values, "o")
plt.grid(True)
plt.show()

sscan_log = np.log10(sscan_env + 1e-6)
sscan_log = np.squeeze(sscan_log)

aux_title = "With impedance matching" if has_impedance_matching else "Without impedance matching"

# print("plotting...")
# plt.figure()
# plt.imshow(
#     sscan_log,
#     extent=[np.rad2deg(alpha_min_scan), np.rad2deg(alpha_max_scan), sim.tspan[-1] * 1e6, sim.tspan[0] * 1e6],  # Use your scan angle limits
#     aspect="auto",
#     interpolation="none",
#     cmap="jet",
# )
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Time (us)")
# plt.title(f"(Log Scale) S-Scan | Mode {mode} | {aux_title} | Max. value: {np.amax(sscan_env):.2f}")
# # plt.title(f"(Log Scale) S-Scan | Sum of all modes with Imp. Matching | Max. value: {np.amax(sscan_env):.2f}")
# plt.show()
