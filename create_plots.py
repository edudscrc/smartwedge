import numpy as np
from framework.post_proc import envelope
import matplotlib.pyplot as plt

# NN_no_imp_sscan = np.load("./NN_no_imp.npy")
NN_sscan = np.load("./NN_with_imp.npy")
# NR_sscan = np.load("./NR_with_imp.npy")
# RN_sscan = np.load("./RN_with_imp.npy")
# RR_sscan = np.load("./RR_with_imp.npy")

# NN_no_imp_sscan = envelope(NN_no_imp_sscan, axis=0)
NN_sscan = envelope(NN_sscan, axis=0)
# NR_sscan = envelope(NR_sscan, axis=0)
# RN_sscan = envelope(RN_sscan, axis=0)
# RR_sscan = envelope(RR_sscan, axis=0)

# SUM_sscan = NN_sscan + NR_sscan + RN_sscan + RR_sscan

num_focus = 51

alpha_max_scan = np.pi / 4.0
alpha_min_scan = -np.pi / 4.0
scan_angles = np.linspace(alpha_min_scan, alpha_max_scan, num_focus)

# max_NN_no_imp_sscan = []
max_NN_sscan = []
# max_NR_sscan = []
# max_RN_sscan = []
# max_RR_sscan = []
# max_sum_sscan = []
for i in range(num_focus):
    # max_NN_no_imp_sscan.append(np.amax(NN_no_imp_sscan[:, :, i]))
    max_NN_sscan.append(np.amax(NN_sscan[:, :, i]))
    # max_NR_sscan.append(np.amax(NR_sscan[:, :, i]))
    # max_RN_sscan.append(np.amax(RN_sscan[:, :, i]))
    # max_RR_sscan.append(np.amax(RR_sscan[:, :, i]))
    # max_sum_sscan.append(np.amax(SUM_sscan[:, :, i]))
# max_NN_no_imp_sscan = np.array(max_NN_no_imp_sscan)
max_NN_sscan = np.array(max_NN_sscan)
# max_NR_sscan = np.array(max_NR_sscan)
# max_RN_sscan = np.array(max_RN_sscan)
# max_RR_sscan = np.array(max_RR_sscan)
# max_sum_sscan = np.array(max_sum_sscan)

plt.figure()
# plt.plot(np.rad2deg(scan_angles), max_NN_no_imp_sscan, "o", label="NN sem camada")
plt.plot(np.rad2deg(scan_angles), max_NN_sscan, "o", label="NN com camada")
# plt.plot(np.rad2deg(scan_angles), max_NR_sscan, "o", label="NR")
# plt.plot(np.rad2deg(scan_angles), max_RN_sscan, "o", label="RN")
# plt.plot(np.rad2deg(scan_angles), max_RR_sscan, "o", label="RR")
plt.grid(True)
plt.ylabel("Amplitude")
plt.xlabel("Angle (degrees)")
plt.legend()
plt.title("Simulação com 51 falhas")
plt.show()

# plt.figure()
# plt.plot(np.rad2deg(scan_angles), max_NN_no_imp_sscan, "o", label="NN sem camada")
# plt.plot(np.rad2deg(scan_angles), max_sum_sscan, "o", label="NN (com camada) + NR + RN + RR")
# plt.grid(True)
# plt.ylabel("Amplitude")
# plt.xlabel("Angle (degrees)")
# plt.legend()
# plt.title("Simulação com 50 falhas")
# plt.show()

# sscan_log = np.log10(sscan_env + 1e-6)
# sscan_log = np.squeeze(sscan_log)

# aux_title = "With impedance matching" if has_impedance_matching else "Without impedance matching"

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
