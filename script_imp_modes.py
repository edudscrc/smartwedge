import numpy as np
import matplotlib.pyplot as plt
from acoustic_lens import AcousticLens
from raytracing import RayTracing
from pipeline import Pipeline
from transducer import Transducer
import matplotlib.ticker as ticker
from pathlib import Path
from framework.post_proc import envelope
from simulator import Simulator


def rotate_point(xy, theta_rad):
    x, y = xy
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return (x_rot, y_rot)


################
## PARAMETERS ##
################

CP_WATER, RHO_WATER = 1483, 1000
CP_STEEL, CS_STEEL, RHO_STEEL = 5900, 2950, 7750
CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM = 6300, 3150, 2700

c1 = np.float32(CP_ALUMINIUM)
c2 = np.float32(CP_WATER)
c3 = np.float32(CP_STEEL)

rho_aluminium = np.float32(RHO_ALUMINIUM)
rho_water = np.float32(RHO_WATER)
rho_steel = np.float32(RHO_STEEL)

d = np.float32(170e-3)                  # Height of transducer (m)
alpha_max = np.float32(np.pi / 4.0)     # Maximum sectorial angle (rad)
alpha_0 = np.float32(0.0)               # Reference angle (boundary condition) (rad)
h0 = np.float32(91.03e-3 + 1e-3)        # Length h chosen at the reference angle (m)

outer_radius = np.float32(139.82e-3 / 2)
wall_width = np.float32(16.23e-3)

########################
## Simulation Configs ##
########################

has_impedance_matching = True
num_elements = 64
num_angles = 181
num_reflectors = 51
mode = "NN"

alpha_step = 1e-5
dist_tol = 1.0
delta_alpha = 0.1

###########
## Setup ##
###########

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=has_impedance_matching)
pipeline = Pipeline(outer_radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)
transducer = Transducer(pitch=0.5e-3, bw=0.4, num_elem=num_elements, fc=5e6)
transducer.zt += acoustic_lens.d
raytracer = RayTracing(acoustic_lens, pipeline, transducer, final_amplitude=True, directivity=True)


########################################
## Define Scan Angles and Focal Depth ##
########################################

alpha_max_scan = np.pi / 4.0
alpha_min_scan = -np.pi / 4.0
scan_angles = np.linspace(alpha_min_scan, alpha_max_scan, num_angles)

# Use a constant focal radius for all angles
focal_radius = pipeline.inner_radius + 10e-3
center_elem = transducer.num_elem // 2

###########################
## Delay Law Calculation ##
###########################

# Calculate all (x, z) coordinates for all focal points at once
xf = focal_radius * np.sin(scan_angles)
zf = focal_radius * np.cos(scan_angles)

# Apply rotation and centering to all points
xf_rot, zf_rot = rotate_point((xf, zf), theta_rad=0)
xf_final = xf_rot + pipeline.xcenter
zf_final = zf_rot + pipeline.zcenter

print(f"--- Calculating S-Scan Delay Laws ---")
print(f"Solving for {num_angles} angles...")

# alpha_step:  Coarse grid step.
# dist_tol:     Error tolerance (mm).
# delta_alpha:  Window for the optimizer (radians).

# Solve for all angles in a single call
tofs, _, _ = raytracer.solve(xf_final, zf_final, mode=mode, 
                             alpha_step=alpha_step, 
                             dist_tol=dist_tol, 
                             delta_alpha=delta_alpha)

# 'tofs' will have shape (num_elem, num_angles)

# Calculate the delay law relative to the center element using broadcasting
# tofs[center_elem, :] has shape (181,)
# tofs has shape (64, 181)
# Result 'all_delay_laws' has shape (64, 181)
all_delay_laws = tofs[center_elem, :] - tofs

print("--- Delay Law Calculation Complete ---")

###################################################
## Use the new Delay Law Matrix in the Simulator ##
###################################################

simulation_parameters = {
    "surface_echoes": True,
    "gate_end": 80e-6,
    "gate_start": 30e-6,
    "fs": 64.5e6,
    "response_type": "s-scan",
    "emission_delaylaw": all_delay_laws,
    "reception_delaylaw": all_delay_laws,
}

sim = Simulator(simulation_parameters, [raytracer], verbose=True)

####################
## Add reflectors ##
####################

focus_radius = pipeline.inner_radius
focus_angle = np.linspace(alpha_min_scan, alpha_max_scan, num_reflectors)
xf_reflectors, zf_reflectors = focus_radius * np.sin(focus_angle), focus_radius * np.cos(focus_angle)
arg = (xf_reflectors, zf_reflectors)

print(f"--- Adding {len(xf_reflectors)} Reflectors ---")
sim.add_reflector(*arg, different_instances=True)

###############################
## Get the Response and Plot ##
###############################

print(f"--- Running Main Simulation (Mode: {mode}) ---")

sscan = sim.get_response(mode, 
                         alpha_step=alpha_step, 
                         dist_tol=dist_tol,
                         delta_alpha=delta_alpha)

aux_filename = "with_imp" if has_impedance_matching else "no_imp"

print(f"--- Simulation Complete. Saving data... ---")
print(f"Final S-Scan shape: {sscan.shape}")

np.save(f"{mode}_{aux_filename}.npy", sscan)

# sscan_env = envelope(sscan, axis=0)
# sscan_log = np.log10(sscan_env + 1e-6)

# The sscan shape will now be (num_samples, num_angles, 1)
# Squeeze it to (num_samples, num_angles) for imshow
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
