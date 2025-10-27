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


CP_WATER, RHO_WATER = 1483, 1000
CP_STEEL, CS_STEEL, RHO_STEEL = 5900, 2950, 7750
CP_ALUMINIUM, CS_ALUMINIUM, RHO_ALUMINIUM = 6300, 3150, 2700

c1 = np.float32(CP_ALUMINIUM)
c2 = np.float32(CP_WATER)
c3 = np.float32(CP_STEEL)

rho_aluminium = np.float32(RHO_ALUMINIUM)
rho_water = np.float32(RHO_WATER)
rho_steel = np.float32(RHO_STEEL)

d = np.float32(170e-3)              # Height of transducer (m)
alpha_max = np.float32(np.pi / 4.0) # Maximum sectorial angle (rad)
alpha_0 = np.float32(0.0)           # Reference angle (boundary condition) (rad)
h0 = np.float32(91.03e-3 + 1e-3)    # Length h chosen at the reference angle (m)

has_impedance_matching = True

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=has_impedance_matching)

outer_radius = np.float32(139.82e-3 / 2)
wall_width = np.float32(16.23e-3)
pipeline = Pipeline(outer_radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)

num_elements = 64

transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=num_elements, fc=5e6)
transducer.zt += acoustic_lens.d

raytracer = RayTracing(acoustic_lens, pipeline, transducer, final_amplitude=True, directivity=True)

mode = 'RR'

# 1. DEFINE YOUR SCAN ANGLES AND FOCAL DEPTH
# (Instead of just one 'arg')
# -----------------------------------------------------------------
num_angles = 181  # Or however many lines you want in your S-Scan
alpha_max_scan = np.pi / 4.0
alpha_min_scan = -np.pi / 4.0
scan_angles = np.linspace(alpha_min_scan, alpha_max_scan, num_angles)

# Use a constant focal radius for all angles
focal_radius = pipeline.inner_radius + 10e-3 
center_elem = transducer.num_elem // 2

# Initialize empty delay law matrices
all_delay_laws = np.zeros((transducer.num_elem, num_angles), dtype=np.float32)

print("Calculating delay laws for S-Scan...")
# 2. LOOP THROUGH EACH ANGLE AND CALCULATE DELAYS
# -----------------------------------------------------------------
for i, angle_rad in enumerate(scan_angles):
    print(f"Solving {i}/{num_angles - 1}")
    # Calculate the (x, z) coordinate for the focal point at this angle
    xf = focal_radius * np.sin(angle_rad)
    zf = focal_radius * np.cos(angle_rad)
    
    # Apply the same rotation and centering as you did for 'arg'
    xf_rot, zf_rot = rotate_point((xf, zf), theta_rad=0)
    xf_final = xf_rot + pipeline.xcenter
    zf_final = zf_rot + pipeline.zcenter
    
    # Solve for the TOFs to this one point
    # Note: We only need 'tofs', so we ignore amps and sol
    tofs_i, _, _ = raytracer.solve(xf_final, zf_final, mode=mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3)
    
    # Calculate the delay law relative to the center element
    # tofs_i has shape (64, 1), so we use [:, 0] to get a (64,) array
    delay_law_i = tofs_i[center_elem, 0] - tofs_i[:, 0]
    
    # Store this delay law as one column in our matrix
    all_delay_laws[:, i] = delay_law_i

print("Delay law calculation complete.")

# 3. USE THE NEW DELAY LAW MATRIX IN THE SIMULATOR
# -----------------------------------------------------------------
simulation_parameters = {
    "surface_echoes": True,
    "gate_end": 80e-6,
    "gate_start": 30e-6,
    "fs": 64.5e6, # Hz
    "response_type": "s-scan",
    "emission_delaylaw": all_delay_laws,     # <-- USE THE FULL MATRIX
    "reception_delaylaw": all_delay_laws    # <-- USE THE FULL MATRIX
}

sim = Simulator(simulation_parameters, [raytracer], verbose=True)

# 4. ADD THE REFLECTOR YOU WANT TO IMAGE
# (This can be the same 'arg' as before)
# -----------------------------------------------------------------
focus_horizontal_offset = 0
arg = (
    focus_horizontal_offset,
    pipeline.inner_radius + 10e-3,
)
arg = rotate_point(arg, theta_rad=0)
arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)
sim.add_reflector(*arg, different_instances=False)

# 5. GET THE RESPONSE AND PLOT
# -----------------------------------------------------------------
sscan = sim.get_response(mode)

aux_filename = 'with_imp' if has_impedance_matching else 'no_imp'

np.save(f'{mode}_{aux_filename}.npy', sscan)

sscan_env = envelope(sscan, axis=0)
sscan_log = np.log10(sscan_env + 1e-6)

# The sscan shape will now be (num_samples, num_angles, 1)
# Squeeze it to (num_samples, num_angles) for imshow
sscan_log = np.squeeze(sscan_log)

aux_title = 'With impedance matching' if has_impedance_matching else 'Without impedance matching'

plt.figure()
plt.imshow(
    sscan_log,
    extent=[np.rad2deg(alpha_min_scan), np.rad2deg(alpha_max_scan), sim.tspan[-1] * 1e6, sim.tspan[0] * 1e6], # Use your scan angle limits
    aspect='auto', interpolation='none', cmap='jet'
)
plt.xlabel("Angle (degrees)")
plt.ylabel("Time (us)")
plt.title(f"(Log Scale) S-Scan | Mode {mode} | {aux_title} | Max. value: {np.amax(sscan_env):.2f}")
plt.show()
