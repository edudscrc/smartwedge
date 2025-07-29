import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pipe_lens_imaging.acoustic_lens import AcousticLens
from pipe_lens_imaging.focus_raytracer import FocusRayTracer
from pipe_lens_imaging.pipeline import Pipeline
from pipe_lens_imaging.transducer import Transducer
from numpy import pi, sin, cos
import matplotlib.ticker as ticker
#
from pipe_lens_imaging.simulator import Simulator

def rotate_point(xy, theta_rad):
    x, y = xy
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return (x_rot, y_rot)

linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Monaspace Neon",
    "font.size": 10,
    "font.weight": "normal",
})

# Acoustic lens parameters:
c1 = 6332.93 # in (m/s)
c2 = 1430.00 # in (m/s)q
d = 170e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 91.03e-3 + 1e-3 # in (m)
rho_aluminium = 2.710 #kg / m
rho_water = 1.000
rho_steel = 7.850

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=True)

# Pipeline-related parameters:
radius = 139.82e-3/2
wall_width = 16.23e-3
inner_radius = (radius - wall_width)
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3, rho_steel, xcenter=-5e-3, zcenter=-5e-3)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

arg = (
    0,
    inner_radius + 10e-3,
)
arg = rotate_point(arg, theta_rad=0)

arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

tofs, amps = raytracer.solve(*arg)

print(f'{tofs.shape = }')
print(f'{amps['transmission_loss'].shape = }')

# print(tofs.shape)
# print(amps["transmission_loss"].shape)

sol = raytracer._solve(*arg)

#%% Extract refraction/reflection points:

extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

xlens, zlens = extract_pts(sol, 'xlens'), extract_pts(sol, 'zlens')
if acoustic_lens.impedance_matching is not None:
    ximp, zimp = extract_pts(sol, 'ximp'), extract_pts(sol, 'zimp')
    xlens_2, zlens_2 = extract_pts(sol, 'xlens_2'), extract_pts(sol, 'zlens_2')
    ximp_2, zimp_2 = extract_pts(sol, 'ximp_2'), extract_pts(sol, 'zimp_2')
xpipe, zpipe = extract_pts(sol, 'xpipe'), extract_pts(sol, 'zpipe')
xf, zf = arg

#%% Debug plots:

firing_angles = extract_pts(sol, 'firing_angle')
pipe_incidence_angles = extract_pts(sol, 'interface_23')

plt.figure(figsize=(8,4))
plt.plot(np.arange(transducer.num_elem), np.degrees(pipe_incidence_angles[0::2]), 'o')
plt.grid()
plt.show()
#%%
plt.figure(figsize=(5, 7))
plt.title("Case A: Focusing on point inside the pipe wall.")
# Plot fix components:
plt.plot(transducer.xt, transducer.zt, 'sk')
plt.plot(0, 0, 'or')
plt.plot(0, acoustic_lens.d, 'or'   )
plt.plot(pipeline.xout, pipeline.zout, 'k')
plt.plot(pipeline.xint, pipeline.zint, 'k')
plt.plot(acoustic_lens.xlens, acoustic_lens.zlens, 'k')
if acoustic_lens.impedance_matching is not None:
    plt.plot(acoustic_lens.x_imp, acoustic_lens.z_imp, 'k')
plt.axis("equal")

plt.grid()
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.xlabel("x-axis / (mm)")
plt.ylabel("y-axis / (mm)")

if acoustic_lens.impedance_matching is not None:
    for iter, n in enumerate(range(transducer.num_elem)):
        # print(f'x: {ximp[n]} == {ximp_2[n]} || z: {zimp[n]} == {zimp_2[n]}')
        if iter == 0:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
            plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5, label="Lens -> Imp. (1)")
            plt.plot([ximp[n], xlens_2[n]], [zimp[n], zlens_2[n]], "C2", linewidth=.5, label="Imp. -> Lens")
            plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5, label="Lens -> Imp. (2)")
            plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5, label="Imp. -> Pipe")
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5, label="Pipe -> Focus")
        else:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5 if iter != 58 else 1.)
            plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5 if iter != 58 else 1.)
            plt.plot([ximp[n], xlens_2[n]], [zimp[n], zlens_2[n]], "C2", linewidth=.5 if iter != 58 else 1.)
            plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5 if iter != 58 else 1.)
            plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5 if iter != 58 else 1.)
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5 if iter != 58 else 1.)
        # plt.plot(
        #     [transducer.xt[n], xlens[n], ximp[n], xlens_2[n], ximp_2[n], xpipe[n], xf],
        #     [transducer.zt[n], zlens[n], zimp[n], zlens_2[n], zimp_2[n], zpipe[n], zf],
        #     linewidth=.5, color='lime', zorder=1
        # )
else:
    for iter, n in enumerate(range(transducer.num_elem)):
        if iter == 0:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
            plt.plot([xlens[n], xpipe[n]], [zlens[n], zpipe[n]], "C1", linewidth=.5, label="Lens -> Pipe")
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C2", linewidth=.5, label="Pipe -> Focus")
        else:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5)
            plt.plot([xlens[n], xpipe[n]], [zlens[n], zpipe[n]], "C1", linewidth=.5)
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C2", linewidth=.5)


plt.plot(xf, zf, 'xr', label='Focus')
plt.legend()
plt.tight_layout()
plt.ylim(-5e-3, acoustic_lens.d + 5e-3)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(transducer.num_elem), tofs[:, 0], '-o', markersize=3)
plt.title("Time of Flight (TOF) para o Foco")
plt.xlabel("Elemento do Transdutor")
plt.ylabel("Time of Flight")
plt.grid(True)
fmc_data = amps['transmission_loss'][:, 0, 0]
plt.subplot(1, 2, 2)
plt.plot(np.arange(transducer.num_elem), fmc_data, '-o', markersize=3)
plt.title("Transmission Loss")
plt.xlabel("Elemento do Transdutor")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
