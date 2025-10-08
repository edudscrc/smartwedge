import numpy as np
import matplotlib.pyplot as plt
from acoustic_lens import AcousticLens
from raytracing import RayTracing
from pipeline import Pipeline
from transducer import Transducer
import matplotlib.ticker as ticker
from pathlib import Path


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

c1 = np.float32(CP_ALUMINIUM)       # Speed of sound in aluminum (m/s)
c2 = np.float32(CP_WATER)           # Speed of sound in water (m/s)
c3 = np.float32(CP_STEEL)           # Speed of sound in steel (m/s)

rho_aluminium = np.float32(RHO_ALUMINIUM)
rho_water = np.float32(RHO_WATER)
rho_steel = np.float32(RHO_STEEL)

d = np.float32(170e-3)              # Height of transducer (m)
alpha_max = np.float32(np.pi / 4.0) # Maximum sectorial angle (rad)
alpha_0 = np.float32(0.0)           # Reference angle (boundary condition) (rad)
h0 = np.float32(91.03e-3 + 1e-3)    # Length h chosen at the reference angle (m)

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=True)

outer_radius = np.float32(139.82e-3 / 2)
wall_width = np.float32(16.23e-3)
pipeline = Pipeline(outer_radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)

num_elements = 12

transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=num_elements, fc=5e6)
transducer.zt += acoustic_lens.d

raytracer = RayTracing(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

focus_horizontal_offset = 0

arg = (
    focus_horizontal_offset,
    pipeline.inner_radius + 10e-3,
)
arg = rotate_point(arg, theta_rad=0)
arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

# NN -> Ida normal + Volta normal
# NR -> Ida normal + Volta refletida
# RN -> Ida refletida + Volta normal
# RR -> Ida refletida + Volta refletida
mode = 'NN'

tofs, amps, sol = raytracer.solve(*arg, mode=mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3)

extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

if mode == 'NN':
    x_lens, z_lens = extract_pts(sol, 'x_lens'), extract_pts(sol, 'z_lens')
    x_imp, z_imp = extract_pts(sol, 'x_imp'), extract_pts(sol, 'z_imp')
    x_pipe, z_pipe = extract_pts(sol, 'x_pipe'), extract_pts(sol, 'z_pipe')

xf, zf = arg

plt.figure()
plt.title("Simulation")
plt.plot(transducer.xt, transducer.zt, 'sk')
plt.plot(0, 0, 'or')
plt.plot(0, acoustic_lens.d, 'or'   )
plt.plot(pipeline.xout, pipeline.zout, 'k')
plt.plot(pipeline.xint, pipeline.zint, 'k')
plt.plot(acoustic_lens.xlens, acoustic_lens.zlens, 'k')
plt.plot(acoustic_lens.x_imp, acoustic_lens.z_imp, 'k')
plt.axis("equal")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.xlabel("x-axis / (mm)")
plt.ylabel("y-axis / (mm)")
for iter, n in enumerate(range(transducer.num_elem)):
    if iter == 0:
        plt.plot([transducer.xt[n], x_lens[n]], [transducer.zt[n], z_lens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
        plt.plot([x_lens[n], x_imp[n]], [z_lens[n], z_imp[n]], "C1", linewidth=.5, label="Lens -> Imp.")
        plt.plot([x_imp[n], x_pipe[n]], [z_imp[n], z_pipe[n]], "C2", linewidth=.5, label="Imp. -> Pipe")
        plt.plot([x_pipe[n], xf], [z_pipe[n], zf], "C3", linewidth=.5, label="Pipe -> Focus")
    else:
        plt.plot([transducer.xt[n], x_lens[n]], [transducer.zt[n], z_lens[n]], "C0", linewidth=.5)
        plt.plot([x_lens[n], x_imp[n]], [z_lens[n], z_imp[n]], "C1", linewidth=.5)
        plt.plot([x_imp[n], x_pipe[n]], [z_imp[n], z_pipe[n]], "C2", linewidth=.5)
        plt.plot([x_pipe[n], xf], [z_pipe[n], zf], "C3", linewidth=.5)
plt.plot(xf, zf, 'xr', label='Focus')
plt.legend()
plt.ylim(-5e-3, acoustic_lens.d + 5e-3)
plt.show()
