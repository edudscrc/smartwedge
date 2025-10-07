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

has_impedance_matching = True

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=has_impedance_matching)

outer_radius = np.float32(139.82e-3 / 2)
wall_width = np.float32(16.23e-3)
pipeline = Pipeline(outer_radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)

num_elements = 12

transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=num_elements, fc=5e6)
transducer.zt += acoustic_lens.d

raytracer = RayTracing(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

focus_horizontal_offset = 4e-3

arg = (
    focus_horizontal_offset,
    pipeline.inner_radius + 10e-3,
)
arg = rotate_point(arg, theta_rad=0)
arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

# rt_solver = 'scipy-bounded'
# rt_solver = 'newton'
rt_solver = 'grid-search'

tofs, amps, sol = raytracer.solve(*arg, solver=rt_solver, maxiter=100)

extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

xlens, zlens = extract_pts(sol, 'xlens'), extract_pts(sol, 'zlens')
xpipe, zpipe = extract_pts(sol, 'xpipe'), extract_pts(sol, 'zpipe')

if has_impedance_matching:
    ximp, zimp = extract_pts(sol, 'ximp'), extract_pts(sol, 'zimp')
    ximp_2, zimp_2 = extract_pts(sol, 'ximp_2'), extract_pts(sol, 'zimp_2')
    xlens_2, zlens_2 = extract_pts(sol, 'xlens_2'), extract_pts(sol, 'zlens_2')
    xpipe_no_refl, ypipe_no_refl = extract_pts(sol, 'xpipe_no_refl'), extract_pts(sol, 'ypipe_no_refl')
    new_xpipe, new_ypipe = extract_pts(sol, 'new_pipe_x'), extract_pts(sol, 'new_pipe_y')
    new_inter_3_x, new_inter_3_y = extract_pts(sol, 'new_inter_3_x'), extract_pts(sol, 'new_inter_3_y')

xf, zf = arg

plt.figure()
plt.title("Simulation")
plt.plot(transducer.xt, transducer.zt, 'sk')
plt.plot(0, 0, 'or')
plt.plot(0, acoustic_lens.d, 'or'   )
plt.plot(pipeline.xout, pipeline.zout, 'k')
plt.plot(pipeline.xint, pipeline.zint, 'k')
plt.plot(acoustic_lens.xlens, acoustic_lens.zlens, 'k')
if has_impedance_matching:
    plt.plot(acoustic_lens.x_imp, acoustic_lens.z_imp, 'k')
plt.axis("equal")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{1e3 * x:.1f}"))
plt.xlabel("x-axis / (mm)")
plt.ylabel("y-axis / (mm)")
for iter, n in enumerate(range(transducer.num_elem)):
    if iter == 0:
        if has_impedance_matching:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
            plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5, label="Lens -> Imp.")
            plt.plot([ximp[n], xlens_2[n]], [zimp[n], zlens_2[n]], "C2", linewidth=.5, label="Imp. -> Lens")
            plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5, label="Lens -> Imp. (2)")
            plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5, label="Imp. -> Pipe")
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5, label="Pipe -> Focus")
            plt.plot([xf, new_xpipe[n]], [zf, new_ypipe[n]], "C6", linewidth=.5, label="Focus -> Pipe")
            plt.plot([new_xpipe[n], new_inter_3_x[n]], [new_ypipe[n], new_inter_3_y[n]], "C7", linewidth=.5, label="Pipe -> Water (Imp.)")
        else:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
            plt.plot([xlens[n], xpipe[n]], [zlens[n], zpipe[n]], "C1", linewidth=.5, label="Lens -> Pipe.")
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5, label="Pipe -> Focus")
    else:
        if has_impedance_matching:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5)
            plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5)
            plt.plot([ximp[n], xlens_2[n]], [zimp[n], zlens_2[n]], "C2", linewidth=.5)
            plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5)
            plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5)
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5)
            plt.plot([xf, new_xpipe[n]], [zf, new_ypipe[n]], "C6", linewidth=.5)
            plt.plot([new_xpipe[n], new_inter_3_x[n]], [new_ypipe[n], new_inter_3_y[n]], "C7", linewidth=.5)
        else:
            plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5)
            plt.plot([xlens[n], xpipe[n]], [zlens[n], zpipe[n]], "C1", linewidth=.5)
            plt.plot([xpipe[n], xf], [zpipe[n], zf], "C5", linewidth=.5)
plt.plot(xf, zf, 'xr', label='Focus')
plt.legend()
plt.ylim(-5e-3, acoustic_lens.d + 5e-3)
plt.show()
