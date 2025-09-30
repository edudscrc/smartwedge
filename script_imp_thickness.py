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


thickness_arr =[1/16, 1/14, 1/12, 1/10, 1/8, 1/6, 1/4, 1/2, 1, 1 + 1/16, 1 + 1/14, 1 + 1/12, 1 + 1/10, 1 + 1/8, 1 + 1/6, 1 + 1/4, 1 + 1/2, 2]
results = []

for impedance_thickness in thickness_arr:
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

    acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=True, impedance_matching_thickness=impedance_thickness)

    outer_radius = np.float32(139.82e-3 / 2)
    wall_width = np.float32(16.23e-3)
    pipeline = Pipeline(outer_radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)

    num_elements = 1

    transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=num_elements, fc=5e6)
    transducer.zt += acoustic_lens.d

    raytracer = RayTracing(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

    focus_horizontal_offset = 4e-3

    arg = (focus_horizontal_offset, pipeline.inner_radius + 10e-3)
    arg = rotate_point(arg, theta_rad=0)
    arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

    # rt_solver = 'scipy-bounded'
    # rt_solver = 'newton'
    rt_solver = 'grid-search'

    tofs, amps, sol = raytracer.solve(*arg, solver=rt_solver, maxiter=100)

    extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

    xlens, zlens = extract_pts(sol, 'xlens'), extract_pts(sol, 'zlens')
    xpipe, zpipe = extract_pts(sol, 'xpipe'), extract_pts(sol, 'zpipe')
    ximp, zimp = extract_pts(sol, 'ximp'), extract_pts(sol, 'zimp')
    ximp_2, zimp_2 = extract_pts(sol, 'ximp_2'), extract_pts(sol, 'zimp_2')
    xlens_2, zlens_2 = extract_pts(sol, 'xlens_2'), extract_pts(sol, 'zlens_2')
    xpipe_no_refl, ypipe_no_refl = extract_pts(sol, 'xpipe_no_refl'), extract_pts(sol, 'ypipe_no_refl')

    xf, zf = arg

    fmc_data = amps['transmission_loss'][:, 0, 0]
    fmc_data_with_refl = amps['transmission_loss_with_refl'][:, 0, 0]
    fmc_data_without_refl = amps['transmission_loss_without_refl'][:, 0, 0]

    results.append(fmc_data)

    all_data = np.concatenate([
        fmc_data,
        fmc_data_without_refl,
        fmc_data_with_refl
    ])
    y_min = np.min(all_data)
    y_max = np.max(all_data)
    y_range = y_max - y_min
    y_lim_min = y_min - 0.05 * y_range
    y_lim_max = y_max + 0.05 * y_range

thickness_arr = np.array(thickness_arr)
results = np.array(results)
plt.figure()
plt.plot(thickness_arr, results, '-o', markersize=3)
plt.title('Final Amplitude x Impedance Matching Thickness')
plt.xlabel('Thickness')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
