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
from numpy import pi
from matplotlib import ticker


def rotate_point(xy, theta_rad):
    x, y = xy
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return (x_rot, y_rot)



plt.rcParams.update({
    "text.usetex": False,
    "font.size": 10,
    "font.weight": "normal",
})

thickness_arr =[1/16, 1/14, 1/12, 1/10, 1/8, 1/6, 1/4, 1/2, 1, 1 + 1/16, 1 + 1/14, 1 + 1/12, 1 + 1/10, 1 + 1/8, 1 + 1/6, 1 + 1/4, 1 + 1/2, 2]
results = []

for impedance_thickness in thickness_arr:
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

    acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0, rho_aluminium, rho_water, impedance_matching=True, impedance_matching_thickness=impedance_thickness)

    # Pipeline-related parameters:
    radius = 139.82e-3/2
    wall_width = 16.23e-3
    inner_radius = (radius - wall_width)
    c3 = 5900
    pipeline = Pipeline(radius, wall_width, c3, rho_steel, xcenter=0, zcenter=-5e-3)

    # Ultrasound phased array transducer specs:
    transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=1, fc=5e6)
    transducer.zt += acoustic_lens.d

    # Raytracer engine to find time of flight between emitter and focus:
    raytracer = FocusRayTracer(acoustic_lens, pipeline, transducer, transmission_loss=True, directivity=True)

    arg = (
        3e-3,
        inner_radius + 10e-3,
    )
    arg = rotate_point(arg, theta_rad=0)
    arg = (arg[0] + pipeline.xcenter, arg[1] + pipeline.zcenter)

    # 'grid-search' || 'newton' || 'scipy-bounded'
    rt_solver = 'scipy-bounded'

    tofs, amps = raytracer.solve(*arg, solver=rt_solver)
    sol = raytracer._solve(*arg, solver=rt_solver)

    #%% Extract refraction/reflection points:

    extract_pts = lambda list_dict, key: np.array([dict_i[key] for dict_i in list_dict]).flatten()

    xlens, zlens = extract_pts(sol, 'xlens'), extract_pts(sol, 'zlens')
    ximp, zimp = extract_pts(sol, 'ximp'), extract_pts(sol, 'zimp')
    xlens_2, zlens_2 = extract_pts(sol, 'xlens_2'), extract_pts(sol, 'zlens_2')
    ximp_2, zimp_2 = extract_pts(sol, 'ximp_2'), extract_pts(sol, 'zimp_2')
    xpipe, zpipe = extract_pts(sol, 'xpipe'), extract_pts(sol, 'zpipe')
    
    xpipe_no_refl, ypipe_no_refl = extract_pts(sol, 'xpipe_no_refl'), extract_pts(sol, 'ypipe_no_refl')

    xf, zf = arg

    if impedance_thickness == 1/16:
        plt.figure()
        plt.title("Simulation with Impedance Matching")
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
                plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5, label="Transd. -> Lens")
                plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5, label="Lens -> Imp.")
                plt.plot([ximp[n], xpipe_no_refl[n]], [zimp[n], ypipe_no_refl[n]], "C2", linewidth=.5, label="Imp. -> Pipe")
                # plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5, label="Lens -> Imp. (2)")
                # plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5, label="Imp. -> Pipe")
                plt.plot([xpipe_no_refl[n], xf], [ypipe_no_refl[n], zf], "C5", linewidth=.5, label="Pipe -> Focus")
            else:
                plt.plot([transducer.xt[n], xlens[n]], [transducer.zt[n], zlens[n]], "C0", linewidth=.5)
                plt.plot([xlens[n], ximp[n]], [zlens[n], zimp[n]], "C1", linewidth=.5)
                plt.plot([ximp[n], xpipe_no_refl[n]], [zimp[n], ypipe_no_refl[n]], "C2", linewidth=.5)
                # plt.plot([xlens_2[n], ximp_2[n]], [zlens_2[n], zimp_2[n]], "C3", linewidth=.5)
                # plt.plot([ximp_2[n], xpipe[n]], [zimp_2[n], zpipe[n]], "C4", linewidth=.5)
                plt.plot([xpipe_no_refl[n], xf], [ypipe_no_refl[n], zf], "C5", linewidth=.5)
        plt.plot(xf, zf, 'xr', label='Focus')
        plt.legend()
        plt.ylim(-5e-3, acoustic_lens.d + 5e-3)
        plt.tight_layout()
        plt.show()

    fmc_data = amps['transmission_loss'][:, 0, 0]
    fmc_data_with_refl = amps['transmission_loss_with_refl'][:, 0, 0]
    fmc_data_without_refl = amps['transmission_loss_without_refl'][:, 0, 0]

    # perpendicular_results_path = Path("./perpendicular_position_results")
    # perpendicular_results_path.mkdir(parents=True, exist_ok=True)

    # fmc_data_perpendicular = np.load(f"{perpendicular_results_path}/fmc_data_perpendicular.npy")
    # fmc_data_with_refl_perpendicular = np.load(f"{perpendicular_results_path}/fmc_data_with_refl_perpendicular.npy")
    # fmc_data_without_refl_perpendicular = np.load(f"{perpendicular_results_path}/fmc_data_without_refl_perpendicular.npy")
    # fmc_data_no_matching_perpendicular = np.load(f"{perpendicular_results_path}/fmc_data_no_matching_perpendicular.npy")

    # np.save(f"{perpendicular_results_path}/fmc_data_perpendicular.npy", fmc_data)
    # np.save(f"{perpendicular_results_path}/fmc_data_with_refl_perpendicular.npy", fmc_data_with_refl)
    # np.save(f"{perpendicular_results_path}/fmc_data_without_refl_perpendicular.npy", fmc_data_without_refl)
    # np.save(f"{perpendicular_results_path}/fmc_data_no_matching_perpendicular.npy", fmc_data_no_matching)

    # all_data = np.concatenate([
    #     fmc_data,
    #     fmc_data_without_refl,
    #     fmc_data_with_refl
    # ])
    # y_min = np.min(all_data)
    # y_max = np.max(all_data)
    # y_range = y_max - y_min
    # y_lim_min = y_min - 0.05 * y_range
    # y_lim_max = y_max + 0.05 * y_range

    results.append(fmc_data)

    # plt.figure()

    # plt.subplot(3, 1, 1)
    # plt.plot(np.arange(transducer.num_elem), fmc_data, '-o', markersize=3)
    # plt.title("Final Amplitude - Sum of paths (with Impedance Matching)")
    # plt.xlabel("Elemento do Transdutor")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.ylim(y_lim_min, y_lim_max)

    # plt.subplot(3, 1, 2)
    # plt.plot(np.arange(transducer.num_elem), fmc_data_without_refl, '-o', markersize=3)
    # plt.title("Final Amplitude - Path without reflections")
    # plt.xlabel("Elemento do Transdutor")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.ylim(y_lim_min, y_lim_max)

    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(transducer.num_elem), fmc_data_with_refl, '-o', markersize=3)
    # plt.title("Final Amplitude - Path with reflections")
    # plt.xlabel("Elemento do Transdutor")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.ylim(y_lim_min, y_lim_max)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

thickness_arr = np.array(thickness_arr)
results = np.array(results)
# min_amp = np.amin(results)
# mean_amp = np.mean(results)
# max_amp = np.amax(results)
plt.figure()
plt.plot(thickness_arr, results, '-o', markersize=3)
plt.axvline(1/4)
plt.axvline(1 + 1/4)
# plt.stem(thickness_arr, results)
plt.title('Final Amplitude x Impedance Matching Thickness')
plt.xlabel('Thickness')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
