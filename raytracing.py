import numpy as np
import cupy as cp
from numpy import ndarray
from numpy.linalg import norm
from raytracing_utils import roots_bhaskara, snell, uhp, reflection
from geometric_utils import findIntersectionBetweenImpedanceMatchingAndRay_fast, findIntersectionBetweenAcousticLensAndRay_fast
from ultrasound import far_field_directivity_solid, fluid2solid_t_coeff, fluid2solid_r_coeff, solid2fluid_t_coeff, solid2solid_tr_coeff
from raytracing_solver import RayTracingSolver

FLOAT = np.float32

def solid2fluid_r_coeff(theta_p1, theta_p2, cp1, cs1, cp2, rho1, rho2):
    """
    Calculates the reflection coefficient for a solid-to-fluid interface (P-wave incidence).
    This is effectively the negative of the fluid-to-solid coefficient with swapped media.
    """
    # R_solid->fluid(theta1) = -R_fluid->solid(theta2)
    return -fluid2solid_r_coeff(theta_p2, theta_p1, cp2, cp1, cs1, rho2, rho1)



class RayTracing(RayTracingSolver):
    def _dist_kernel_NN(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        impedance_thickness = self.acoustic_lens.impedance_matching.thickness

        #########
        ## IDA ##
        #########

        # Coords. in Lens (Transducer Interface -> Lens Interface)
        x_lens, z_lens = self.acoustic_lens.xy_from_alpha(acurve)
        # Use cupy for arctan2, pi
        gamma_1 = cp.arctan2((z_lens - zc), (x_lens - xc))
        gamma_1 = gamma_1 + (gamma_1 < 0) * cp.pi

        # Lens Interface -> Impedance Interface
        gamma_2, inc_2, ref_2 = snell(c1, c_impedance, gamma_1, self.acoustic_lens.dydx_from_alpha(acurve))
        # Use cupy for tan
        a_2 = cp.tan(uhp(gamma_2))
        b_2 = z_lens - a_2 * x_lens
        alpha_2 = findIntersectionBetweenImpedanceMatchingAndRay_fast(a_2, b_2, self.acoustic_lens)
        # Coords. in Impedance
        x_imp, z_imp = self.acoustic_lens.xy_from_alpha(alpha_2, thickness=impedance_thickness)

        # Impedance Interface -> Pipe Interface
        gamma_3, inc_3, ref_3 = snell(c_impedance, c2, gamma_2, self.acoustic_lens.dydx_from_alpha(alpha_2, thickness=impedance_thickness))
        # Use cupy for tan
        a_3 = cp.tan(uhp(gamma_3))
        b_3 = z_imp - a_3 * x_imp
        aux_a_3 = a_3**2 + 1
        aux_b_3 = 2 * a_3 * b_3 - 2 * (self.pipeline.xcenter + a_3 * self.pipeline.zcenter)
        aux_c_3 = b_3 ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_3 - self.pipeline.outer_radius ** 2
        x_pipe_1, x_pipe_2 = roots_bhaskara(aux_a_3, aux_b_3, aux_c_3)
        z_pipe_1, z_pipe_2 = a_3 * x_pipe_1 + b_3, a_3 * x_pipe_2 + b_3
        z_upper = z_pipe_1 > z_pipe_2
        # Coords. in Pipe
        x_pipe = x_pipe_1 * z_upper + x_pipe_2 * (1 - z_upper)
        z_pipe = z_pipe_1 * z_upper + z_pipe_2 * (1 - z_upper)

        # Pipe Interface -> Focus
        gamma_4, inc_4, ref_4 = snell(c2, c3, gamma_3, self.pipeline.dydx(x_pipe))
        
        # --- START FIX: Numerically Stable Ray Calculation ---
        # The previous method using cp.tan(gamma_4) and -1/a_4 fails for
        # vertical rays (gamma_4 = +/- pi/2), which occurs at angle = 0.
        # This new method is stable for all angles.
        sin_g4 = cp.sin(gamma_4)
        cos_g4 = cp.cos(gamma_4)
        sin2_g4 = sin_g4**2
        cos2_g4 = cos_g4**2
        sincos_g4 = sin_g4 * cos_g4

        # Coords. in Focus (intersection of ray and its normal from target)
        x_found_focus = sincos_g4 * (zf - z_pipe) + sin2_g4 * x_pipe + cos2_g4 * xf
        z_found_focus = sin2_g4 * zf + cos2_g4 * z_pipe - sincos_g4 * (x_pipe - xf)
        # --- END FIX ---

        # Distance between computed ray and focus
        dist = (x_found_focus - xf)**2 + (z_found_focus - zf)**2

        return {
            'x_lens': x_lens, 'z_lens': z_lens,
            'x_imp': x_imp, 'z_imp': z_imp,
            'x_pipe': x_pipe, 'z_pipe': z_pipe,
            'xf': xf, 'zf': zf,
            'x_pipe_2': x_pipe_2, 'z_pipe_2': z_pipe_2,

            'dist': dist,

            'interface_lens2imp': [inc_2, ref_2],
            'interface_imp2pipe': [inc_3, ref_3],
            'interface_pipe2focus': [inc_4, ref_4],
        }

    def get_tofs_NN(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = len(solution[0]['x_lens'])

        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        cs_impedance = c_impedance / 2  
        cs1 = c1 / 2
        cs3 = c3 / 2

        # Convert to cupy arrays for GPU calculation
        coords_transducer = cp.asarray(np.array([self.transducer.xt, self.transducer.zt]).T)
        # solution[0] contains numpy arrays, convert them
        coords_focus = cp.asarray(np.array([solution[0]['xf'], solution[0]['zf']]).T)
        coords_lens = cp.zeros((n_elem, 2, n_focii))
        coords_imp = cp.zeros((n_elem, 2, n_focii))
        coords_pipe = cp.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "final_amplitude": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "final_amplitude_volta": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "directivity": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            # Data is numpy, assign to cupy arrays
            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['x_lens'][i], solution[j]['z_lens'][i]
            coords_imp[j, 0, i], coords_imp[j, 1, i] = solution[j]['x_imp'][i], solution[j]['z_imp'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution[j]['x_pipe'][i], solution[j]['z_pipe'][i]

            if self.final_amplitude:

                #########
                ## IDA ##
                #########

                # 1. Transmission: Lens Inteface -> Impedance Interface
                # Aluminum -> Impedance Matching : (Solid -> Solid)
                Tpp_lens2imp, _ = solid2solid_tr_coeff(
                    solution[j]['interface_lens2imp'][0][i], solution[j]['interface_lens2imp'][1][i],
                    c1, c_impedance, cs1, cs_impedance,
                    self.acoustic_lens.rho1, self.acoustic_lens.impedance_matching.rho
                )

                # 2. Transmission: Impedance Inteface -> Pipe Interface
                # Impedance Matching -> Water : (Solid -> Fluid)
                Tpp_imp2pipe = solid2fluid_t_coeff(
                    solution[j]['interface_imp2pipe'][0][i], solution[j]['interface_imp2pipe'][1][i],
                    c_impedance, c2, cs_impedance,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 3. Transmission: Pipe Interface -> Focus
                # Water -> Steel : (Fluid -> Solid)
                Tpp_pipe2focus, _ = fluid2solid_t_coeff(
                    solution[j]['interface_pipe2focus'][0][i], solution[j]['interface_pipe2focus'][1][i],
                    c2, c3, cs3,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )

                ###########
                ## VOLTA ##
                ###########

                # 4. Transmission: Focus -> Pipe Interface
                # Steel -> Water : (Solid -> Fluid)
                Tpp_focus2pipe = solid2fluid_t_coeff(
                    solution[j]['interface_pipe2focus'][1][i], solution[j]['interface_pipe2focus'][0][i],
                    c3, c2, cs3,
                    self.pipeline.rho, self.acoustic_lens.rho2
                )

                # 5. Transmission: Pipe Interface -> Impedance Interface
                # Water -> Impedance Matching : (Fluid -> Solid)
                Tpp_pipe2imp, _ = fluid2solid_t_coeff(
                    solution[j]['interface_imp2pipe'][1][i], solution[j]['interface_imp2pipe'][0][i],
                    c2, c_impedance, cs_impedance,
                    self.acoustic_lens.rho2, self.acoustic_lens.impedance_matching.rho
                )

                # 6. Transmission: Impedance Interface -> Lens Interface
                # Impedance Matching -> Aluminum : (Solid -> Solid)
                Tpp_imp2lens, _ = solid2solid_tr_coeff(
                    solution[j]['interface_lens2imp'][1][i], solution[j]['interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                amplitudes["final_amplitude"][j, :, i] = Tpp_lens2imp * Tpp_imp2pipe * Tpp_pipe2focus
                amplitudes["final_amplitude_volta"][j, :, i] = Tpp_focus2pipe * Tpp_pipe2imp * Tpp_imp2lens

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        # Use cupy for tile and linalg.norm
        coords_transducer_mat = cp.tile(coords_transducer[:, :, cp.newaxis], (1, 1, n_focii))
        coords_focus_mat = cp.tile(coords_focus[:, :, cp.newaxis], (1, 1, n_elem))

        d1 = cp.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2 = cp.linalg.norm(coords_lens - coords_imp, axis=1)
        d3 = cp.linalg.norm(coords_imp - coords_pipe, axis=1)
        d4 = cp.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = d1 / c1 + d2 / c_impedance + d3 / c2 + d4 / c3

        return tofs, amplitudes

    def _dist_kernel_RR(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        impedance_thickness = self.acoustic_lens.impedance_matching.thickness

        #########
        ## IDA ##
        #########

        # Coords. in Lens (Transducer Interface -> Lens Interface)
        x_lens, z_lens = self.acoustic_lens.xy_from_alpha(acurve)
        # Use cupy for arctan2, pi
        gamma_1 = cp.arctan2((z_lens - zc), (x_lens - xc))
        gamma_1 = gamma_1 + (gamma_1 < 0) * cp.pi

        # Lens Interface -> Impedance Interface
        gamma_2, inc_2, ref_2 = snell(c1, c_impedance, gamma_1, self.acoustic_lens.dydx_from_alpha(acurve))
        # Use cupy for tan
        a_2 = cp.tan(uhp(gamma_2))
        b_2 = z_lens - a_2 * x_lens
        alpha_2 = findIntersectionBetweenImpedanceMatchingAndRay_fast(a_2, b_2, self.acoustic_lens)
        # Coords. in Impedance
        x_imp, z_imp = self.acoustic_lens.xy_from_alpha(alpha_2, thickness=impedance_thickness)

        # Impedance Interface -> Lens Interface
        gamma_refl_1, _, inc_refl_1, ref_refl_1 = reflection(gamma_2, self.acoustic_lens.dydx_from_alpha(alpha_2, thickness=impedance_thickness))
        # Use cupy for tan
        a_refl_1 = cp.tan(uhp(gamma_refl_1))
        b_refl_1 = z_imp - a_refl_1 * x_imp
        alpha_refl_1 = findIntersectionBetweenAcousticLensAndRay_fast(a_refl_1, b_refl_1, self.acoustic_lens)
        # Coords. in Lens
        x_lens_refl_1, z_lens_refl_1 = self.acoustic_lens.xy_from_alpha(alpha_refl_1)

        # Lens Interface -> Impedance Interface
        gamma_refl_2, _, inc_refl_2, ref_refl_2 = reflection(gamma_refl_1, self.acoustic_lens.dydx_from_alpha(alpha_refl_1))
        # Use cupy for tan
        a_refl_2 = cp.tan(uhp(gamma_refl_2))
        b_refl_2 = z_lens_refl_1 - a_refl_2 * x_lens_refl_1
        alpha_refl_2 = findIntersectionBetweenImpedanceMatchingAndRay_fast(a_refl_2, b_refl_2, self.acoustic_lens)
        # Coords. in Impedance
        x_imp_refl_2, z_imp_refl_2 = self.acoustic_lens.xy_from_alpha(alpha_refl_2, thickness=impedance_thickness)

        # Impedance Interface -> Pipe Interface
        gamma_3, inc_3, ref_3 = snell(c_impedance, c2, gamma_refl_2, self.acoustic_lens.dydx_from_alpha(alpha_refl_2, thickness=impedance_thickness))
        # Use cupy for tan
        a_3 = cp.tan(uhp(gamma_3))
        b_3 = z_imp_refl_2 - a_3 * x_imp_refl_2
        aux_a_3 = a_3**2 + 1
        aux_b_3 = 2 * a_3 * b_3 - 2 * (self.pipeline.xcenter + a_3 * self.pipeline.zcenter)
        aux_c_3 = b_3 ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_3 - self.pipeline.outer_radius ** 2
        x_pipe_1, x_pipe_2 = roots_bhaskara(aux_a_3, aux_b_3, aux_c_3)
        z_pipe_1, z_pipe_2 = a_3 * x_pipe_1 + b_3, a_3 * x_pipe_2 + b_3
        z_upper = z_pipe_1 > z_pipe_2
        # Coords. in Pipe
        x_pipe = x_pipe_1 * z_upper + x_pipe_2 * (1 - z_upper)
        z_pipe = z_pipe_1 * z_upper + z_pipe_2 * (1 - z_upper)

        # Pipe Interface -> Focus
        gamma_4, inc_4, ref_4 = snell(c2, c3, gamma_3, self.pipeline.dydx(x_pipe))
        
        # --- START FIX: Numerically Stable Ray Calculation ---
        sin_g4 = cp.sin(gamma_4)
        cos_g4 = cp.cos(gamma_4)
        sin2_g4 = sin_g4**2
        cos2_g4 = cos_g4**2
        sincos_g4 = sin_g4 * cos_g4

        # Coords. in Focus (intersection of ray and its normal from target)
        x_found_focus = sincos_g4 * (zf - z_pipe) + sin2_g4 * x_pipe + cos2_g4 * xf
        z_found_focus = sin2_g4 * zf + cos2_g4 * z_pipe - sincos_g4 * (x_pipe - xf)
        # --- END FIX ---

        # Distance between computed ray and focus
        dist = (x_found_focus - xf)**2 + (z_found_focus - zf)**2

        return {
            'x_lens': x_lens, 'z_lens': z_lens,
            'x_imp': x_imp, 'z_imp': z_imp,
            'x_lens_refl_1': x_lens_refl_1, 'z_lens_refl_1': z_lens_refl_1,
            'x_imp_refl_2': x_imp_refl_2, 'z_imp_refl_2': z_imp_refl_2,
            'x_pipe': x_pipe, 'z_pipe': z_pipe,
            'xf': xf, 'zf': zf,

            'dist': dist,

            'interface_lens2imp': [inc_2, ref_2],
            'r_interface_imp2lens': [inc_refl_1, ref_refl_1],
            'r_interface_lens2imp': [inc_refl_2, ref_refl_2],
            'interface_imp2pipe': [inc_3, ref_3],
            'interface_pipe2focus': [inc_4, ref_4],
        }
    
    def get_tofs_RR(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = len(solution[0]['x_lens'])

        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        cs_impedance = c_impedance / 2  
        cs1 = c1 / 2
        cs3 = c3 / 2

        # Convert to cupy arrays for GPU calculation
        coords_transducer = cp.asarray(np.array([self.transducer.xt, self.transducer.zt]).T)
        coords_focus = cp.asarray(np.array([solution[0]['xf'], solution[0]['zf']]).T)
        coords_lens = cp.zeros((n_elem, 2, n_focii))
        coords_imp = cp.zeros((n_elem, 2, n_focii))
        coords_lens_2 = cp.zeros((n_elem, 2, n_focii))
        coords_imp_2 = cp.zeros((n_elem, 2, n_focii))
        coords_pipe = cp.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "final_amplitude": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "final_amplitude_volta": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "directivity": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            # Data is numpy, assign to cupy arrays
            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['x_lens'][i], solution[j]['z_lens'][i]
            coords_imp[j, 0, i], coords_imp[j, 1, i] = solution[j]['x_imp'][i], solution[j]['z_imp'][i]
            coords_lens_2[j, 0, i], coords_lens_2[j, 1, i] = solution[j]['x_lens_refl_1'][i], solution[j]['z_lens_refl_1'][i]
            coords_imp_2[j, 0, i], coords_imp_2[j, 1, i] = solution[j]['x_imp_refl_2'][i], solution[j]['z_imp_refl_2'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution[j]['x_pipe'][i], solution[j]['z_pipe'][i]

            if self.final_amplitude:

                #########
                ## IDA ##
                #########

                # 1. Transmission: Lens Inteface -> Impedance Interface
                # Aluminum -> Impedance Matching : (Solid -> Solid)
                Tpp_lens2imp, _ = solid2solid_tr_coeff(
                    solution[j]['interface_lens2imp'][0][i], solution[j]['interface_lens2imp'][1][i],
                    c1, c_impedance, cs1, cs_impedance,
                    self.acoustic_lens.rho1, self.acoustic_lens.impedance_matching.rho
                )

                # 2. Reflection: Impedance Inteface -> Lens Interface
                Rpp_imp2lens = solid2fluid_r_coeff(
                    solution[j]['r_interface_imp2lens'][0][i], solution[j]['r_interface_imp2lens'][1][i],
                    c_impedance, cs_impedance, c2,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 3. Reflection: Lens Inteface -> Impedance Interface
                _, Rpp_lens2imp = solid2solid_tr_coeff(
                    solution[j]['r_interface_lens2imp'][0][i], solution[j]['r_interface_lens2imp'][1][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                # 4. Transmission: Impedance Interface -> Pipe Interface
                Tpp_imp2pipe = solid2fluid_t_coeff(
                    solution[j]['interface_imp2pipe'][0][i], solution[j]['interface_imp2pipe'][1][i],
                    c_impedance, c2, cs_impedance,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 5. Transmission: Pipe Interface -> Focus
                Tpp_pipe2focus, _ = fluid2solid_t_coeff(
                    solution[j]['interface_pipe2focus'][0][i], solution[j]['interface_pipe2focus'][1][i],
                    c2, c3, cs3,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )

                ###########
                ## VOLTA ##
                ###########

                # 6. Transmission: Focus -> Pipe Interface
                # Steel -> Water : (Solid -> Fluid)
                Tpp_focus2pipe_return = solid2fluid_t_coeff(
                    solution[j]['interface_pipe2focus'][1][i], solution[j]['interface_pipe2focus'][0][i],
                    c3, c2, cs3,
                    self.pipeline.rho, self.acoustic_lens.rho2
                )

                # 7. Transmission: Pipe Interface -> Impedance Interface
                # Water -> Impedance Matching : (Fluid -> Solid)
                Tpp_pipe2imp_return, _ = fluid2solid_t_coeff(
                    solution[j]['interface_imp2pipe'][1][i], solution[j]['interface_imp2pipe'][0][i],
                    c2, c_impedance, cs_impedance,
                    self.acoustic_lens.rho2, self.acoustic_lens.impedance_matching.rho
                )

                # 8. Reflection
                _, Rpp_lens2imp_return = solid2solid_tr_coeff(
                    solution[j]['r_interface_lens2imp'][1][i], solution[j]['r_interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                # 9. Reflection
                Rpp_imp2lens_return = solid2fluid_r_coeff(
                    solution[j]['r_interface_imp2lens'][1][i], solution[j]['r_interface_imp2lens'][0][i],
                    c_impedance, cs_impedance, c2,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 10. Transmission
                Tpp_lens2imp_return, _ = solid2solid_tr_coeff(
                    solution[j]['interface_lens2imp'][1][i], solution[j]['interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1 
                )

                amplitudes["final_amplitude"][j, :, i] = Tpp_lens2imp * Rpp_imp2lens * Rpp_lens2imp * Tpp_imp2pipe * Tpp_pipe2focus
                amplitudes["final_amplitude_volta"][j, :, i] = Tpp_focus2pipe_return * Tpp_pipe2imp_return * Rpp_lens2imp_return * Rpp_imp2lens_return * Tpp_lens2imp_return

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        # Use cupy for tile and linalg.norm
        coords_transducer_mat = cp.tile(coords_transducer[:, :, cp.newaxis], (1, 1, n_focii))
        coords_focus_mat = cp.tile(coords_focus[:, :, cp.newaxis], (1, 1, n_elem))

        d1_c1 = cp.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2_c_imp = cp.linalg.norm(coords_lens - coords_imp, axis=1)
        d3_c_imp = cp.linalg.norm(coords_imp - coords_lens_2, axis=1)
        d4_c_imp = cp.linalg.norm(coords_lens_2 - coords_imp_2, axis=1)
        d5_c2 = cp.linalg.norm(coords_imp_2 - coords_pipe, axis=1)
        d6_c3 = cp.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = d1_c1 / c1 + d2_c_imp / c_impedance + d3_c_imp / c_impedance + d4_c_imp / c_impedance + d5_c2 / c2 + d6_c3 / c3

        return tofs, amplitudes

    def _dist_kernel_NN_without_imp(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        #########
        ## IDA ##
        #########

        # Coords. in Lens (Transducer Interface -> Lens Interface)
        x_lens, z_lens = self.acoustic_lens.xy_from_alpha(acurve)
        # Use cupy for arctan2, pi
        gamma_1 = cp.arctan2((z_lens - zc), (x_lens - xc))
        gamma_1 = gamma_1 + (gamma_1 < 0) * cp.pi

        # Lens Interface -> Pipe Interface
        gamma_2, inc_2, ref_2 = snell(c1, c2, gamma_1, self.acoustic_lens.dydx_from_alpha(acurve))
        # Use cupy for tan
        a_2 = cp.tan(uhp(gamma_2))
        b_2 = z_lens - a_2 * x_lens
        aux_a_2 = a_2**2 + 1
        aux_b_2 = 2 * a_2 * b_2 - 2 * (self.pipeline.xcenter + a_2 * self.pipeline.zcenter)
        aux_c_2 = b_2 ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_2 - self.pipeline.outer_radius ** 2
        x_pipe_1, x_pipe_2 = roots_bhaskara(aux_a_2, aux_b_2, aux_c_2)
        z_pipe_1, z_pipe_2 = a_2 * x_pipe_1 + b_2, a_2 * x_pipe_2 + b_2
        z_upper = z_pipe_1 > z_pipe_2
        # Coords. in Pipe
        x_pipe = x_pipe_1 * z_upper + x_pipe_2 * (1 - z_upper)
        z_pipe = z_pipe_1 * z_upper + z_pipe_2 * (1 - z_upper)

        # Pipe Interface -> Focus
        gamma_3, inc_3, ref_3 = snell(c2, c3, gamma_2, self.pipeline.dydx(x_pipe))
        
        # --- START FIX: Numerically Stable Ray Calculation ---
        sin_g3 = cp.sin(gamma_3)
        cos_g3 = cp.cos(gamma_3)
        sin2_g3 = sin_g3**2
        cos2_g3 = cos_g3**2
        sincos_g3 = sin_g3 * cos_g3

        # Coords. in Focus (intersection of ray and its normal from target)
        x_found_focus = sincos_g3 * (zf - z_pipe) + sin2_g3 * x_pipe + cos2_g3 * xf
        z_found_focus = sin2_g3 * zf + cos2_g3 * z_pipe - sincos_g3 * (x_pipe - xf)
        # --- END FIX ---

        # Distance between computed ray and focus
        dist = (x_found_focus - xf)**2 + (z_found_focus - zf)**2

        return {
            'x_lens': x_lens, 'z_lens': z_lens,
            'x_pipe': x_pipe, 'z_pipe': z_pipe,
            'xf': xf, 'zf': zf,
            'x_pipe_2': x_pipe_2, 'z_pipe_2': z_pipe_2,
            
            'dist': dist,

            'interface_lens2pipe': [inc_2, ref_2],
            'interface_pipe2focus': [inc_3, ref_3],
        }

    def get_tofs_NN_without_imp(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = len(solution[0]['x_lens'])

        c1, c2, c3 = self.get_speeds()

        cs1 = c1 / 2
        cs3 = c3 / 2

        # Convert to cupy arrays for GPU calculation
        coords_transducer = cp.asarray(np.array([self.transducer.xt, self.transducer.zt]).T)
        coords_focus = cp.asarray(np.array([solution[0]['xf'], solution[0]['zf']]).T)
        coords_lens = cp.zeros((n_elem, 2, n_focii))
        coords_pipe = cp.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "final_amplitude": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "final_amplitude_volta": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "directivity": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            # Data is numpy, assign to cupy arrays
            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['x_lens'][i], solution[j]['z_lens'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution[j]['x_pipe'][i], solution[j]['z_pipe'][i]

            if self.final_amplitude:

                #########
                ## IDA ##
                #########

                # 1. Transmission: Lens Inteface -> Pipe Interface
                # Aluminum -> Water : (Solid -> Fluid)
                Tpp_lens2pipe = solid2fluid_t_coeff(
                    solution[j]['interface_lens2pipe'][0][i], solution[j]['interface_lens2pipe'][1][i],
                    c1, c2, cs1,
                    self.acoustic_lens.rho1, self.acoustic_lens.rho2
                )

                # 2. Transmission: Pipe Inteface -> Focus
                # Water -> Steel : (Fluid -> Solid)
                Tpp_pipe2focus, _ = fluid2solid_t_coeff(
                    solution[j]['interface_pipe2focus'][0][i], solution[j]['interface_pipe2focus'][1][i],
                    c2, c3, cs3,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )

                ###########
                ## VOLTA ##
                ###########

                # 3. Transmission: Focus -> Pipe Interface
                # Steel -> Water : (Solid -> Fluid)
                Tpp_focus2pipe = solid2fluid_t_coeff(
                    solution[j]['interface_pipe2focus'][1][i], solution[j]['interface_pipe2focus'][0][i],
                    c3, c2, cs3,
                    self.pipeline.rho, self.acoustic_lens.rho2
                )

                # 4. Transmission: Pipe Interface -> Lens Interface
                # Water -> Aluminum : (Fluid -> Solid)
                Tpp_pipe2lens, _ = fluid2solid_t_coeff(
                    solution[j]['interface_lens2pipe'][1][i], solution[j]['interface_lens2pipe'][0][i],
                    c2, c1, cs1,
                    self.acoustic_lens.rho2, self.acoustic_lens.rho1
                )

                amplitudes["final_amplitude"][j, :, i] = Tpp_lens2pipe * Tpp_pipe2focus
                amplitudes["final_amplitude_volta"][j, :, i] = Tpp_focus2pipe * Tpp_pipe2lens

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        # Use cupy for tile and linalg.norm
        coords_transducer_mat = cp.tile(coords_transducer[:, :, cp.newaxis], (1, 1, n_focii))
        coords_focus_mat = cp.tile(coords_focus[:, :, cp.newaxis], (1, 1, n_elem))

        d1 = cp.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2 = cp.linalg.norm(coords_lens - coords_pipe, axis=1)
        d3 = cp.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = d1 / c1 + d2 / c2 + d3 / c3

        return tofs, amplitudes

    def get_tofs_RN(self, solution_R, solution_N):
        """
        Calcula os TOFs para o modo RN (Ida Refletida, Volta Normal).
        O TOF (one-way) é calculado para o caminho 'R' (ida).
        A amplitude 'final_amplitude' (ida) é do 'solution_R'.
        A amplitude 'final_amplitude_volta' (volta) é do 'solution_N'.
        """
        n_elem = self.transducer.num_elem
        n_focii = len(solution_R[0]['x_lens']) # Assume R e N têm os mesmos focos

        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        cs_impedance = c_impedance / 2  
        cs1 = c1 / 2
        cs3 = c3 / 2

        # Coordenadas do caminho 'R' (para o TOF)
        # Convert to cupy arrays for GPU calculation
        coords_transducer = cp.asarray(np.array([self.transducer.xt, self.transducer.zt]).T)
        coords_focus = cp.asarray(np.array([solution_R[0]['xf'], solution_R[0]['zf']]).T)
        coords_lens = cp.zeros((n_elem, 2, n_focii))
        coords_imp = cp.zeros((n_elem, 2, n_focii))
        coords_lens_2 = cp.zeros((n_elem, 2, n_focii))
        coords_imp_2 = cp.zeros((n_elem, 2, n_focii))
        coords_pipe = cp.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "final_amplitude": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "final_amplitude_volta": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "directivity": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            # Extrai coordenadas do caminho 'R' (para o TOF)
            # Data is numpy, assign to cupy arrays
            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution_R[j]['x_lens'][i], solution_R[j]['z_lens'][i]
            coords_imp[j, 0, i], coords_imp[j, 1, i] = solution_R[j]['x_imp'][i], solution_R[j]['z_imp'][i]
            coords_lens_2[j, 0, i], coords_lens_2[j, 1, i] = solution_R[j]['x_lens_refl_1'][i], solution_R[j]['z_lens_refl_1'][i]
            coords_imp_2[j, 0, i], coords_imp_2[j, 1, i] = solution_R[j]['x_imp_refl_2'][i], solution_R[j]['z_imp_refl_2'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution_R[j]['x_pipe'][i], solution_R[j]['z_pipe'][i]

            # Corrigido: usa self.final_amplitude como definido no __init__
            if self.final_amplitude:

                #########
                ## IDA (Reflected 'R' Path) - Usa solution_R
                #########

                # 1. Transmission: Lens Inteface -> Impedance Interface
                Tpp_lens2imp, _ = solid2solid_tr_coeff(
                    solution_R[j]['interface_lens2imp'][0][i], solution_R[j]['interface_lens2imp'][1][i],
                    c1, c_impedance, cs1, cs_impedance,
                    self.acoustic_lens.rho1, self.acoustic_lens.impedance_matching.rho
                )

                # 2. Reflection: Impedance Inteface -> Lens Interface
                Rpp_imp2lens = solid2fluid_r_coeff(
                    solution_R[j]['r_interface_imp2lens'][0][i], solution_R[j]['r_interface_imp2lens'][1][i],
                    c_impedance, cs_impedance, c2,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 3. Reflection: Lens Inteface -> Impedance Interface
                _, Rpp_lens2imp = solid2solid_tr_coeff(
                    solution_R[j]['r_interface_lens2imp'][0][i], solution_R[j]['r_interface_lens2imp'][1][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                # 4. Transmission: Impedance Interface -> Pipe Interface
                Tpp_imp2pipe = solid2fluid_t_coeff(
                    solution_R[j]['interface_imp2pipe'][0][i], solution_R[j]['interface_imp2pipe'][1][i],
                    c_impedance, c2, cs_impedance,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 5. Transmission: Pipe Interface -> Focus
                Tpp_pipe2focus, _ = fluid2solid_t_coeff(
                    solution_R[j]['interface_pipe2focus'][0][i], solution_R[j]['interface_pipe2focus'][1][i],
                    c2, c3, cs3,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )

                ###########
                ## VOLTA (Normal 'N' Path) - Usa solution_N
                ###########
                
                # Pega os ângulos da solução 'N' (baseado na reciprocidade)
                # O caminho de volta N (foco -> elem j) usa os mesmos ângulos
                # do caminho de ida N (elem j -> foco) da solution_N

                # 6. Transmission: Focus -> Pipe Interface
                Tpp_focus2pipe = solid2fluid_t_coeff(
                    solution_N[j]['interface_pipe2focus'][1][i], solution_N[j]['interface_pipe2focus'][0][i],
                    c3, c2, cs3,
                    self.pipeline.rho, self.acoustic_lens.rho2
                )

                # 7. Transmission: Pipe Interface -> Impedance Interface
                Tpp_pipe2imp, _ = fluid2solid_t_coeff(
                    solution_N[j]['interface_imp2pipe'][1][i], solution_N[j]['interface_imp2pipe'][0][i],
                    c2, c_impedance, cs_impedance,
                    self.acoustic_lens.rho2, self.acoustic_lens.impedance_matching.rho
                )

                # 8. Transmission: Impedance Interface -> Lens Interface
                Tpp_imp2lens, _ = solid2solid_tr_coeff(
                    solution_N[j]['interface_lens2imp'][1][i], solution_N[j]['interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                # Atribui as amplitudes finais para cada caminho
                amplitudes["final_amplitude"][j, :, i] = Tpp_lens2imp * Rpp_imp2lens * Rpp_lens2imp * Tpp_imp2pipe * Tpp_pipe2focus
                amplitudes["final_amplitude_volta"][j, :, i] = Tpp_focus2pipe * Tpp_pipe2imp * Tpp_imp2lens

            if self.directivity:
                # NOTA: A diretividade de 'ida' usa o ângulo 'R'
                # A diretividade de 'volta' deveria usar o ângulo 'N'
                # A forma como o 'fmc_sim_kernel' está escrito
                # complica isso. Por simplicidade, usamos o ângulo de 
                # disparo do caminho de 'ida' (R) para o 'amplitudes["directivity"]'
                # que é usado para AMBOS os caminhos em 'simulator.py'.
                theta = solution_R[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        # O TOF de ida (one-way) é o do caminho 'R'
        # Use cupy for tile and linalg.norm
        coords_transducer_mat = cp.tile(coords_transducer[:, :, cp.newaxis], (1, 1, n_focii))
        coords_focus_mat = cp.tile(coords_focus[:, :, cp.newaxis], (1, 1, n_elem))

        d1_c1 = cp.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2_c_imp = cp.linalg.norm(coords_lens - coords_imp, axis=1)
        d3_c_imp = cp.linalg.norm(coords_imp - coords_lens_2, axis=1)
        d4_c_imp = cp.linalg.norm(coords_lens_2 - coords_imp_2, axis=1)
        d5_c2 = cp.linalg.norm(coords_imp_2 - coords_pipe, axis=1)
        d6_c3 = cp.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = (d1_c1 / c1 + 
                d2_c_imp / c_impedance + 
                d3_c_imp / c_impedance + 
                d4_c_imp / c_impedance + 
                d5_c2 / c2 + 
                d6_c3 / c3)

        return tofs, amplitudes

    def get_tofs_NR(self, solution_N, solution_R):
        """
        Calcula os TOFs para o modo NR (Ida Normal, Volta Refletida).
        O TOF (one-way) é calculado para o caminho 'N' (ida) de solution_N.
        A amplitude 'final_amplitude' (ida) é do 'solution_N'.
        A amplitude 'final_amplitude_volta' (volta) é do 'solution_R'.
        """
        n_elem = self.transducer.num_elem
        n_focii = len(solution_N[0]['x_lens']) # Usa solução 'N' como base

        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        cs_impedance = c_impedance / 2  
        cs1 = c1 / 2
        cs3 = c3 / 2

        # Coordenadas do caminho 'N' (para o TOF)
        # Convert to cupy arrays for GPU calculation
        coords_transducer = cp.asarray(np.array([self.transducer.xt, self.transducer.zt]).T)
        coords_focus = cp.asarray(np.array([solution_N[0]['xf'], solution_N[0]['zf']]).T)
        coords_lens = cp.zeros((n_elem, 2, n_focii))
        coords_imp = cp.zeros((n_elem, 2, n_focii))
        coords_pipe = cp.zeros((n_elem, 2, n_focii))
        
        amplitudes = {
            "final_amplitude": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "final_amplitude_volta": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64),
            "directivity": cp.ones((n_elem, n_elem, n_focii), dtype=cp.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            # Extrai coordenadas do caminho 'N' (para o TOF)
            # Data is numpy, assign to cupy arrays
            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution_N[j]['x_lens'][i], solution_N[j]['z_lens'][i]
            coords_imp[j, 0, i], coords_imp[j, 1, i] = solution_N[j]['x_imp'][i], solution_N[j]['z_imp'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution_N[j]['x_pipe'][i], solution_N[j]['z_pipe'][i]

            # Corrigido: usa self.final_amplitude como definido no __init__
            if self.final_amplitude:

                #########
                ## IDA (Normal 'N' Path) - Usa solution_N
                ## Lógica de get_tofs_NN
                #########

                # 1. Transmission: Lens Inteface -> Impedance Interface
                Tpp_lens2imp, _ = solid2solid_tr_coeff(
                    solution_N[j]['interface_lens2imp'][0][i], solution_N[j]['interface_lens2imp'][1][i],
                    c1, c_impedance, cs1, cs_impedance,
                    self.acoustic_lens.rho1, self.acoustic_lens.impedance_matching.rho
                )

                # 2. Transmission: Impedance Inteface -> Pipe Interface
                Tpp_imp2pipe = solid2fluid_t_coeff(
                    solution_N[j]['interface_imp2pipe'][0][i], solution_N[j]['interface_imp2pipe'][1][i],
                    c_impedance, c2, cs_impedance,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 3. Transmission: Pipe Interface -> Focus
                Tpp_pipe2focus, _ = fluid2solid_t_coeff(
                    solution_N[j]['interface_pipe2focus'][0][i], solution_N[j]['interface_pipe2focus'][1][i],
                    c2, c3, cs3,
                    self.acoustic_lens.rho2, self.pipeline.rho
                )

                ###########
                ## VOLTA (Reflected 'R' Path) - Usa solution_R
                ## Lógica de get_tofs_RR (volta)
                ###########
                
                # 6. Transmission: Focus -> Pipe Interface
                Tpp_focus2pipe_return = solid2fluid_t_coeff(
                    solution_R[j]['interface_pipe2focus'][1][i], solution_R[j]['interface_pipe2focus'][0][i],
                    c3, c2, cs3,
                    self.pipeline.rho, self.acoustic_lens.rho2
                )

                # 7. Transmission: Pipe Interface -> Impedance Interface
                Tpp_pipe2imp_return, _ = fluid2solid_t_coeff(
                    solution_R[j]['interface_imp2pipe'][1][i], solution_R[j]['interface_imp2pipe'][0][i],
                    c2, c_impedance, cs_impedance,
                    self.acoustic_lens.rho2, self.acoustic_lens.impedance_matching.rho
                )

                # 8. Reflection
                _, Rpp_lens2imp_return = solid2solid_tr_coeff(
                    solution_R[j]['r_interface_lens2imp'][1][i], solution_R[j]['r_interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                )

                # 9. Reflection
                Rpp_imp2lens_return = solid2fluid_r_coeff(
                    solution_R[j]['r_interface_imp2lens'][1][i], solution_R[j]['r_interface_imp2lens'][0][i],
                    c_impedance, cs_impedance, c2,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                )

                # 10. Transmission
                Tpp_lens2imp_return, _ = solid2solid_tr_coeff(
                    solution_R[j]['interface_lens2imp'][1][i], solution_R[j]['interface_lens2imp'][0][i],
                    c_impedance, c1, cs_impedance, cs1,
                    self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1 
                )

                # Atribui as amplitudes finais para cada caminho
                amplitudes["final_amplitude"][j, :, i] = Tpp_lens2imp * Tpp_imp2pipe * Tpp_pipe2focus
                amplitudes["final_amplitude_volta"][j, :, i] = Tpp_focus2pipe_return * Tpp_pipe2imp_return * Rpp_lens2imp_return * Rpp_imp2lens_return * Tpp_lens2imp_return

            if self.directivity:
                # Diretividade usa o ângulo 'N' (caminho de ida)
                theta = solution_N[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        # O TOF de ida (one-way) é o do caminho 'N'
        # Use cupy for tile and linalg.norm
        coords_transducer_mat = cp.tile(coords_transducer[:, :, cp.newaxis], (1, 1, n_focii))
        coords_focus_mat = cp.tile(coords_focus[:, :, cp.newaxis], (1, 1, n_elem))

        d1 = cp.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2 = cp.linalg.norm(coords_lens - coords_imp, axis=1)
        d3 = cp.linalg.norm(coords_imp - coords_pipe, axis=1)
        d4 = cp.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = d1 / c1 + d2 / c_impedance + d3 / c2 + d4 / c3 # TOF do caminho 'N'

        return tofs, amplitudes