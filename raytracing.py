import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from raytracing_utils import roots_bhaskara, snell, uhp, reflection
from geometric_utils import findIntersectionBetweenImpedanceMatchingAndRay, findIntersectionBetweenAcousticLensAndRay
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
    def get_tofs(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = len(solution[0]['x_lens'])

        c1, c2, c3 = self.get_speeds()

        if self.acoustic_lens.impedance_matching is not None:
            c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
            cs_impedance = c_impedance / 2  
            cs1 = c1 / 2
            cs3 = c3 / 2

        coords_transducer = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_focus = np.array([solution[0]['xf'], solution[0]['zf']]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        if self.acoustic_lens.impedance_matching is not None:
            coords_imp = np.zeros((n_elem, 2, n_focii))
            coords_lens_2 = np.zeros((n_elem, 2, n_focii))
            coords_imp_2 = np.zeros((n_elem, 2, n_focii))
        coords_pipe = np.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "transmission_loss": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "transmission_loss_with_refl": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "transmission_loss_without_refl": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "directivity": np.ones((n_elem, n_elem, n_focii), dtype=np.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['x_lens'][i], solution[j]['z_lens'][i]
            if self.acoustic_lens.impedance_matching is not None:
                coords_imp[j, 0, i], coords_imp[j, 1, i] = solution[j]['x_imp'][i], solution[j]['z_imp'][i]
                coords_lens_2[j, 0, i], coords_lens_2[j, 1, i] = solution[j]['xlens_2'][i], solution[j]['zlens_2'][i]
                coords_imp_2[j, 0, i], coords_imp_2[j, 1, i] = solution[j]['ximp_2'][i], solution[j]['zimp_2'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution[j]['x_pipe'][i], solution[j]['z_pipe'][i]

            if self.transmission_loss:
                if self.acoustic_lens.impedance_matching is not None:
                    # Transmissão - Alumínio -> Camada de Casamento (Sólido -> Sólido)
                    Tpp_1_imp, _ = solid2solid_tr_coeff(
                        solution[j]['interface_1_imp'][0][i], solution[j]['interface_1_imp'][1][i],
                        c1, c_impedance, cs1, cs_impedance,
                        self.acoustic_lens.rho1, self.acoustic_lens.impedance_matching.rho
                    )

                    ##########################
                    # Considerando reflexões #
                    ##########################

                    # Reflexão - Interface: Camada de Casamento -> Água (Sólido -> Fluido)
                    Tpp_imp_1 = solid2fluid_r_coeff(
                        solution[j]['interface_imp_1'][0][i], solution[j]['interface_imp_1'][1][i],
                        c_impedance, cs_impedance, c2,
                        self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                    )

                    # Reflexão - Interface: Camada de Casamento -> Alumínio (Sólido -> Sólido)
                    _, Tpp_1_imp_refl = solid2solid_tr_coeff(
                        solution[j]['interface_1_imp_refl'][0][i], solution[j]['interface_1_imp_refl'][1][i],
                        c_impedance, c1, cs_impedance, cs1,
                        self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho1
                    )

                    # Transmissão - Camada de Casamento -> Água (Sólido -> Fluido)
                    Tpp_imp_2 = solid2fluid_t_coeff(
                        solution[j]['interface_imp_2'][0][i], solution[j]['interface_imp_2'][1][i],
                        c_impedance, c2, cs_impedance,
                        self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                    )

                    # Transmissão - Água -> Tubo (Fluido -> Sólido)
                    Tpp_23, _ = fluid2solid_t_coeff(
                        solution[j]['interface_23'][0][i], solution[j]['interface_23'][1][i],
                        c2, c3, cs3,
                        self.acoustic_lens.rho2, self.pipeline.rho
                    )

                    ############################
                    # Sem considerar reflexões #
                    ############################

                    # Transmissão - Camada de Casamento -> Água (Sólido -> Fluido)
                    T_imp2water = solid2fluid_t_coeff(
                        solution[j]['interface_imp_2_no_refl'][0][i], solution[j]['interface_imp_2_no_refl'][1][i],
                        c_impedance, c2, cs_impedance,
                        self.acoustic_lens.impedance_matching.rho, self.acoustic_lens.rho2
                    )

                    # Transmissão - Água -> Tubo (Fluido -> Sólido)
                    T_water2pipe, _ = fluid2solid_t_coeff(
                        solution[j]['interface_23_no_refl'][0][i], solution[j]['interface_23_no_refl'][1][i],
                        c2, c3, cs3,
                        self.acoustic_lens.rho2, self.pipeline.rho
                    )

                    transmission_with_refl = Tpp_1_imp * Tpp_imp_1 * Tpp_1_imp_refl * Tpp_imp_2 * Tpp_23
                    amplitudes["transmission_loss_with_refl"][j, :, i] = transmission_with_refl
                    transmission_without_refl = Tpp_1_imp * T_imp2water * T_water2pipe
                    amplitudes["transmission_loss_without_refl"][j, :, i] = transmission_without_refl

                    amplitudes["transmission_loss"][j, :, i] = transmission_with_refl + transmission_without_refl
                else:
                    Tpp_12 = solid2fluid_t_coeff(
                        solution[j]['interface_12'][0][i], solution[j]['interface_12'][1][i],
                        c1, c2, c1 / 2,
                        self.acoustic_lens.rho1, self.acoustic_lens.rho2
                    )
                    Tpp_23, _ = fluid2solid_t_coeff(
                        solution[j]['interface_23'][0][i], solution[j]['interface_23'][1][i],
                        c2, c3, c3 / 2,
                        self.acoustic_lens.rho2, self.pipeline.rho
                    )
                    amplitudes["transmission_loss"][j, :, i] *= Tpp_12 * Tpp_23

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        coords_transducer_mat = np.tile(coords_transducer[:, :, np.newaxis], (1, 1, n_focii))
        coords_focus_mat = np.tile(coords_focus[:, :, np.newaxis], (1, 1, n_elem))

        d1 = np.linalg.norm(coords_lens - coords_transducer_mat, axis=1)             # Transd. -> Lens
        if self.acoustic_lens.impedance_matching is not None:
            d2 = np.linalg.norm(coords_imp - coords_lens, axis=1)                 # Lens -> Imp.
            d3 = np.linalg.norm(coords_lens_2 - coords_imp, axis=1)               # Imp. -> Lens
            d4 = np.linalg.norm(coords_imp_2 - coords_lens_2, axis=1)             # Lens -> Imp.
            d5 = np.linalg.norm(coords_pipe - coords_imp_2, axis=1)              # Imp. -> Pipe
            d6 = np.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)    # Pipe -> Focii
        else:
            d2 = np.linalg.norm(coords_pipe - coords_lens, axis=1)               # Lens -> Pipe
            d3 = np.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)    # Pipe -> Focii

        if self.acoustic_lens.impedance_matching is not None:
            tofs = d1 / c1 + (d2 + d3 + d4) / c_impedance + d5 / c2 + d6 / c3
        else:
            tofs = d1 / c1 + d2 / c2 + d3 / c3

        return tofs, amplitudes

    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        if self.acoustic_lens.impedance_matching is not None:
            c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
            impedance_thickness = self.acoustic_lens.impedance_matching.thickness

        xlens, ylens = self.acoustic_lens.xy_from_alpha(acurve)
        gamma1 = np.arctan2((ylens - zc), (xlens - xc))
        gamma1 = gamma1 + (gamma1 < 0) * np.pi

        if self.acoustic_lens.impedance_matching is not None:
            # Refraction: Lens -> Impedance
            gamma_imp, inc_1_imp, ref_1_imp = snell(c1, c_impedance, gamma1, self.acoustic_lens.dydx_from_alpha(acurve))
            a_2 = np.tan(uhp(gamma_imp))
            b_2 = ylens - a_2 * xlens

            alpha_2 = findIntersectionBetweenImpedanceMatchingAndRay(a_2, b_2, self.acoustic_lens)
            x_impedance_intersection, z_impedance_intersection = self.acoustic_lens.xy_from_alpha(alpha_2, thickness=impedance_thickness)

            ###########################
            # SIMULANDO SEM REFLEXÕES #
            ###########################

            # Refraction: Impedance -> Water
            gamma_3, inc_3, ref_3 = snell(c_impedance, c2, gamma_imp, self.acoustic_lens.dydx_from_alpha(alpha_2, thickness=impedance_thickness))
            a_3 = np.tan(uhp(gamma_3))
            b_3 = z_impedance_intersection - a_3 * x_impedance_intersection

            aux_a_3 = a_3**2 + 1
            aux_b_3 = 2 * a_3 * b_3 - 2 * (self.pipeline.xcenter + a_3 * self.pipeline.zcenter)
            aux_c_3 = b_3 ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_3 - self.pipeline.outer_radius ** 2

            x_pipe_1, x_pipe_2 = roots_bhaskara(aux_a_3, aux_b_3, aux_c_3)
            z_pipe_1, z_pipe_2 = a_3 * x_pipe_1 + b_3, a_3 * x_pipe_2 + b_3
            z_upper = z_pipe_1 > z_pipe_2
            x_pipe = x_pipe_1 * z_upper + x_pipe_2 * (1 - z_upper)
            z_pipe = z_pipe_1 * z_upper + z_pipe_2 * (1 - z_upper)

            # Refraction: Water -> Pipe
            gamma_4, inc_4, ref_4 = snell(c2, c3, gamma_3, self.pipeline.dydx(x_pipe))
            a_4 = np.tan(gamma_4)
            b_4 = z_pipe - a_4 * x_pipe

            # xbottom = -b_4 / a_4
            aux_a_4 = -1 / a_4
            aux_b_4 = zf - aux_a_4 * xf

            x_focus = (aux_b_4 - b_4) / (a_4 - aux_a_4)
            z_focus = a_4 * x_focus + b_4

            ###########################
            ###########################
            ###########################

            # Reflection: Impedance -> Lens
            gamma_imp_refl_1, _, inc_imp_1, ref_imp_1 = reflection(gamma_imp, self.acoustic_lens.dydx_from_alpha(alpha_2, thickness=impedance_thickness))
            a_l_imp_1 = np.tan(uhp(gamma_imp_refl_1))
            b_l_imp_1 = z_impedance_intersection - a_l_imp_1 * x_impedance_intersection

            alpha_lens_refl = findIntersectionBetweenAcousticLensAndRay(a_l_imp_1, b_l_imp_1, self.acoustic_lens)
            x_lens_intersection, z_lens_intersection = self.acoustic_lens.xy_from_alpha(alpha_lens_refl)

            # Reflection: Lens -> Impedance
            gamma_imp_refl_2, _, inc_1_imp_refl, ref_1_imp_refl = reflection(gamma_imp_refl_1, self.acoustic_lens.dydx_from_alpha(alpha_lens_refl))
            a_l_imp_2 = np.tan(uhp(gamma_imp_refl_2))
            b_l_imp_2 = z_lens_intersection - a_l_imp_2 * x_lens_intersection

            alpha_impedance_2 = findIntersectionBetweenImpedanceMatchingAndRay(a_l_imp_2, b_l_imp_2, self.acoustic_lens)
            x_impedance_intersection_2, z_impedance_intersection_2 = self.acoustic_lens.xy_from_alpha(alpha_impedance_2, thickness=impedance_thickness)

            # Refraction: Impedance -> Water
            gamma2, inc_imp_2, ref_imp_2 = snell(c_impedance, c2, gamma_imp_refl_2, self.acoustic_lens.dydx_from_alpha(alpha_impedance_2, thickness=impedance_thickness))
            a_line = np.tan(uhp(gamma2))
            b_line = z_impedance_intersection_2 - a_line * x_impedance_intersection_2
        else:
            # Refraction: Lens -> Water
            gamma2, inc12, ref12 = snell(c1, c2, gamma1, self.acoustic_lens.dydx_from_alpha(acurve))
            a_line = np.tan(uhp(gamma2))
            b_line = ylens - a_line * xlens

        a = a_line**2 + 1
        b = 2 * a_line * b_line - 2 * (self.pipeline.xcenter + a_line * self.pipeline.zcenter)
        c = b_line ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_line - self.pipeline.outer_radius ** 2

        xcirc1, xcirc2 = roots_bhaskara(a, b, c)
        ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
        upper = ycirc1 > ycirc2
        xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
        ycirc = ycirc1 * upper + ycirc2 * (1 - upper)

        # Refraction: Water -> Pipe
        gamma3, inc23, ref23 = snell(c2, c3, gamma2, self.pipeline.dydx(xcirc))
        a3 = np.tan(gamma3)
        b3 = ycirc - a3 * xcirc

        # xbottom = -b3 / a3
        a4 = -1 / a3
        b4 = zf - a4 * xf

        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf)**2 + (yin - zf)**2

        # Reflection: Focus -> Pipe
        dy = xin - self.pipeline.xcenter
        dx = -(yin - self.pipeline.zcenter)
        g, _, inc, ref = reflection(gamma3, dy / dx)
        new_a = np.tan(uhp(g))
        new_b = yin - new_a * xin

        a = new_a**2 + 1
        b = 2 * new_a * new_b - 2 * (self.pipeline.xcenter + new_a * self.pipeline.zcenter)
        c = new_b ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * new_b - self.pipeline.outer_radius ** 2

        new_xcirc1, new_xcirc2 = roots_bhaskara(a, b, c)
        new_ycirc1, new_ycirc2 = new_a * new_xcirc1 + new_b, new_a * new_xcirc2 + new_b
        new_upper = new_ycirc1 > new_ycirc2
        new_xcirc = new_xcirc1 * new_upper + new_xcirc2 * (1 - new_upper)
        new_ycirc = new_ycirc1 * new_upper + new_ycirc2 * (1 - new_upper)

        # Refraction: Pipe -> Water
        alpha2 = findIntersectionBetweenImpedanceMatchingAndRay(new_a, new_b, self.acoustic_lens)

        g2, inc2, ref2 = snell(c3, c2, g, self.acoustic_lens.dydx_from_alpha(alpha2, thickness=impedance_thickness))
        new_a2 = np.tan(uhp(g2))
        new_b2 = new_ycirc - new_a2 * new_xcirc

        alpha3 = findIntersectionBetweenImpedanceMatchingAndRay(new_a2, new_b2, self.acoustic_lens)
        x_intersection_3, y_intersection_3 = self.acoustic_lens.xy_from_alpha(alpha3, thickness=impedance_thickness)

        g3, inc3, ref3 = snell(c2, c_impedance, g2, self.acoustic_lens.dydx_from_alpha(alpha3, thickness=impedance_thickness))
        new_a3 = np.tan(uhp(g3))
        new_b3 = y_intersection_3 - new_a3 * x_intersection_3

        alpha4 = findIntersectionBetweenAcousticLensAndRay(new_a3, new_b3, self.acoustic_lens)
        x_intersection_4, y_intersection_4 = self.acoustic_lens.xy_from_alpha(alpha4)

        alpha_intersection = np.arctan2(x_intersection_4, y_intersection_4)

        g4, inc4, ref4 = snell(c_impedance, c1, g3, self.acoustic_lens.dydx_from_alpha(alpha_intersection))
        new_a4 = np.tan(uhp(g4))
        new_b4 = y_intersection_4 - new_a4 * x_intersection_4

        x_in_2 = (np.repeat(self.transducer.zt[0], len(new_b4)) - new_b4) / new_a4
        z_in_2 = np.repeat(self.transducer.zt[0], len(new_b4))

        if self.acoustic_lens.impedance_matching is not None:
            return {
                'x_lens': xlens, 'z_lens': ylens,
                'x_imp': x_impedance_intersection, 'z_imp': z_impedance_intersection,
                'xlens_2': x_lens_intersection, 'zlens_2': z_lens_intersection,
                'ximp_2': x_impedance_intersection_2, 'zimp_2': z_impedance_intersection_2,
                'x_pipe': xcirc, 'z_pipe': ycirc,
                'xpipe_no_refl': x_pipe, 'ypipe_no_refl': z_pipe,
                'xf_no_refl': x_focus, 'zf_no_refl': z_focus,
                'dist': dist, 'xf': xf, 'zf': zf,
                'interface_1_imp': [inc_1_imp, ref_1_imp],
                'interface_imp_1': [inc_imp_1, ref_imp_1],
                'interface_1_imp_refl': [inc_1_imp_refl, ref_1_imp_refl],
                'interface_imp_2': [inc_imp_2, ref_imp_2],
                'interface_23': [inc23, ref23],
                'interface_imp_2_no_refl': [inc_3, ref_3],
                'interface_23_no_refl': [inc_4, ref_4],
                'new_pipe_x': new_xcirc, 'new_pipe_y': new_ycirc,
                'new_inter_3_x': x_intersection_3, 'new_inter_3_y': y_intersection_3,
                'new_inter_4_x': x_intersection_4, 'new_inter_4_y': y_intersection_4,
                'last_x': x_in_2, 'last_y': z_in_2,
            }
        else:
            return {
                'x_lens': xlens, 'z_lens': ylens,
                'x_pipe': xcirc, 'z_pipe': ycirc,
                'dist': dist, 'xf': xf, 'zf': zf,
                'interface_12': [inc12, ref12],
                'interface_23': [inc23, ref23]
            }

    def _dist_kernel_NN(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: float):
        c1, c2, c3 = self.get_speeds()

        c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
        impedance_thickness = self.acoustic_lens.impedance_matching.thickness

        # Coords. in Lens (Transducer Interface -> Lens Interface)
        x_lens, z_lens = self.acoustic_lens.xy_from_alpha(acurve)
        gamma_1 = np.arctan2((z_lens - zc), (x_lens - xc))
        gamma_1 = gamma_1 + (gamma_1 < 0) * np.pi

        # Lens Interface -> Impedance Interface
        gamma_2, inc_2, ref_2 = snell(c1, c_impedance, gamma_1, self.acoustic_lens.dydx_from_alpha(acurve))
        a_2 = np.tan(uhp(gamma_2))
        b_2 = z_lens - a_2 * x_lens
        alpha_2 = findIntersectionBetweenImpedanceMatchingAndRay(a_2, b_2, self.acoustic_lens)
        # Coords. in Impedance
        x_imp, z_imp = self.acoustic_lens.xy_from_alpha(alpha_2, thickness=impedance_thickness)

        # Impedance Interface -> Pipe Interface
        gamma_3, inc_3, ref_3 = snell(c_impedance, c2, gamma_2, self.acoustic_lens.dydx_from_alpha(alpha_2, thickness=impedance_thickness))
        a_3 = np.tan(uhp(gamma_3))
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
        a_4 = np.tan(gamma_4)
        b_4 = z_pipe - a_4 * x_pipe
        aux_a_4 = -1 / a_4
        aux_b_4 = zf - aux_a_4 * xf
        # Coords. in Focus
        x_found_focus = (aux_b_4 - b_4) / (a_4 - aux_a_4)
        z_found_focus = a_4 * x_found_focus + b_4

        # Distance between computed ray and focus
        dist = (x_found_focus - xf)**2 + (z_found_focus - zf)**2

        return {
            'x_lens': x_lens, 'z_lens': z_lens,
            'x_imp': x_imp, 'z_imp': z_imp,
            'x_pipe': x_pipe, 'z_pipe': z_pipe,
            'xf': xf, 'zf': zf,
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

        coords_transducer = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_focus = np.array([solution[0]['xf'], solution[0]['zf']]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        coords_imp = np.zeros((n_elem, 2, n_focii))
        coords_pipe = np.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "transmission_loss": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "directivity": np.ones((n_elem, n_elem, n_focii), dtype=np.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['x_lens'][i], solution[j]['z_lens'][i]
            coords_imp[j, 0, i], coords_imp[j, 1, i] = solution[j]['x_imp'][i], solution[j]['z_imp'][i]
            coords_pipe[j, 0, i], coords_pipe[j, 1, i] = solution[j]['x_pipe'][i], solution[j]['z_pipe'][i]

            if self.transmission_loss:
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

                amplitudes["transmission_loss"][j, :, i] = Tpp_lens2imp * Tpp_imp2pipe * Tpp_pipe2focus

            if self.directivity:
                theta = solution[j]['firing_angle'][i]
                k = self.transducer.fc * 2 * np.pi / self.acoustic_lens.c1
                amplitudes["directivity"][j, :, i] *= far_field_directivity_solid(
                    theta, c1, c1 / 2, k, self.transducer.element_width
                )

        coords_transducer_mat = np.tile(coords_transducer[:, :, np.newaxis], (1, 1, n_focii))
        coords_focus_mat = np.tile(coords_focus[:, :, np.newaxis], (1, 1, n_elem))

        d1 = np.linalg.norm(coords_lens - coords_transducer_mat, axis=1)
        d2 = np.linalg.norm(coords_imp - coords_lens, axis=1)
        d3 = np.linalg.norm(coords_pipe - coords_imp, axis=1)
        d4 = np.linalg.norm(coords_pipe - coords_focus_mat.T, axis=1)

        tofs = d1 / c1 + d2 / c_impedance + d3 / c2 + d4 / c3

        return tofs, amplitudes
