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
        n_focii = len(solution[0]['xlens'])

        c1, c2, c3 = self.get_speeds()

        if self.acoustic_lens.impedance_matching is not None:
            c_impedance = self.acoustic_lens.impedance_matching.p_wave_speed
            cs_impedance = c_impedance / 2  
            cs1 = c1 / 2
            cs3 = c3 / 2

        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([solution[0]['xf'], solution[0]['zf']]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        if self.acoustic_lens.impedance_matching is not None:
            coords_imp = np.zeros((n_elem, 2, n_focii))
            coords_lens_2 = np.zeros((n_elem, 2, n_focii))
            coords_imp_2 = np.zeros((n_elem, 2, n_focii))
        coords_outer = np.zeros((n_elem, 2, n_focii))

        amplitudes = {
            "transmission_loss": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "transmission_loss_with_refl": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "transmission_loss_without_refl": np.ones((n_elem, n_elem, n_focii), dtype=np.float64),
            "directivity": np.ones((n_elem, n_elem, n_focii), dtype=np.float64)
        }

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j]['xlens'][i], solution[j]['zlens'][i]
            if self.acoustic_lens.impedance_matching is not None:
                coords_imp[j, 0, i], coords_imp[j, 1, i] = solution[j]['ximp'][i], solution[j]['zimp'][i]
                coords_lens_2[j, 0, i], coords_lens_2[j, 1, i] = solution[j]['xlens_2'][i], solution[j]['zlens_2'][i]
                coords_imp_2[j, 0, i], coords_imp_2[j, 1, i] = solution[j]['ximp_2'][i], solution[j]['zimp_2'][i]
            coords_outer[j, 0, i], coords_outer[j, 1, i] = solution[j]['xpipe'][i], solution[j]['zpipe'][i]

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

        coord_elements_mat = np.tile(coord_elements[:, :, np.newaxis], (1, 1, n_focii))
        coord_reflectors_mat = np.tile(coords_reflectors[:, :, np.newaxis], (1, 1, n_elem))

        d1 = np.linalg.norm(coords_lens - coord_elements_mat, axis=1)             # Transd. -> Lens
        if self.acoustic_lens.impedance_matching is not None:
            d2 = np.linalg.norm(coords_imp - coords_lens, axis=1)                 # Lens -> Imp.
            d3 = np.linalg.norm(coords_lens_2 - coords_imp, axis=1)               # Imp. -> Lens
            d4 = np.linalg.norm(coords_imp_2 - coords_lens_2, axis=1)             # Lens -> Imp.
            d5 = np.linalg.norm(coords_outer - coords_imp_2, axis=1)              # Imp. -> Pipe
            d6 = np.linalg.norm(coords_outer - coord_reflectors_mat.T, axis=1)    # Pipe -> Focii
        else:
            d2 = np.linalg.norm(coords_outer - coords_lens, axis=1)               # Lens -> Pipe
            d3 = np.linalg.norm(coords_outer - coord_reflectors_mat.T, axis=1)    # Pipe -> Focii

        if self.acoustic_lens.impedance_matching is not None:
            tofs = d1 / c1 + (d2 + d3 + d4) / c_impedance + d5 / c2 + d6 / c3
        else:
            tofs = d1 / c1 + d2 / c2 + d3 / c3

        return tofs, amplitudes

    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: float):
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
            a_l = np.tan(uhp(gamma_imp))
            b_l = ylens - a_l * xlens

            alpha_impedance = findIntersectionBetweenImpedanceMatchingAndRay(a_l, b_l, self.acoustic_lens)
            x_impedance_intersection, z_impedance_intersection = self.acoustic_lens.xy_from_alpha(alpha_impedance, thickness=impedance_thickness)

            ###########################
            # SIMULANDO SEM REFLEXÕES #
            ###########################

            # Refraction: Impedance -> Water
            gamma2_no_refl, inc_imp_2_no_refl, ref_imp_2_no_refl = snell(c_impedance, c2, gamma_imp, self.acoustic_lens.dydx_from_alpha(alpha_impedance, thickness=impedance_thickness))
            a_line_no_refl = np.tan(uhp(gamma2_no_refl))
            b_line_no_refl = z_impedance_intersection - a_line_no_refl * x_impedance_intersection

            a_no_refl = a_line_no_refl**2 + 1
            b_no_refl = 2 * a_line_no_refl * b_line_no_refl - 2 * (self.pipeline.xcenter + a_line_no_refl * self.pipeline.zcenter)
            c_no_refl = b_line_no_refl ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_line_no_refl - self.pipeline.outer_radius ** 2

            xcirc1_no_refl, xcirc2_no_refl = roots_bhaskara(a_no_refl, b_no_refl, c_no_refl)
            ycirc1_no_refl, ycirc2_no_refl = a_line_no_refl * xcirc1_no_refl + b_line_no_refl, a_line_no_refl * xcirc2_no_refl + b_line_no_refl
            upper_no_refl = ycirc1_no_refl > ycirc2_no_refl
            xcirc_no_refl = xcirc1_no_refl * upper_no_refl + xcirc2_no_refl * (1 - upper_no_refl)
            ycirc_no_refl = ycirc1_no_refl * upper_no_refl + ycirc2_no_refl * (1 - upper_no_refl)

            # Refraction: Water -> Pipe
            _, inc23_no_refl, ref23_no_refl = snell(c2, c3, gamma2_no_refl, self.pipeline.dydx(xcirc_no_refl))

            ###########################
            ###########################
            ###########################

            # Reflection: Impedance -> Lens
            gamma_imp_refl_1, _, inc_imp_1, ref_imp_1 = reflection(gamma_imp, self.acoustic_lens.dydx_from_alpha(alpha_impedance, thickness=impedance_thickness))
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
        b4 = yf - a4 * xf

        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf)**2 + (yin - yf)**2

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
                'xlens': xlens, 'zlens': ylens,
                'ximp': x_impedance_intersection, 'zimp': z_impedance_intersection,
                'xlens_2': x_lens_intersection, 'zlens_2': z_lens_intersection,
                'ximp_2': x_impedance_intersection_2, 'zimp_2': z_impedance_intersection_2,
                'xpipe': xcirc, 'zpipe': ycirc,
                'xpipe_no_refl': xcirc_no_refl, 'ypipe_no_refl': ycirc_no_refl,
                'dist': dist, 'xf': xf, 'zf': yf,
                'interface_1_imp': [inc_1_imp, ref_1_imp],
                'interface_imp_1': [inc_imp_1, ref_imp_1],
                'interface_1_imp_refl': [inc_1_imp_refl, ref_1_imp_refl],
                'interface_imp_2': [inc_imp_2, ref_imp_2],
                'interface_23': [inc23, ref23],
                'interface_imp_2_no_refl': [inc_imp_2_no_refl, ref_imp_2_no_refl],
                'interface_23_no_refl': [inc23_no_refl, ref23_no_refl],
                'new_pipe_x': new_xcirc, 'new_pipe_y': new_ycirc,
                'new_inter_3_x': x_intersection_3, 'new_inter_3_y': y_intersection_3,
                'new_inter_4_x': x_intersection_4, 'new_inter_4_y': y_intersection_4,
                'last_x': x_in_2, 'last_y': z_in_2,
            }
        else:
            return {
                'xlens': xlens, 'zlens': ylens,
                'xpipe': xcirc, 'zpipe': ycirc,
                'dist': dist, 'xf': xf, 'zf': yf,
                'interface_12': [inc12, ref12],
                'interface_23': [inc23, ref23]
            }
