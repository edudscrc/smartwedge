import numpy as  np


class ImpedanceMatching:
    def __init__(self, p_wave_speed, density, central_frequency=5.0e6, impedance_matching_thickness=1/4):
        self.p_wave_speed = p_wave_speed
        self.central_frequency = central_frequency
        self.rho = density
        self.wave_length = (self.p_wave_speed) / (self.central_frequency)
        self.thickness = impedance_matching_thickness * self.wave_length


class AcousticLens:
    def __init__(self, c1: float, c2: float, d: float, alpha_max: float, alpha_0: float, h0: float, rho1: float, rho2: float, impedance_matching: bool, impedance_matching_thickness: float=1/4):
        """
        :param c1: Speed of sound (acoustic lens) in (m/s)
        :param c2: Speed of sound (coupling medium) in (m/s)
        :param d: Height of transducer in (m)
        :param alpha_max: Maximum sectorial angle in (rad)
        :param alpha_0: Reference angle (boundary condition) in (rad)
        :param h0: Length h chosen at the reference angle in (m)
        """
        self.num_of_points = 1000

        self.c1 = c1
        self.c2 = c2
        self.d = d
        self.alpha_max = alpha_max
        self.alpha_0 = alpha_0
        self.h0 = h0
        self.rho1 = rho1
        self.rho2 = rho2

        self.l0 = self.d - self.h0

        x0, z0 = h0 * np.cos(np.pi/2 - alpha_0), h0 * np.sin(np.pi/2 - alpha_0)
        xt, zt = 0, d

        self.T = np.sqrt((x0 - xt)**2 + (z0 - zt)**2)/self.c1 + self.h0/self.c2

        self.a = (c1/c2)**2 - 1
        self.b = lambda alpha : 2 * d * np.cos(alpha) - 2 * self.T * c1 ** 2 / c2
        self.c = (c1 * self.T) ** 2 - d ** 2

        self.xlens, self.zlens = self.xy_from_alpha(np.linspace(-self.alpha_max, self.alpha_max, self.num_of_points))

        if impedance_matching:
            self.impedance_matching = ImpedanceMatching(p_wave_speed=np.float32(2900), density=np.float32(1700), impedance_matching_thickness=impedance_matching_thickness)
            self.x_imp, self.z_imp = self.xy_from_alpha(np.linspace(-self.alpha_max, self.alpha_max, self.num_of_points), 
                                                        thickness=self.impedance_matching.thickness)
        else:
            self.impedance_matching = None
            self.x_imp, self.z_imp = None, None

    def h(self, alpha):
        return (-self.b(alpha) - np.sqrt(self.b(alpha) ** 2 - 4 * self.a * self.c)) / (2 * self.a)

    def dhda(self, alpha):
        """
        This function computes the acoustic lens derivative in polar coordinates.

        :param alpha: pipe inspection angle in rad.
        :return: derivative value of z(alpha) in polar coordinates.
        """
        return -1 / (2 * self.a) * (
                    -2 * self.d * np.sin(alpha) + 1 / 2 * 1 / np.sqrt(self.b(alpha) ** 2 - 4 * self.a * self.c) * (
                        -4 * self.b(alpha) * self.d * np.sin(alpha)))

    def xy_from_alpha(self, alpha, thickness=0):
        """Computes the (x,y) coordinates of the lens for a given pipe angle"""
        z = self.h(alpha)

        scale_factor = (z - thickness) / z
        z *= scale_factor

        y = z * np.cos(alpha)
        x = z * np.sin(alpha)
        return x, y

    def dydx_from_alpha(self, alpha, mode='full', thickness=0):
        """Computes the slope (dy/dx) of the lens for a given alpha"""

        h_ = self.h(alpha)

        dh_dAlpha = self.dhda(alpha)

        scale_factor = (dh_dAlpha - thickness) / dh_dAlpha
        dh_dAlpha *= scale_factor

        # Equations (A.19a) and (A.19b) in Appendix A.2.2.
        dy = dh_dAlpha * np.cos(alpha) - h_ * np.sin(alpha)
        dx = dh_dAlpha * np.sin(alpha) + h_ * np.cos(alpha)

        if mode == 'full':
            return  dy/dx
        elif mode == 'partial':
            return dy, dx
        else:
            raise ValueError("mode must be 'full' or 'parts'")
