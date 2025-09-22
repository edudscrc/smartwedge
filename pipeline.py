import numpy as np
from geometric_utils import circle_cartesian


class Pipeline:
    def __init__(self, outer_radius: float, wall_thickness: float, c: float, rho: float, xcenter: float, zcenter: float):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_width = wall_thickness
        self.c = c
        self.rho = rho
        self.xcenter = xcenter
        self.zcenter = zcenter

        self.xint, self.zint = circle_cartesian(self.inner_radius, xcenter, zcenter)
        self.xout, self.zout = circle_cartesian(self.outer_radius, xcenter, zcenter)

    def dydx(self, x, mode='full'):
        dy = -(x - self.xcenter)
        dx = np.sqrt(self.outer_radius ** 2 - (x - self.xcenter) ** 2)
        if mode == "full":
            return dy/dx
        elif mode == 'partial':
            return dy, dx
