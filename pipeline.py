import numpy as np
import cupy as cp
from geometric_utils import circle_cartesian


class Pipeline:
    def __init__(self, outer_radius: float, wall_thickness: float, c: float, rho: float, xcenter: float, zcenter: float):
        # Convert all scalar inputs to standard Python floats
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(outer_radius - wall_thickness)
        self.wall_width = float(wall_thickness)
        self.c = float(c)
        self.rho = float(rho)
        self.xcenter = float(xcenter)
        self.zcenter = float(zcenter)

        # circle_cartesian already uses cupy
        self.xint, self.zint = circle_cartesian(self.inner_radius, self.xcenter, self.zcenter)
        self.xout, self.zout = circle_cartesian(self.outer_radius, self.xcenter, self.zcenter)

    def dydx(self, x, mode='full'):
        # x is expected to be a cupy array from the raytracer
        dy = -(x - self.xcenter)
        # Use cp.sqrt since x is a cupy array
        dx = cp.sqrt(self.outer_radius ** 2 - (x - self.xcenter) ** 2)
        if mode == "full":
            return dy/dx
        elif mode == 'partial':
            return dy, dx