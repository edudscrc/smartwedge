import numpy as np
import cupy as cp

########################
## MATHEMATICAL UTILS ##
########################
def roots_bhaskara(a, b, c):
    # Use cupy for sqrt
    sqdelta = cp.sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqdelta) / (2 * a)
    x2 = (-b - sqdelta) / (2 * a)
    return x1, x2

###################
## PHYSICS UTILS ##
###################
def rhp(x):
    # Use cupy for mod, pi
    x = cp.mod(x, cp.pi)
    x = x - (x > cp.pi / 2) * cp.pi
    x = x + (x < -cp.pi / 2) * cp.pi
    return x


def uhp(x):
    x = rhp(x)
    x = x + (x < 0) * cp.pi
    return x


def snell(v1, v2, gamma1, dydx):
    gamma1 = uhp(gamma1)
    # Use cupy for arctan, pi, sin, abs, tanh, arcsin
    slope = rhp(cp.arctan(dydx))
    normal = slope + cp.pi / 2
    theta1 = gamma1 - normal
    arg = cp.sin(theta1) * v2 / v1
    bad_index = cp.abs(arg) > 1
    arg[bad_index] = cp.tanh(arg[bad_index])
    theta2 = cp.arcsin(arg)
    gamma2 = slope - cp.pi / 2 + theta2
    return gamma2, theta1, theta2


def refraction(incidence_phi, dzdx, v1, v2):
    if isinstance(dzdx, tuple):
        # Use cupy for arctan2, pi, sin, arcsin
        phi_slope = cp.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, cp.ndarray) or isinstance(dzdx, float) or isinstance(dzdx, np.ndarray):
        phi_slope = cp.arctan(dzdx)
    phi_normal = phi_slope + cp.pi / 2
    theta_1 = incidence_phi - (phi_slope + cp.pi / 2)
    theta_2 = cp.arcsin((v2 / v1) * np.sin(theta_1))
    refractive_phi = phi_slope - (cp.pi / 2) + theta_2
    return refractive_phi, phi_normal, theta_1, theta_2


def reflection(incidence_phi, dzdx):
    if isinstance(dzdx, tuple):
        # Use cupy for arctan2, pi
        phi_slope = cp.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, cp.ndarray) or isinstance(dzdx, float) or isinstance(dzdx, np.ndarray):
        phi_slope = cp.arctan(dzdx)
    phi_normal = phi_slope + cp.pi / 2
    theta_1 = incidence_phi - (phi_slope + cp.pi / 2)
    theta_2 = -theta_1
    reflective_phi = phi_slope - (cp.pi / 2) + theta_2
    return reflective_phi, phi_normal, theta_1, theta_2
