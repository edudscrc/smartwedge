import numpy as np


def roots_bhaskara(a, b, c):
    sqdelta = np.sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqdelta) / (2 * a)
    x2 = (-b - sqdelta) / (2 * a)
    return x1, x2


def rhp(x):
    x = np.mod(x, np.pi)
    x = x - (x > np.pi / 2) * np.pi
    x = x + (x < -np.pi / 2) * np.pi
    return x


def uhp(x):
    x = rhp(x)
    x = x + (x < 0) * np.pi
    return x


def snell(v1, v2, gamma1, dydx):
    gamma1 = uhp(gamma1)
    slope = rhp(np.arctan(dydx))
    normal = slope + np.pi / 2
    theta1 = gamma1 - normal
    arg = np.sin(theta1) * v2 / v1
    bad_index = np.abs(arg) > 1
    arg[bad_index] = np.tanh(arg[bad_index])
    theta2 = np.arcsin(arg)
    gamma2 = slope - np.pi / 2 + theta2
    return gamma2, theta1, theta2


def refraction(incidence_phi, dzdx, v1, v2):
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray) or isinstance(dzdx, float):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = np.arcsin((v2 / v1) * np.sin(theta_1))
    refractive_phi = phi_slope - (np.pi / 2) + theta_2
    return refractive_phi, phi_normal, theta_1, theta_2


def reflection(incidence_phi, dzdx):
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray) or isinstance(dzdx, float):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = -theta_1
    reflective_phi = phi_slope - (np.pi / 2) + theta_2
    return reflective_phi, phi_normal, theta_1, theta_2
