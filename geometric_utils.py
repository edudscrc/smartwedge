import numpy as np
import cupy as cp
import numba


@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    return zc - np.sqrt(r**2 - (x - xc)**2)


def rotate(x, y, angle, shift_x=0, shift_y=0):
    newx = x * np.cos(angle) - y * np.sin(angle) + shift_x
    newy = x * np.sin(angle) + y * np.cos(angle) + shift_y
    return newx, newy


def dxdy_tube(x, r_circ):
    return -x / np.sqrt(r_circ ** 2 - x ** 2)


def circle_cartesian(r, xcenter=0.0, zcenter=0.0, angstep=1e-2):
    alpha = cp.arange(-cp.pi, cp.pi + angstep, angstep)
    x, z = pol2cart(r, alpha)
    return x + xcenter, z + zcenter


def pol2cart(rho, phi):
    z = rho * cp.cos(phi)
    x = rho * cp.sin(phi)
    return x, z


def line_equation_polar(alpha, a, b, eps=1e-12):
    denom = cp.cos(alpha) - a * cp.sin(alpha)
    safe_denom = cp.where(cp.abs(denom) < eps, cp.sign(denom) * eps, denom)
    return b / safe_denom


def findIntersectionBetweenAcousticLensAndRay_fast(a_ray, b_ray, acoustic_lens, tol=1e-3):
    """
    Vectorized version of findIntersectionBetweenAcousticLensAndRay.
    """
    alpha_step = np.radians(0.1)
    ang_span = np.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step)
    N_rays = len(a_ray)

    # --- 1. Coarse Grid Search (Vectorized) ---
    
    # Reshape inputs for broadcasting:
    # a_ray, b_ray -> (N_rays, 1)
    # ang_span     -> (1, N_angles)
    a_ray_col = a_ray[:, np.newaxis]
    b_ray_col = b_ray[:, np.newaxis]
    ang_span_row = ang_span[np.newaxis, :]

    # Calculate ray coordinates (N_rays, N_angles)
    r1_grid = line_equation_polar(ang_span_row, a_ray_col, b_ray_col)
    x1_grid, y1_grid = pol2cart(r1_grid, ang_span_row)

    # Calculate lens coordinates (1, N_angles)
    # This is the only change from the "impedance" version: no thickness
    x2_grid, y2_grid = acoustic_lens.xy_from_alpha(ang_span)
    x2_grid_row = x2_grid[np.newaxis, :]
    y2_grid_row = y2_grid[np.newaxis, :]

    # Calculate distance (N_rays, N_angles)
    # Use squared distance to avoid sqrt()
    dist_grid_sq = (x1_grid - x2_grid_row)**2 + (y1_grid - y2_grid_row)**2

    # Find the index of the minimum distance for each ray
    # coarse_indices is shape (N_rays,)
    # coarse_indices = np.nanargmin(dist_grid_sq, axis=1)
    # Replace NaN with +infinity so argmin can safely find the minimum *finite* value
    dist_grid_safe = np.nan_to_num(dist_grid_sq, nan=np.inf)
    coarse_indices = np.argmin(dist_grid_safe, axis=1)
    
    # Get the best coarse angle for each ray (N_rays,)
    alpha_0_coarse = ang_span[coarse_indices]

    # --- 2. Fine Grid Search (Vectorized) ---
    
    # Number of points in the fine grid
    N_fine_points = 200 # Original was (alpha_step * 2) / (alpha_step / 100)

    # Create a 2D grid of fine angles, shape (N_rays, N_fine_points)
    # Each row is a different fine grid, centered on that ray's coarse_alpha
    fine_start = (alpha_0_coarse - alpha_step)[:, np.newaxis]
    fine_end = (alpha_0_coarse + alpha_step)[:, np.newaxis]
    
    # Force the shape to (N_rays, N_fine_points) to remove potential extra dims
    ang_span_finer_2D = np.linspace(fine_start, fine_end, N_fine_points, axis=1).reshape(N_rays, N_fine_points)

    # --- 3. Repeat Calculations on the 2D Fine Grid ---

    # Calculate ray coordinates (N_rays, N_fine_points)
    r1_fine_grid = line_equation_polar(ang_span_finer_2D, a_ray_col, b_ray_col)
    x1_fine_grid, y1_fine_grid = pol2cart(r1_fine_grid, ang_span_finer_2D)

    # Calculate lens coordinates (N_rays, N_fine_points)
    finer_flat = ang_span_finer_2D.ravel()
    
    # This is the second change: no thickness
    x2_fine_flat, y2_fine_flat = acoustic_lens.xy_from_alpha(finer_flat)
    
    x2_fine_grid = x2_fine_flat.reshape(ang_span_finer_2D.shape)
    y2_fine_grid = y2_fine_flat.reshape(ang_span_finer_2D.shape)

    # Calculate final distances (N_rays, N_fine_points)
    dist_fine_sq = (x1_fine_grid - x2_fine_grid)**2 + (y1_fine_grid - y2_fine_grid)**2

    # Find the minimum for each ray
    # fine_indices = np.nanargmin(dist_fine_sq, axis=1)
    dist_fine_safe = np.nan_to_num(dist_fine_sq, nan=np.inf)
    fine_indices = np.argmin(dist_fine_safe, axis=1)
    
    # Get the minimum distances for tolerance check
    min_dist_sq = dist_fine_sq[np.arange(N_rays), fine_indices]
    
    # Get the final best alpha value for each ray
    alpha_root = ang_span_finer_2D[np.arange(N_rays), fine_indices]
    
    # Set to NaN where the minimum distance is not within tolerance
    alpha_root[min_dist_sq >= tol**2] = np.nan

    return alpha_root


def findIntersectionBetweenImpedanceMatchingAndRay_fast(a_ray, b_ray, acoustic_lens, tol=1e-3):
    """
    Vectorized version of findIntersectionBetweenImpedanceMatchingAndRay.
    """

    a_ray = cp.array(a_ray)
    b_ray = cp.array(b_ray)

    alpha_step = cp.radians(0.1)
    ang_span = cp.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step)
    thickness = acoustic_lens.impedance_matching.thickness
    N_rays = len(a_ray)

    # --- 1. Coarse Grid Search (Vectorized) ---
    
    # Reshape inputs for broadcasting:
    # a_ray, b_ray -> (N_rays, 1)
    # ang_span     -> (1, N_angles)
    a_ray_col = a_ray[:, cp.newaxis]
    b_ray_col = b_ray[:, cp.newaxis]
    ang_span_row = ang_span[cp.newaxis, :]

    # Calculate ray coordinates (N_rays, N_angles)
    r1_grid = line_equation_polar(ang_span_row, a_ray_col, b_ray_col)
    x1_grid, y1_grid = pol2cart(r1_grid, ang_span_row)

    # Calculate lens coordinates (1, N_angles)
    x2_grid, y2_grid = acoustic_lens.xy_from_alpha(ang_span, thickness=thickness)
    x2_grid_row = x2_grid[cp.newaxis, :]
    y2_grid_row = y2_grid[cp.newaxis, :]

    # Calculate distance (N_rays, N_angles)
    # Use squared distance to avoid sqrt()
    dist_grid_sq = (x1_grid - x2_grid_row)**2 + (y1_grid - y2_grid_row)**2

    # Find the index of the minimum distance for each ray
    # coarse_indices is shape (N_rays,)
    # coarse_indices = cp.nanargmin(dist_grid_sq, axis=1)
    # Replace NaN with +infinity so argmin can safely find the minimum *finite* value
    dist_grid_safe = cp.nan_to_num(dist_grid_sq, nan=cp.inf)
    coarse_indices = cp.argmin(dist_grid_safe, axis=1)
    
    # Get the best coarse angle for each ray (N_rays,)
    alpha_0_coarse = ang_span[coarse_indices]

    # --- 2. Fine Grid Search (Vectorized) ---
    
    # Number of points in the fine grid
    N_fine_points = 200 # Original was (alpha_step * 2) / (alpha_step / 100)

    # Create a 2D grid of fine angles, shape (N_rays, N_fine_points)
    # Each row is a different fine grid, centered on that ray's coarse_alpha
    fine_start = (alpha_0_coarse - alpha_step)[:, cp.newaxis]
    fine_end = (alpha_0_coarse + alpha_step)[:, cp.newaxis]
    # ang_span_finer_2D = cp.linspace(fine_start, fine_end, N_fine_points, axis=1)
    ang_span_finer_2D = cp.linspace(fine_start, fine_end, N_fine_points, axis=1).reshape(N_rays, N_fine_points)

    # --- 3. Repeat Calculations on the 2D Fine Grid ---

    # Calculate ray coordinates (N_rays, N_fine_points)
    r1_fine_grid = line_equation_polar(ang_span_finer_2D, a_ray_col, b_ray_col)
    x1_fine_grid, y1_fine_grid = pol2cart(r1_fine_grid, ang_span_finer_2D)

    # Calculate lens coordinates (N_rays, N_fine_points)
    # We must flatten the 2D angle grid, get the flat (x,y) results,
    # and then reshape them back to 2D.
    finer_flat = ang_span_finer_2D.ravel()
    x2_fine_flat, y2_fine_flat = acoustic_lens.xy_from_alpha(finer_flat, thickness=thickness)
    x2_fine_grid = x2_fine_flat.reshape(ang_span_finer_2D.shape)
    y2_fine_grid = y2_fine_flat.reshape(ang_span_finer_2D.shape)

    # Calculate final distances (N_rays, N_fine_points)
    dist_fine_sq = (x1_fine_grid - x2_fine_grid)**2 + (y1_fine_grid - y2_fine_grid)**2

    # Find the minimum for each ray
    # fine_indices = cp.nanargmin(dist_fine_sq, axis=1)
    dist_fine_safe = cp.nan_to_num(dist_fine_sq, nan=cp.inf)
    fine_indices = cp.argmin(dist_fine_safe, axis=1)
    
    # Get the minimum distances for tolerance check
    min_dist_sq = dist_fine_sq[cp.arange(N_rays), fine_indices]
    
    # Get the final best alpha value for each ray
    alpha_root = ang_span_finer_2D[cp.arange(N_rays), fine_indices]
    
    # Set to NaN where the minimum distance is not within tolerance
    alpha_root[min_dist_sq >= tol**2] = cp.nan

    return alpha_root.get()
