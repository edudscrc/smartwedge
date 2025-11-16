import numpy as np
import cupy as cp
import numba


@numba.njit(fastmath=True)
def f_circ(x, xc, zc, r):
    # This function is from the original file, preserved for context.
    # Note: It uses numpy, so it runs on the CPU.
    return zc - np.sqrt(r**2 - (x - xc)**2)


def rotate(x, y, angle, shift_x=0, shift_y=0):
    # This function is from the original file, preserved for context.
    # Note: It uses numpy, so it runs on the CPU.
    newx = x * np.cos(angle) - y * np.sin(angle) + shift_x
    newy = x * np.sin(angle) + y * np.cos(angle) + shift_y
    return newx, newy


def dxdy_tube(x, r_circ):
    # This function is from the original file, preserved for context.
    # Note: It uses numpy, so it runs on the CPU.
    return -x / np.sqrt(r_circ ** 2 - x ** 2)


def circle_cartesian(r, xcenter=0.0, zcenter=0.0, angstep=1e-2):
    """Calculates circle coordinates using CuPy."""
    alpha = cp.arange(-cp.pi, cp.pi + angstep, angstep)
    x, z = pol2cart(r, alpha)
    return x + xcenter, z + zcenter


def pol2cart(rho, phi):
    """
    Converts polar coordinates (rho, phi) to Cartesian (x, z).
    Note the original function's return variable names (x, z).
    """
    z = rho * cp.cos(phi)
    x = rho * cp.sin(phi)
    return x, z


def line_equation_polar(alpha, a, b, eps=1e-12):
    """Calculates r for a line in polar coordinates using CuPy."""
    denom = cp.cos(alpha) - a * cp.sin(alpha)
    # Use cp.where for safe division, handling potential zeros.
    safe_denom = cp.where(cp.abs(denom) < eps, cp.sign(denom) * eps, denom)
    return b / safe_denom


# --- Custom CuPy Kernel for Coarse Search ---
# This kernel replaces the memory-intensive Part 1 of the original function.
coarse_search_kernel = cp.RawKernel(r'''
extern "C" {
    /*
     * On Windows, NVRTC (the runtime compiler) can fail to find standard
     * C headers like math.h. We remove the include, as CUDA's compiler
     * should provide standard math functions (sin, cos, isnan, etc.)
     * implicitly in the device compilation environment.
     */
    // #include <crt/math.h>  <-- Removed this problematic line

    // --- CUDA device-side helper functions ---
    // These are CUDA C implementations of the Python helper functions.
    // They must be self-contained and use `double` for precision.

    /**
     * @brief Calculates r for a line in polar coordinates.
     */
    __device__ double _line_eq_polar(double alpha, double a, double b, double eps) {
        double denom = cos(alpha) - a * sin(alpha);
        if (fabs(denom) < eps) {
            denom = (denom >= 0.0 ? 1.0 : -1.0) * eps;
        }
        return b / denom;
    }

    /**
     * @brief Converts polar (rho, phi) to Cartesian x-coordinate.
     */
    __device__ double _pol2cart_x(double rho, double phi) {
        return rho * sin(phi);
    }

    /**
     * @brief Converts polar (rho, phi) to Cartesian z-coordinate.
     * (Matches the `z = rho * cp.cos(phi)` in the original Python function)
     */
    __device__ double _pol2cart_z(double rho, double phi) {
        return rho * cos(phi);
    }

    // --- Main CUDA Kernel ---
    /**
     * @brief Performs the coarse grid search for ray intersections.
     *
     * This kernel loops over every ray. For each ray, it iterates through all
     * `N_angles` in `ang_span`, calculates the distance between the ray and the
     * lens at that angle, and keeps track of the angle index (`min_index`)
     * that corresponds to the minimum distance.
     *
     * This avoids creating the massive (N_rays, N_angles) distance matrix.
     *
     * @param a_ray              Input: array of 'a' parameters for each ray.
     * @param b_ray              Input: array of 'b' parameters for each ray.
     * @param ang_span           Input: 1D array of coarse angles to check.
     * @param x2_grid            Input: 1D array of lens x-coordinates (precomputed).
     * @param z2_grid            Input: 1D array of lens z-coordinates (precomputed).
     * @param N_rays             Input: Total number of rays.
     * @param N_angles           Input: Total number of coarse angles.
     * @param eps                Input: Epsilon for safe division in _line_eq_polar.
     * @param out_coarse_indices Output: 1D array to store the best angle index for each ray.
     */
    __global__ void find_coarse_indices(
        const double* a_ray,
        const double* b_ray,
        const double* ang_span,
        const double* x2_grid,
        const double* z2_grid,
        int N_rays,
        int N_angles,
        double eps,
        int* out_coarse_indices)
    {
        // Get the global thread ID, which corresponds to the ray index.
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Ensure we don't run past the number of rays.
        if (i >= N_rays) {
            return;
        }

        double a_i = a_ray[i];
        double b_i = b_ray[i];

        // Replace INFINITY with (1.0 / 0.0) which is the IEEE 754
        // standard for positive infinity and works in CUDA C++.
        double min_dist_sq = (1.0 / 0.0);
        int min_index = -1;

        // This is the inner loop from the original vectorized code.
        // We iterate through all coarse angles for *this* ray (ray 'i').
        for (int j = 0; j < N_angles; j++) {
            double alpha_j = ang_span[j];

            // 1. Calculate ray's (x1, z1) at this angle
            double r1 = _line_eq_polar(alpha_j, a_i, b_i, eps);
            double x1 = _pol2cart_x(r1, alpha_j);
            double z1 = _pol2cart_z(r1, alpha_j);

            // 2. Get precomputed lens's (x2, z2) at this angle
            double x2 = x2_grid[j];
            double z2 = z2_grid[j];

            // 3. Calculate squared distance
            double dx = x1 - x2;
            double dz = z1 - z2;
            double dist_sq = dx*dx + dz*dz;

            // 4. Update minimum distance
            // We must check for isnan/isinf, equivalent to cp.nan_to_num(..., nan=cp.inf)
            if (!isnan(dist_sq) && !isinf(dist_sq) && dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                min_index = j;
            }
        }

        // Store the best index for this ray.
        out_coarse_indices[i] = min_index;
    }
}
''', 'find_coarse_indices')


def findIntersectionBetweenImpedanceMatchingAndRay_fast(a_ray, b_ray, acoustic_lens, tol=1e-5):
    """
    Finds the intersection between rays and the impedance matching layer using
    a two-stage grid search, optimized with a custom CuPy kernel.

    The coarse search (Part 1) uses a custom CUDA kernel to avoid materializing
    a very large (N_rays, N_angles) distance matrix, which is the primary
    bottleneck in the original implementation.
    """
    # Ensure inputs are CuPy arrays, defaulting to float64 for precision
    a_ray = cp.asarray(a_ray, dtype=cp.float64)
    b_ray = cp.asarray(b_ray, dtype=cp.float64)

    alpha_step = cp.radians(0.1)
    ang_span = cp.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step, dtype=cp.float64)
    thickness = acoustic_lens.impedance_matching.thickness
    N_rays = len(a_ray)
    N_angles_coarse = len(ang_span)
    
    # Epsilon for safe division in the kernel
    eps = 1e-12

    # --- 1. Coarse Grid Search (Optimized with Custom Kernel) ---

    # Pre-calculate lens coordinates ONCE.
    # Note: We use z2_grid to match the `pol2cart` function's (x, z) output.
    # The original's `y2_grid` variable actually held 'z' coordinates.
    x2_grid, z2_grid = acoustic_lens.xy_from_alpha(ang_span, thickness=thickness)
    x2_grid = cp.asarray(x2_grid, dtype=cp.float64)
    z2_grid = cp.asarray(z2_grid, dtype=cp.float64)

    # Allocate output memory for the kernel
    coarse_indices_out = cp.empty(N_rays, dtype=cp.int32)

    # Configure and launch the kernel
    threads_per_block = 256
    blocks_per_grid = (N_rays + threads_per_block - 1) // threads_per_block

    coarse_search_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (a_ray, b_ray, ang_span, x2_grid, z2_grid, N_rays, N_angles_coarse, eps, coarse_indices_out)
    )

    # Get the best coarse angle for each ray (N_rays,)
    # Handle cases where min_index was not found (remained -1)
    valid_indices = coarse_indices_out != -1
    alpha_0_coarse = cp.full(N_rays, cp.nan) # Default to NaN
    alpha_0_coarse[valid_indices] = ang_span[coarse_indices_out[valid_indices]]


    # --- 2. Fine Grid Search (Vectorized) ---
    # This part is unchanged. It is already efficient as it scales
    # O(N_rays * N_fine_points), and N_fine_points is a small constant (200).

    # Number of points in the fine grid
    N_fine_points = int((alpha_step * 2) / (alpha_step / 100)) # This is 200

    # Create a 2D grid of fine angles, shape (N_rays, N_fine_points)
    # Each row is a different fine grid, centered on that ray's coarse_alpha
    # REMOVED [:, cp.newaxis] to keep fine_start/fine_end as 1D arrays (N_rays,)
    fine_start = (alpha_0_coarse - alpha_step)
    fine_end = (alpha_0_coarse + alpha_step)
    # cp.linspace(start_1D, stop_1D, num, axis=1) will correctly create a (N_rays, num) 2D array.
    ang_span_finer_2D = cp.linspace(fine_start, fine_end, N_fine_points, axis=1)

    # --- 3. Repeat Calculations on the 2D Fine Grid ---

    # Calculate ray coordinates (N_rays, N_fine_points)
    # We need a (N_rays, 1) shape for broadcasting
    a_ray_col = a_ray[:, cp.newaxis]
    b_ray_col = b_ray[:, cp.newaxis]
    r1_fine_grid = line_equation_polar(ang_span_finer_2D, a_ray_col, b_ray_col, eps)
    x1_fine_grid, z1_fine_grid = pol2cart(r1_fine_grid, ang_span_finer_2D) # Using z1_fine_grid

    # Calculate lens coordinates (N_rays, N_fine_points)
    # We must flatten the 2D angle grid, get the flat (x,z) results,
    # and then reshape them back to 2D.
    finer_flat = ang_span_finer_2D.ravel()
    x2_fine_flat, z2_fine_flat = acoustic_lens.xy_from_alpha(finer_flat, thickness=thickness)
    x2_fine_grid = x2_fine_flat.reshape(ang_span_finer_2D.shape)
    z2_fine_grid = z2_fine_flat.reshape(ang_span_finer_2D.shape)

    # Calculate final distances (N_rays, N_fine_points)
    dist_fine_sq = (x1_fine_grid - x2_fine_grid)**2 + (z1_fine_grid - z2_fine_grid)**2 # Using z1/z2

    # Find the minimum for each ray
    dist_fine_safe = cp.nan_to_num(dist_fine_sq, nan=cp.inf)
    fine_indices = cp.argmin(dist_fine_safe, axis=1)

    # Get the minimum distances for tolerance check
    min_dist_sq = dist_fine_sq[cp.arange(N_rays), fine_indices]

    # Get the final best alpha value for each ray
    alpha_root = ang_span_finer_2D[cp.arange(N_rays), fine_indices]

    # Set to NaN where the minimum distance is not within tolerance
    # or where the coarse search failed (alpha_root will be nan from ang_span_finer_2D)
    alpha_root[min_dist_sq >= tol**2] = cp.nan
    
    # Also ensure NaNs from coarse search propagate
    alpha_root[cp.isnan(alpha_0_coarse)] = cp.nan

    return alpha_root


def findIntersectionBetweenAcousticLensAndRay_fast(a_ray, b_ray, acoustic_lens, tol=1e-5):
    """
    Finds the intersection between rays and the acoustic lens surface using
    a two-stage grid search, optimized with a custom CuPy kernel.

    The coarse search (Part 1) uses a custom CUDA kernel to avoid materializing
    a very large (N_rays, N_angles) distance matrix.
    """
    # Ensure inputs are CuPy arrays, defaulting to float64 for precision
    a_ray = cp.asarray(a_ray, dtype=cp.float64)
    b_ray = cp.asarray(b_ray, dtype=cp.float64)

    alpha_step = cp.radians(0.1)
    ang_span = cp.arange(-acoustic_lens.alpha_max, acoustic_lens.alpha_max + alpha_step, alpha_step, dtype=cp.float64)
    N_rays = len(a_ray)
    N_angles_coarse = len(ang_span)
    
    # Epsilon for safe division in the kernel
    eps = 1e-12

    # --- 1. Coarse Grid Search (Optimized with Custom Kernel) ---

    # Pre-calculate lens coordinates ONCE.
    # This is the only difference from the 'ImpedanceMatching' version:
    # no 'thickness' parameter is passed.
    x2_grid, z2_grid = acoustic_lens.xy_from_alpha(ang_span)
    x2_grid = cp.asarray(x2_grid, dtype=cp.float64)
    z2_grid = cp.asarray(z2_grid, dtype=cp.float64)

    # Allocate output memory for the kernel
    coarse_indices_out = cp.empty(N_rays, dtype=cp.int32)

    # Configure and launch the kernel
    threads_per_block = 256
    blocks_per_grid = (N_rays + threads_per_block - 1) // threads_per_block

    # We reuse the *exact same kernel* as the other function
    coarse_search_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (a_ray, b_ray, ang_span, x2_grid, z2_grid, N_rays, N_angles_coarse, eps, coarse_indices_out)
    )

    # Get the best coarse angle for each ray (N_rays,)
    # Handle cases where min_index was not found (remained -1)
    valid_indices = coarse_indices_out != -1
    alpha_0_coarse = cp.full(N_rays, cp.nan) # Default to NaN
    alpha_0_coarse[valid_indices] = ang_span[coarse_indices_out[valid_indices]]


    # --- 2. Fine Grid Search (Vectorized) ---
    # This part is identical to the other optimized function.

    # Number of points in the fine grid
    N_fine_points = int((alpha_step * 2) / (alpha_step / 100)) # This is 200

    # Create a 2D grid of fine angles, shape (N_rays, N_fine_points)
    # Each row is a different fine grid, centered on that ray's coarse_alpha
    # REMOVED [:, cp.newaxis] to keep fine_start/fine_end as 1D arrays (N_rays,)
    fine_start = (alpha_0_coarse - alpha_step)
    fine_end = (alpha_0_coarse + alpha_step)
    # cp.linspace(start_1D, stop_1D, num, axis=1) will correctly create a (N_rays, num) 2D array.
    ang_span_finer_2D = cp.linspace(fine_start, fine_end, N_fine_points, axis=1)

    # --- 3. Repeat Calculations on the 2D Fine Grid ---

    # Calculate ray coordinates (N_rays, N_fine_points)
    a_ray_col = a_ray[:, cp.newaxis]
    b_ray_col = b_ray[:, cp.newaxis]
    r1_fine_grid = line_equation_polar(ang_span_finer_2D, a_ray_col, b_ray_col, eps)
    x1_fine_grid, z1_fine_grid = pol2cart(r1_fine_grid, ang_span_finer_2D)

    # Calculate lens coordinates (N_rays, N_fine_points)
    finer_flat = ang_span_finer_2D.ravel()
    # Call xy_from_alpha without 'thickness'
    x2_fine_flat, z2_fine_flat = acoustic_lens.xy_from_alpha(finer_flat)
    x2_fine_grid = x2_fine_flat.reshape(ang_span_finer_2D.shape)
    z2_fine_grid = z2_fine_flat.reshape(ang_span_finer_2D.shape)

    # Calculate final distances (N_rays, N_fine_points)
    dist_fine_sq = (x1_fine_grid - x2_fine_grid)**2 + (z1_fine_grid - z2_fine_grid)**2

    # Find the minimum for each ray
    dist_fine_safe = cp.nan_to_num(dist_fine_sq, nan=cp.inf)
    fine_indices = cp.argmin(dist_fine_safe, axis=1)

    # Get the minimum distances for tolerance check
    min_dist_sq = dist_fine_sq[cp.arange(N_rays), fine_indices]

    # Get the final best alpha value for each ray
    alpha_root = ang_span_finer_2D[cp.arange(N_rays), fine_indices]

    # Set to NaN where the minimum distance is not within tolerance
    # or where the coarse search failed (alpha_root will be nan from ang_span_finer_2D)
    alpha_root[min_dist_sq >= tol**2] = cp.nan
    
    # Also ensure NaNs from coarse search propagate
    alpha_root[cp.isnan(alpha_0_coarse)] = cp.nan

    return alpha_root