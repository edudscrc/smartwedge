import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
from bisect import bisect
from scipy.optimize import minimize_scalar
from acoustic_lens import AcousticLens
from transducer import Transducer
from pipeline import Pipeline
from ultrasound import *

FLOAT = np.float32


class RayTracingSolver(ABC):
    def __init__(self, acoustic_lens: AcousticLens, pipeline: Pipeline, transducer: Transducer, final_amplitude: bool= False, directivity: bool= False):
        self.transducer = transducer
        self.pipeline = pipeline
        self.acoustic_lens = acoustic_lens

        self.final_amplitude = final_amplitude
        self.directivity = directivity

        self.c1 = self.c2 = self.c3 = None

    def _solve(self, xf, zf, mode, alpha_step, dist_tol, delta_alpha):
        if isinstance(xf, (int, float)) and isinstance(zf, (int, float)):
            xf, zf = np.array([xf]), np.array([zf])

        solution = self._grid_search_batch(xf, zf, mode, alpha_step, dist_tol, delta_alpha)

        return solution

    def get_speeds(self):
        c1 = self.acoustic_lens.c1 if self.c1 is None else self.c1  # Wedge material
        c2 = self.acoustic_lens.c2 if self.c2 is None else self.c2  # Coupling medium
        c3 = self.pipeline.c if self.c3 is None else self.c3  # Pipeline material
        return c1, c2, c3

    def solve(self, xf, zf, mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3):
        # Find focii TOF:

        if mode == 'RN':
            # Para o modo RN (Refletido-Normal), precisamos de *ambos* os caminhos
            # O 'solution_R' usa _dist_kernel_RR (baseado no 'mode')
            # print("Resolvendo caminho 'R' (Ida)...")
            solution_R = self._solve(xf, zf, 'RR', alpha_step, dist_tol, delta_alpha) 
            
            # O 'solution_N' usa _dist_kernel_NN (baseado no 'mode')
            # print("Resolvendo caminho 'N' (Volta)...")
            solution_N = self._solve(xf, zf, 'NN', alpha_step, dist_tol, delta_alpha)
            
            # print("Calculando TOFs e Amplitudes...")
            # Passamos ambas as soluções para a função get_tofs
            tofs, amplitudes = self.get_tofs_RN(solution_R, solution_N)
            
            # Retorna a solução 'R' como a principal para plotagem de raios
            return tofs, amplitudes, solution_R
        elif mode == 'NR':
            # NR = Ida 'N' (para TOF e Amp Ida) + Volta 'R' (para Amp Volta)
            # print("Resolvendo caminho 'N' (Ida)...")
            solution_N = self._solve(xf, zf, 'NN', alpha_step, dist_tol, delta_alpha) # 'NN' resolve o caminho N
            
            # print("Resolvendo caminho 'R' (Volta)...")
            solution_R = self._solve(xf, zf, 'RR', alpha_step, dist_tol, delta_alpha) # 'RR' resolve o caminho R
            
            # print("Calculando TOFs e Amplitudes...")
            tofs, amplitudes = self.get_tofs_NR(solution_N, solution_R)
            return tofs, amplitudes, solution_N # Retorna solution_N para plotagem
        else:
            solution = self._solve(xf, zf, mode, alpha_step, dist_tol, delta_alpha)

            if mode == 'NN':
                if self.acoustic_lens.impedance_matching is not None:
                    tofs, amplitudes = self.get_tofs_NN(solution)
                else:
                    tofs, amplitudes = self.get_tofs_NN_without_imp(solution)
            elif mode == 'RR':
                tofs, amplitudes = self.get_tofs_RR(solution)

            return tofs, amplitudes, solution

    def _grid_search_batch(self, xf: np.ndarray, zf: np.ndarray, mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3) -> list:
        '''Calls the function _grid_search one time for each transducer element.
      The set of angles found for a given element are used as initial guess for
      the next one. Starts from the center of the transducer.'''

        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem

        # This loop over elements (N_elem) is difficult to vectorize
        # without significant refactoring of the kernel functions.
        # The main vectorization gain is from passing all xf, zf points at once.
        for i in range(N_elem):
            print(f"  [Solver] Processing Element {i+1}/{N_elem} (xc = {xc[i]:.4f})...")
            results[i] = self._grid_search(xc[i], yc[i], xf, zf, mode, alpha_step, dist_tol, delta_alpha)
        return results

    def _grid_search(self, xc: float, yc: float, xf: np.ndarray, zf: np.ndarray, mode, alpha_step: float, tol: float, delta_alpha: float) -> dict:
        """
        Hybrid search: Loops over focal points, but uses vectorized GPU
        grid searches internally instead of scipy.optimize.
        """
        
        # --- 0. SETUP ---
        # Number of steps in the fine grid. Controls precision.
        N_FINE_STEPS = 30 
        
        # Coarse grid (1D)
        alpha_grid_coarse = cp.arange(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max + alpha_step, alpha_step)
        
        # Fine grid (1D)
        fine_alphas_1D = cp.linspace(-delta_alpha, delta_alpha, N_FINE_STEPS) 
        
        # Convert all inputs to CuPy arrays
        xf_cp = cp.asarray(xf) # Shape (N_points,)
        zf_cp = cp.asarray(zf) # Shape (N_points,)
        
        alphaa = cp.zeros_like(xf_cp) # Final array for best angles
        num_focal_points = len(xf_cp)

        # This loop iterates over all focal points (e.g., 181 for delay laws)
        for i, (x_target, y_target) in enumerate(zip(xf, zf)): # Loop over CPU arrays
            
            # Print progress sparsely
            if (i+1) % 20 == 0 or i == 0 or i == num_focal_points - 1:
                print(f"    [GridSearch] Element @ {xc:.4f}: Solving focal point {i+1}/{num_focal_points}...")

            # Create cupy 1-element arrays for this specific target
            x_target_cp = cp.array([x_target])
            y_target_cp = cp.array([y_target])

            # --- 1. COARSE GRID SEARCH (Vectorized, 1D) ---
            # Broadcast target points to match the coarse grid shape
            xf_grid_coarse = cp.ones_like(alpha_grid_coarse) * x_target_cp
            zf_grid_coarse = cp.ones_like(alpha_grid_coarse) * y_target_cp

            if mode == 'NN':
                if self.acoustic_lens.impedance_matching is not None:
                    dic_coarse = self._dist_kernel_NN(xc, yc, xf_grid_coarse, zf_grid_coarse, alpha_grid_coarse)
                else:
                    dic_coarse = self._dist_kernel_NN_without_imp(xc, yc, xf_grid_coarse, zf_grid_coarse, alpha_grid_coarse)
            elif mode == 'RR' or mode == 'RN':
                dic_coarse = self._dist_kernel_RR(xc, yc, xf_grid_coarse, zf_grid_coarse, alpha_grid_coarse)
            elif mode == 'NR':
                dic_coarse = self._dist_kernel_NN(xc, yc, xf_grid_coarse, zf_grid_coarse, alpha_grid_coarse)

            # Find alpha minimizing distance on coarse grid
            dist_coarse_cp = dic_coarse['dist']
            alpha_coarse_min = alpha_grid_coarse[cp.nanargmin(dist_coarse_cp)] # This is a 0-dim cupy array (scalar)

            # --- 2. FINE GRID SEARCH (Vectorized, 1D) ---
            # Create a 1D fine grid centered around the coarse minimum
            alpha_grid_fine_1D = alpha_coarse_min + fine_alphas_1D

            # Broadcast target points to match the fine grid shape
            xf_grid_fine = cp.ones_like(alpha_grid_fine_1D) * x_target_cp
            zf_grid_fine = cp.ones_like(alpha_grid_fine_1D) * y_target_cp

            # Call the kernel *once* for the entire fine grid
            if mode == 'NN':
                if self.acoustic_lens.impedance_matching is not None:
                    dic_fine = self._dist_kernel_NN(xc, yc, xf_grid_fine, zf_grid_fine, alpha_grid_fine_1D)
                else:
                    dic_fine = self._dist_kernel_NN_without_imp(xc, yc, xf_grid_fine, zf_grid_fine, alpha_grid_fine_1D)
            elif mode == 'RR' or mode == 'RN':
                dic_fine = self._dist_kernel_RR(xc, yc, xf_grid_fine, zf_grid_fine, alpha_grid_fine_1D)
            elif mode == 'NR':
                dic_fine = self._dist_kernel_NN(xc, yc, xf_grid_fine, zf_grid_fine, alpha_grid_fine_1D)
            
            # Find the best angle from the fine search
            dist_fine_cp = dic_fine['dist']
            best_fine_angle = alpha_grid_fine_1D[cp.nanargmin(dist_fine_cp)]
            
            # Store the best angle for this focal point
            alphaa[i] = best_fine_angle

        # --- 3. FINAL KERNEL CALL ---
        # All optimal alphas (alphaa) have been found.
        # Now, call the kernel function *once* with the full cupy arrays.
        print(f"    [GridSearch] Element @ {xc:.4f}: Fine search done. Running final kernel call...")
        if mode == 'NN':
            if self.acoustic_lens.impedance_matching is not None:
                final_results = self._dist_kernel_NN(xc, yc, xf_cp, zf_cp, alphaa)
            else:
                final_results = self._dist_kernel_NN_without_imp(xc, yc, xf_cp, zf_cp, alphaa)
        elif mode == 'RR' or mode == 'RN':
            final_results = self._dist_kernel_RR(xc, yc, xf_cp, zf_cp, alphaa)
        elif mode == 'NR':
            final_results = self._dist_kernel_NN(xc, yc, xf_cp, zf_cp, alphaa)            

        final_results['firing_angle'] = alphaa
        # Set distances above tolerance to NaN
        final_results['dist'] = cp.where(final_results['dist'] >= tol, cp.nan, final_results['dist'])

        # Convert all cupy arrays in the dictionary back to numpy
        final_results_cpu = {}
        for key, value in final_results.items():
            if isinstance(value, cp.ndarray):
                final_results_cpu[key] = cp.asnumpy(value)
            else:
                final_results_cpu[key] = value # e.g., for 'interface_lens2imp' which is a list
        
        print(f"    [GridSearch] Element @ {xc:.4f}: Done.")
        return final_results_cpu

    @abstractmethod
    def _dist_kernel_NN(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass

    @abstractmethod
    def get_tofs_NN(self, solutions):
        pass

    @abstractmethod
    def _dist_kernel_RR(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass

    @abstractmethod
    def get_tofs_RR(self, solutions):
        pass

    @abstractmethod
    def _dist_kernel_NN_without_imp(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass

    @abstractmethod
    def get_tofs_NN_without_imp(self, solutions):
        pass

    @abstractmethod
    def get_tofs_RN(self, solutions):
        pass

    @abstractmethod
    def get_tofs_NR(self, solutions):
        pass