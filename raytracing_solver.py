import numpy as np
from abc import ABC, abstractmethod
from bisect import bisect
from scipy.optimize import minimize_scalar
from acoustic_lens import AcousticLens
from transducer import Transducer
from pipeline import Pipeline
from ultrasound import *

FLOAT = np.float32


class RayTracingSolver(ABC):
    def __init__(self, acoustic_lens: AcousticLens, pipeline: Pipeline, transducer: Transducer, transmission_loss: bool= False, directivity: bool= False):
        self.transducer = transducer
        self.pipeline = pipeline
        self.acoustic_lens = acoustic_lens

        self.transmission_loss = transmission_loss
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
        solution = self._solve(xf, zf, mode, alpha_step, dist_tol, delta_alpha)

        if mode == 'NN':
            tofs, amplitudes = self.get_tofs_NN(solution)
        elif mode == 'RN':
            tofs, amplitudes = self.get_tofs_NN(solution)

        return tofs, amplitudes, solution

    def _grid_search_batch(self, xf: np.ndarray, zf: np.ndarray, mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3) -> list:
        '''Calls the function newton() one time for each transducer element.
      The set of angles found for a given element are used as initial guess for
      the next one. Starts from the center of the transducer.'''

        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem

        for i in range(N_elem):
            results[i] = self._grid_search(xc[i], yc[i], xf, zf, mode, alpha_step, dist_tol, delta_alpha)
            print(f'{i = }')
        return results

    def _grid_search(self, xc: float, yc: float, xf: np.ndarray, zf: np.ndarray, mode, alpha_step: float, tol: float, delta_alpha: float) -> dict:
        alpha_grid_coarse = np.arange(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max + alpha_step, alpha_step)
        alpha_grid_fine = np.arange(-self.acoustic_lens.alpha_max, self.acoustic_lens.alpha_max + alpha_step/10, alpha_step/10)
        alphaa = np.zeros_like(xf)

        for i, (x_target, y_target) in enumerate(zip(xf, zf)):
            if mode == 'NN':
                # Compute distances for the coarse grid
                dic_coarse_distances = self._dist_kernel_NN(
                    xc, yc,
                    x_target * np.ones_like(alpha_grid_coarse),
                    y_target * np.ones_like(alpha_grid_coarse),
                    alpha_grid_coarse
                )
            elif mode == 'RN':
                # Compute distances for the coarse grid
                dic_coarse_distances = self._dist_kernel_RN(
                    xc, yc,
                    x_target * np.ones_like(alpha_grid_coarse),
                    y_target * np.ones_like(alpha_grid_coarse),
                    alpha_grid_coarse
                )
            # Find alpha minimizing distance on coarse grid
            alpha_coarse_min = alpha_grid_coarse[np.nanargmin(dic_coarse_distances['dist'])]

            # Define fine grid search bounds around coarse minimum
            fine_start_idx = bisect(alpha_grid_fine, alpha_coarse_min - delta_alpha)
            fine_end_idx = bisect(alpha_grid_fine, alpha_coarse_min + delta_alpha)
            alpha_fine_subset = alpha_grid_fine[fine_start_idx:fine_end_idx]

            if mode == 'NN':
                # Compute distances on the fine grid subset
                fine_distances = self._dist_kernel_NN(
                    xc, yc,
                    x_target * np.ones_like(alpha_fine_subset),
                    y_target * np.ones_like(alpha_fine_subset),
                    alpha_fine_subset
                )
            elif mode == 'RN':
                # Compute distances on the fine grid subset
                fine_distances = self._dist_kernel_RN(
                    xc, yc,
                    x_target * np.ones_like(alpha_fine_subset),
                    y_target * np.ones_like(alpha_fine_subset),
                    alpha_fine_subset
                )
            # Find alpha minimizing distance on fine grid subset
            alphaa[i] = alpha_fine_subset[np.nanargmin(fine_distances['dist'])]

        if mode == 'NN':
            # Final evaluation with all optimal alphas
            final_results = self._dist_kernel_NN(xc, yc, xf, zf, alphaa)
        elif mode == 'RN':
            # Final evaluation with all optimal alphas
            final_results = self._dist_kernel_RN(xc, yc, xf, zf, alphaa)
        final_results['firing_angle'] = alphaa
        # Set distances above tolerance to NaN
        final_results['dist'][final_results['dist'] >= tol] = np.nan

        return final_results

    ##### Case-specific:
    @abstractmethod
    def _dist_kernel(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass

    @abstractmethod
    def get_tofs(self, solutions):
        pass
    
    @abstractmethod
    def _dist_kernel_NN(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass

    @abstractmethod
    def get_tofs_NN(self, solutions):
        pass

    @abstractmethod
    def _dist_kernel_RN(self, xc: float, zc: float, xf: np.ndarray, zf: np.ndarray, acurve: np.ndarray):
        pass
