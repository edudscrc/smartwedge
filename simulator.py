import numpy as np
import cupy as cp
from numba import prange

from raytracing_solver import RayTracingSolver
from raytracing import RayTracing
from simulator_utils import fmc_sim_kernel, fmc2sscan

__all__ = ["Simulator"]


class Simulator:
    def __init__(self, sim_params: dict, raytracer_list: list, verbose: bool = True):
        self.raytracer_list = raytracer_list
        self.transducer = self.raytracer_list[0].transducer
        # Reflectors position:
        self.xf, self.zf = None, None
        self.verbose = verbose

        # Gate-related parameters:
        self.fs = sim_params["fs"]  # Sampling frequency in Hz
        self.gate_start = sim_params["gate_start"]  # Gate start in seconds
        self.gate_end = sim_params["gate_end"]  # Gate end in seconds
        self.surface_echoes = sim_params["surface_echoes"]
        self.tspan = np.arange(self.gate_start, self.gate_end + 1 / self.fs, 1 / self.fs)  # Time-grid.

        # Inspection type:
        self.response_type = sim_params["response_type"]  # e.g. fmc, s-scan
        match self.response_type:
            case "s-scan":
                self.delaylaw_t, self.delaylaw_r = sim_params["emission_delaylaw"], sim_params["reception_delaylaw"]

                # Delay law in number of samples to shift during DAS:
                self.shifts_e = np.round(self.delaylaw_t * self.fs)
                self.shifts_r = np.round(self.delaylaw_r * self.fs)

            case "fmc":
                pass

            case _:
                raise NotImplementedError

        # Configuration list of multiple simulations instances:
        self.sim_list = []
        self.fmcs = None
        self.sscans = None

    def add_reflector(self, xf, zf, different_instances: bool=False):
        self.xf, self.zf = xf, zf
        if different_instances:
            for x, z in zip(xf, zf):
                sim = {
                    "reflectors": [x, z],
                    "response": None,
                }
                self.sim_list.append(sim)
        else:
            sim = {
                "reflectors": [xf, zf],
                "response": None,
            }
            self.sim_list.append(sim)

    def __simulate(self, mode, alpha_step, dist_tol, delta_alpha):
        Nel = self.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)
        self.fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=np.float32)

        for i, raytracer in enumerate(self.raytracer_list):
            if self.verbose:
                print(f"  [Simulator] Simulating mode '{mode}'. Raytracer {i+1}/{len(self.raytracer_list)}...")
            if isinstance(raytracer, RayTracingSolver) and self.surface_echoes:
                self.fmcs += self._simulate_focus_raytracer(raytracer, mode, alpha_step, dist_tol, delta_alpha)
            else:
                raise NotImplementedError(f"Unknown raytracer type: {type(raytracer)}")

    def _simulate_focus_raytracer(self, focus_raytracer: RayTracing, mode, alpha_step, dist_tol, delta_alpha):
        Nel = focus_raytracer.transducer.num_elem
        Nt = len(self.tspan)
        Nsim = len(self.sim_list)

        # Pass the solver parameters from the get_response() call
        print(f"    [Sim.Kernel] Solving rays for {Nsim} reflectors (Mode: {mode})...")
        # solve() now returns cupy arrays for tofs and amplitudes dict
        tofs_cp, amplitudes_cp, _ = focus_raytracer.solve(
            self.xf, self.zf, mode=mode, 
            alpha_step=alpha_step, 
            dist_tol=dist_tol,
            delta_alpha=delta_alpha
        )
        
        # Convert results from CuPy back to NumPy for Numba kernel
        print(f"    [Sim.Kernel] Moving raytracing results from GPU to CPU...")
        tofs_np = cp.asnumpy(tofs_cp)
        amplitudes_np = {k: cp.asnumpy(v) for k, v in amplitudes_cp.items()}

        fmcs = np.zeros(shape=(Nt, Nel, Nel, Nsim), dtype=np.float32)

        print(f"    [Sim.Kernel] Ray tracing complete. Applying FMC kernel for {Nsim} reflectors...")
        for i in prange(Nsim):
            tx_coeff_i = amplitudes_np['final_amplitude'][..., i]
            rx_coeff_i = amplitudes_np['directivity'][..., i] * amplitudes_np['final_amplitude_volta'][..., i]
            tofs_i = tofs_np[..., i]
            tofs_i = np.tile(tofs_i[:, np.newaxis], reps=(1, Nel))

            fmcs[..., i] = fmc_sim_kernel(
                self.tspan,
                tofs_i, tofs_i.T,
                tx_coeff_i, rx_coeff_i,
                Nel, focus_raytracer.transducer.fc, focus_raytracer.transducer.bw
            )
        
        print(f"    [Sim.Kernel] FMC kernel complete.")
        return fmcs

    def __get_sscan(self, mode, alpha_step, dist_tol, delta_alpha):
        self.fmcs = self.__get_fmc(mode, alpha_step, dist_tol, delta_alpha)
        print(f"  [Simulator] Applying Delay-and-Sum for S-Scan...")
        self.sscans = fmc2sscan(
            self.fmcs,
            self.shifts_e,
            self.shifts_r,
            self.transducer.num_elem
        )
        return self.sscans

    def __get_fmc(self, mode, alpha_step, dist_tol, delta_alpha):
        if self.fmcs is None:
            self.__simulate(mode, alpha_step, dist_tol, delta_alpha)
        return self.fmcs

    def get_response(self, mode, alpha_step=1e-3, dist_tol=100, delta_alpha=30e-3):
        if len(self.sim_list) == 0:
            raise ValueError("No reflector set. You must add at least one reflector to simulate its response.")

        match self.response_type:
            case "s-scan":
                # Pass parameters down
                return self.__get_sscan(mode, alpha_step, dist_tol, delta_alpha)

            case "fmc":
                # Pass parameters down
                return self.__get_fmc(mode, alpha_step, dist_tol, delta_alpha)

            case _:
                raise NotImplementedError