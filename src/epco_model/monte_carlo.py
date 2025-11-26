# # monte_carlo.py
# """
# Parallelized virtual population simulation for the epcoritamab PBPK/PD model.
# """

# from __future__ import annotations
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed

# from .parameters import (
#     get_default_parameters,
#     sample_virtual_patient,
#     ModelParameters,
# )
# from .simulation import simulate_patient
# from .state_vector import StateIx
# from .dosing import DosingRegimen, clinical_phaseI_regimen_mg


# # ------------------------------
# # ---- Parallel Worker ----------
# # ------------------------------

# def _simulate_one_patient(
#     pid: int,
#     base_params: ModelParameters,
#     dosing: DosingRegimen,
#     t_span: tuple[float, float],
#     seed: int,
#     tumor_fraction_cutoff: float,
# ) -> dict:
#     """
#     Simulate ONE virtual patient (worker for joblib).
#     Returns a metrics dict.
#     """

#     rng = np.random.default_rng(seed)

#     # Sample individual
#     params_i = sample_virtual_patient(base_params, rng=rng)

#     # Simulate ODE
#     t, y = simulate_patient(params_i, dosing, t_span=t_span)

#     t_end = t_span[1]
#     pk = params_i.pk

#     # PK metrics
#     C_plasma = y[StateIx.DRUG_PLASMA] / pk.Vplasma
#     Cmax = float(C_plasma.max())
#     AUC = float(np.trapezoid(C_plasma, t))

#     # B cells
#     B0 = float(y[StateIx.B_BLOOD, 0])
#     B_end = float(np.interp(t_end, t, y[StateIx.B_BLOOD]))
#     B_frac = B_end / B0 if B0 > 0 else np.nan

#     # Tumor
#     tumor0 = float(y[StateIx.TUMOR_CELLS_TOTAL, 0])
#     tumor_end = float(np.interp(t_end, t, y[StateIx.TUMOR_CELLS_TOTAL]))
#     tumor_frac = tumor_end / tumor0 if tumor0 > 0 else np.nan

#     responder = (tumor_frac < tumor_fraction_cutoff)

#     return dict(
#         patient_id=pid,
#         Cmax_plasma=Cmax,
#         AUC_plasma=AUC,
#         B0=B0,
#         B_end=B_end,
#         B_frac=B_frac,
#         tumor0=tumor0,
#         tumor_end=tumor_end,
#         tumor_frac=tumor_frac,
#         responder=responder,
#     )


# # ------------------------------
# # ---- Parallel Population ------
# # ------------------------------

# def run_virtual_population_parallel(
#     n_patients: int,
#     dosing: DosingRegimen,
#     t_span=(0.0, 84.0),
#     tumor_fraction_cutoff: float = 0.3,
#     seed: int = 123,
#     n_jobs: int = -1,
# ) -> pd.DataFrame:
#     """
#     Parallel Monte Carlo simulation of N virtual patients.
#     Returns a DataFrame of metrics.
#     """
#     base_params = get_default_parameters()
#     rng = np.random.default_rng(seed)

#     # Generate unique seeds per patient
#     seeds = rng.integers(0, 1_000_000_000, n_patients, dtype=np.int64)

#     # Parallel execution
#     rows = Parallel(n_jobs=n_jobs)(
#         delayed(_simulate_one_patient)(
#             pid=i,
#             base_params=base_params,
#             dosing=dosing,
#             t_span=t_span,
#             seed=int(seeds[i]),
#             tumor_fraction_cutoff=tumor_fraction_cutoff,
#         )
#         for i in range(n_patients)
#     )

#     return pd.DataFrame(rows)


# # ------------------------------
# # ---- Per-Dose Wrapper ---------
# # ------------------------------

# def run_population_for_dose_parallel(
#     full_mg: float,
#     n_patients: int,
#     priming_fraction: float = 0.1,
#     intermediate_fraction: float = 0.5,
#     t_end: float = 84.0,
#     tumor_fraction_cutoff: float = 0.3,
#     seed: int = 123,
#     n_jobs: int = -1,
# ) -> pd.DataFrame:
#     """
#     Convenience wrapper:
#     Builds a clinical Phase I regimen and runs a full parallel virtual population.
#     """

#     dosing = clinical_phaseI_regimen_mg(
#         full_mg=full_mg,
#         priming_fraction=priming_fraction,
#         intermediate_fraction=intermediate_fraction,
#         t_end=t_end,
#         first_dose_day=0.0,
#     )

#     return run_virtual_population_parallel(
#         n_patients=n_patients,
#         dosing=dosing,
#         t_span=(0.0, t_end),
#         tumor_fraction_cutoff=tumor_fraction_cutoff,
#         seed=seed,
#         n_jobs=n_jobs,
#     )


# monte_carlo.py
"""
Parallelized virtual population simulation for the epcoritamab PBPK/PD model,
with tqdm progress and robust handling of ODE failures.
"""

from __future__ import annotations
import os

# CRITICAL: Disable threading in NumPy/SciPy BLAS libraries to avoid conflicts with multiprocessing
# This MUST be set before importing numpy/scipy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from typing import Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial

from .parameters import (
    get_default_parameters,
    sample_virtual_patient,
    ModelParameters,
)
from .simulation import simulate_patient
from .state_vector import StateIx
from .dosing import DosingRegimen, clinical_phaseI_regimen_mg


# --------------------------------------------------------------------
# Worker functions
# --------------------------------------------------------------------
def _worker_wrapper(args):
    """Wrapper for multiprocessing.Pool.imap to unpack arguments."""
    worker_fn, pid, seed = args
    return worker_fn(pid=pid, seed=seed)


def _simulate_one_patient(
    pid: int,
    base_params: ModelParameters,
    dosing: DosingRegimen,
    t_span: tuple[float, float],
    seed: int,
    tumor_fraction_cutoff: float,
    n_t_eval: Optional[int] = 400,  # ↑ Increased for smoother curves (was 250)
    rtol: float = 1e-4,  # ↑ Tightened now that equations are fixed (was 1e-3)
    atol: float = 1e-7,  # ↑ Tightened now that equations are fixed (was 1e-6)
    max_step: Optional[float] = None,  # Maximum step size for ODE solver
    timeout_seconds: float = 15.0,  # Timeout per patient to catch hangs
) -> dict:
    """
    Simulate ONE virtual patient (worker for joblib).
    If the ODE solver fails or times out, return a row with failed=True and NaN metrics.
    """
    import signal
    import threading
    
    rng = np.random.default_rng(seed)
    
    # Flag to track timeout
    timed_out = [False]
    result = [None]
    exception = [None]
    
    def run_simulation():
        """Inner function to run simulation in thread."""
        try:
            # Sample individual
            params_i = sample_virtual_patient(base_params, rng=rng)

            # Simulate ODE
            t, y = simulate_patient(
                params_i, 
                dosing, 
                t_span=t_span,
                n_t_eval=n_t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )

            t_end = t_span[1]
            pk = params_i.pk

            # PK metrics
            C_plasma = y[StateIx.DRUG_PLASMA] / pk.Vplasma
            Cmax = float(C_plasma.max())
            AUC = float(np.trapezoid(C_plasma, t))

            # B cells
            B0 = float(y[StateIx.B_BLOOD, 0])
            B_end = float(np.interp(t_end, t, y[StateIx.B_BLOOD]))
            B_frac = B_end / B0 if B0 > 0 else np.nan

            # Tumor
            tumor0 = float(y[StateIx.TUMOR_CELLS_TOTAL, 0])
            tumor_end = float(np.interp(t_end, t, y[StateIx.TUMOR_CELLS_TOTAL]))
            tumor_frac = tumor_end / tumor0 if tumor0 > 0 else np.nan
            responder = bool(tumor_frac < tumor_fraction_cutoff)

            result[0] = dict(
                patient_id=pid,
                Cmax_plasma=Cmax,
                AUC_plasma=AUC,
                B0=B0,
                B_end=B_end,
                B_frac=B_frac,
                tumor0=tumor0,
                tumor_end=tumor_end,
                tumor_frac=tumor_frac,
                responder=responder,
                failed=False,
                error_message="",
            )

        except Exception as e:
            exception[0] = e
    
    # Run simulation in thread with timeout
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    sim_thread.join(timeout=timeout_seconds)
    
    # Check if timed out
    if sim_thread.is_alive():
        # Thread is still running - timed out
        timeout_msg = f"TIMEOUT: Patient {pid} (seed={seed}) exceeded {timeout_seconds}s limit"
        print(f"[WARNING] {timeout_msg}")
        return dict(
            patient_id=pid,
            Cmax_plasma=np.nan,
            AUC_plasma=np.nan,
            B0=np.nan,
            B_end=np.nan,
            B_frac=np.nan,
            tumor0=np.nan,
            tumor_end=np.nan,
            tumor_frac=np.nan,
            responder=False,
            failed=True,
            error_message=timeout_msg,
        )
    
    # Check if exception occurred
    if exception[0] is not None:
        e = exception[0]
        import traceback
        tb_str = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {str(e)}\n{tb_str}"
        
        # Add diagnostic info about dosing if it's a comparison error
        if "'<=' not supported" in str(e) or "NoneType" in str(e):
            try:
                dosing_info = f"DosingRegimen has {len(dosing.events)} events\n"
                for i, ev in enumerate(dosing.events[:5]):  # First 5 events
                    dosing_info += f"  Event {i}: time={ev.time} (type={type(ev.time)}), amount={ev.amount} (type={type(ev.amount)}), route={ev.route}\n"
                error_msg = f"{error_msg}\n\nDosing diagnostics:\n{dosing_info}"
            except:
                error_msg = f"{error_msg}\n\nCould not inspect dosing object"
        
        return dict(
            patient_id=pid,
            Cmax_plasma=np.nan,
            AUC_plasma=np.nan,
            B0=np.nan,
            B_end=np.nan,
            B_frac=np.nan,
            tumor0=np.nan,
            tumor_end=np.nan,
            tumor_frac=np.nan,
            responder=False,
            failed=True,
            error_message=error_msg[:2000],
        )
    
    # Success - return result
    if result[0] is not None:
        return result[0]
    
    # Should never reach here
    return dict(
        patient_id=pid,
        Cmax_plasma=np.nan,
        AUC_plasma=np.nan,
        B0=np.nan,
        B_end=np.nan,
        B_frac=np.nan,
        tumor0=np.nan,
        tumor_end=np.nan,
        tumor_frac=np.nan,
        responder=False,
        failed=True,
        error_message="Unknown error in simulation",
    )


# --------------------------------------------------------------------
# Virtual population
# --------------------------------------------------------------------
def run_virtual_population_parallel(
    n_patients: int,
    dosing: DosingRegimen,
    t_span=(0.0, 84.0),
    tumor_fraction_cutoff: float = 0.3,
    seed: int = 123,
    n_jobs: int = 1,  # Default to sequential (multiprocessing has stability issues)
    backend: str = "sequential",   # "sequential" is stable, "loky"/"threading" are experimental
    n_t_eval: Optional[int] = 250,  # Relaxed default (was 500) for better performance
    rtol: float = 1e-3,  # Relaxed default (was 1e-4) for better performance
    atol: float = 1e-6,  # Relaxed default (was 1e-7) for better performance
    max_step: Optional[float] = None,  # Maximum step size for ODE solver
    timeout_per_patient: Optional[float] = 15.0,  # Timeout in seconds per patient (default 15s)
) -> pd.DataFrame:
    """
    Parallel Monte Carlo simulation of N virtual patients.
    Returns a DataFrame of metrics, including a 'failed' column.
    """

    import time
    start_time = time.time()
    
    # Determine effective number of workers
    cpu_count = os.cpu_count() or 4
    if n_jobs == -1:
        # Cap at number of patients to avoid overhead
        effective_n_jobs = min(cpu_count, n_patients)
    else:
        effective_n_jobs = min(n_jobs, n_patients)
    
    print(f"[Main] Starting population: n_patients={n_patients}, backend={backend}, n_jobs={effective_n_jobs} (capped to n_patients)")
    if timeout_per_patient is not None:
        print(f"[Main] Timeout per patient: {timeout_per_patient}s")

    base_params = get_default_parameters()
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000_000, n_patients, dtype=np.int64)

    # Execute simulations
    # Sequential is the only stable method - multiprocessing has thread contention issues
    rows = []
    
    print(f"[Main] Using sequential execution (stable)")
    timeout_val = timeout_per_patient if timeout_per_patient else 300.0
    print(f"[Main] Timeout per patient: {timeout_val}s")
    
    # Track timeout info
    timeout_patients = []
    
    with tqdm(total=n_patients, desc="Patients") as pbar:
        for i in range(n_patients):
            result = _simulate_one_patient(
                pid=i,
                base_params=base_params,
                dosing=dosing,
                t_span=t_span,
                seed=int(seeds[i]),
                tumor_fraction_cutoff=tumor_fraction_cutoff,
                n_t_eval=n_t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                timeout_seconds=timeout_val,
            )
            rows.append(result)
            
            # Track timeouts for summary
            if result['failed'] and 'TIMEOUT' in result.get('error_message', ''):
                timeout_patients.append((i, int(seeds[i])))
            
            pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"[Main] Completed {n_patients} patients in {elapsed:.2f}s ({elapsed/n_patients:.2f}s per patient)")

    df = pd.DataFrame(rows)

    n_failed = int(df["failed"].sum())
    if n_failed > 0:
        print(f"[Main] WARNING: {n_failed} / {n_patients} patients failed ODE solve.")
        # You can inspect df[df.failed] to see their error_message.

    return df



# --------------------------------------------------------------------
# Population trajectories (for Figure 3 / other plots)
# --------------------------------------------------------------------
from typing import List, Tuple, Dict
from joblib import Parallel, delayed

def _simulate_one_patient_trajectory(
    base_params: ModelParameters,
    dosing: DosingRegimen,
    t_eval: np.ndarray,
    seed: int,
    rtol: float,
    atol: float,
    max_step: Optional[float],
    timeout_seconds: float,
) -> Tuple[Optional[np.ndarray], Optional[ModelParameters], bool, str]:
    """
    Simulate ONE patient trajectory (worker for joblib).
    
    Returns:
        y : trajectory array [n_states, n_time] or None if failed
        params_i : patient-specific parameters or None if failed
        failed : bool
        error_message : str
    """
    import threading
    
    rng = np.random.default_rng(seed)
    result = [None]
    params_result = [None]
    exception = [None]
    
    def run_simulation():
        """Inner function to run simulation in thread."""
        try:
            params_i = sample_virtual_patient(base_params, rng=rng)
            params_result[0] = params_i
            t, y = simulate_patient(
                params_i,
                dosing,
                t_span=(t_eval[0], t_eval[-1]),
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            result[0] = y
        except Exception as e:
            exception[0] = e
    
    # Run simulation in thread with timeout
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    sim_thread.join(timeout=timeout_seconds)
    
    # Check if timed out
    if sim_thread.is_alive():
        return None, None, True, f"TIMEOUT (>{timeout_seconds:.1f}s)"
    
    # Check if exception occurred
    if exception[0] is not None:
        return None, params_result[0], True, str(exception[0])
    
    # Success
    return result[0], params_result[0], False, ""


def simulate_population_trajectories_for_dose(
    full_mg: float,
    n_patients: int,
    priming_fraction: float = 0.1,
    intermediate_fraction: float = 0.5,
    t_end: float = 168.0,
    first_dose_day: float = 0.0,
    seed: int = 123,
    n_t_eval: Optional[int] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_step: Optional[float] = None,
    timeout_per_patient: Optional[float] = None,
    n_jobs: int = -1,
    backend: str = "loky",
) -> Tuple[np.ndarray, List[Optional[np.ndarray]], Dict]:
    """
    Run a virtual population for a given full dose and keep
    patient-specific parameter sets in meta.
    
    Returns:
        t : 1D array (common time grid for all patients) [n_time]
        Ys : list of y-trajectories (y or None if failed) [n_patients]
             each y has shape [n_states, n_time]
        meta : dict with
            - 'params': list[ModelParameters] - patient-specific params
            - 'failed': list[bool] - failure flags
            - 'error_messages': list[str] - error messages
            - 'seeds': list[int] - random seeds
            - 'dose_mg': float - full dose level
    """
    # Choose solver settings from dose-based helper
    dose_settings = get_solver_settings_for_dose(full_mg)
    if n_t_eval is None:
        n_t_eval = dose_settings["n_t_eval"]
    if rtol is None:
        rtol = dose_settings["rtol"]
    if atol is None:
        atol = dose_settings["atol"]
    if max_step is None:
        max_step = dose_settings["max_step"]
    
    # Auto-calculate timeout based on dose if not provided
    if timeout_per_patient is None:
        timeout_per_patient = calculate_timeout_for_dose(full_mg)

    print(
        f"[Traj] Dose {full_mg} mg: n_patients={n_patients}, "
        f"rtol={rtol:.0e}, atol={atol:.0e}, max_step={max_step}, n_t_eval={n_t_eval}, "
        f"timeout={timeout_per_patient:.1f}s, backend={backend}"
    )

    # Build regimen
    dosing = clinical_phaseI_regimen_mg(
        full_mg=full_mg,
        priming_fraction=priming_fraction,
        intermediate_fraction=intermediate_fraction,
        t_end=t_end,
        first_dose_day=first_dose_day,
    )

    base_params = get_default_parameters()
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000_000, n_patients, dtype=np.int64)

    # Fixed time grid for all patients so we can overlay trajectories
    t_eval = np.linspace(0.0, t_end, n_t_eval)

    from time import time
    t_start = time()

    # Run parallel simulations
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_simulate_one_patient_trajectory)(
            base_params, dosing, t_eval, int(s),
            rtol, atol, max_step, timeout_per_patient
        )
        for s in tqdm(seeds, desc=f"Traj {full_mg} mg")
    )

    # Unpack results
    Ys = []
    params_list = []
    failed_list = []
    error_messages = []
    
    for y_i, params_i, failed, err_msg in results:
        Ys.append(y_i)
        params_list.append(params_i)
        failed_list.append(failed)
        error_messages.append(err_msg)

    elapsed = time() - t_start
    n_failed = sum(failed_list)
    print(f"[Traj] Completed {n_patients} patients at {full_mg} mg in {elapsed:.2f}s")
    if n_failed > 0:
        print(f"[Traj] WARNING: {n_failed}/{n_patients} patients failed for dose {full_mg} mg")

    meta = {
        "params": params_list,
        "failed": failed_list,
        "error_messages": error_messages,
        "seeds": seeds.tolist(),
        "dose_mg": full_mg,
    }

    return t_eval, Ys, meta


# --------------------------------------------------------------------
# Exposure (C_avg) calculations with patient-specific parameters
# --------------------------------------------------------------------

def compute_cavg_from_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    params: ModelParameters,
    t_end: float = 84.0,
) -> float:
    """
    Compute time-averaged plasma concentration (C_avg) from a single patient trajectory.
    
    Args:
        t : time array [n_time]
        y : state array [n_states, n_time]
        params : patient-specific parameters (uses params.pk.Vplasma)
        t_end : end of averaging window (days)
    
    Returns:
        C_avg in nM
    """
    # Extract plasma drug amount and convert to concentration
    A_plasma = y[StateIx.DRUG_PLASMA, :]  # nmol
    C_plasma = A_plasma / params.pk.Vplasma  # nM
    
    # Restrict to [0, t_end]
    mask = t <= t_end
    t_window = t[mask]
    C_window = C_plasma[mask]
    
    # Time-averaged concentration using trapezoidal integration
    if len(t_window) < 2:
        return 0.0
    
    AUC = np.trapezoid(C_window, t_window)  # nM·day
    C_avg = AUC / t_window[-1]  # nM
    
    return float(C_avg)


def compute_cavg_distributions(
    dose_to_trajs: Dict,
    t_end: float = 84.0,
) -> Dict[float, np.ndarray]:
    """
    Compute C_avg distribution for each dose using each patient's
    own parameter set (not a common base_params).
    
    Args:
        dose_to_trajs : dict[dose] -> (t, Ys, meta)
            meta['params'] must be a list of ModelParameters
            meta['failed'] must be a list of booleans
        t_end : end of averaging window (days)
    
    Returns:
        dict[dose] -> np.ndarray of C_avg values [n_patients]
            (NaN for failed patients)
    """
    dose_to_cavg = {}
    
    for dose, (t, Ys, meta) in dose_to_trajs.items():
        params_list = meta["params"]
        failed_list = meta["failed"]
        
        cavg_list = []
        for y, params_i, failed in zip(Ys, params_list, failed_list):
            if failed or y is None or params_i is None:
                cavg_list.append(np.nan)
                continue
            
            cavg = compute_cavg_from_trajectory(t, y, params_i, t_end=t_end)
            cavg_list.append(cavg)
        
        cavg_array = np.array(cavg_list)
        dose_to_cavg[dose] = cavg_array
        
        n_success = np.sum(~np.isnan(cavg_array))
        print(
            f"Dose {dose:7.4f} mg: "
            f"{n_success:3d}/{len(cavg_array)} patients, "
            f"C_avg median={np.nanmedian(cavg_array):7.3f} nM, "
            f"mean={np.nanmean(cavg_array):7.3f} nM"
        )
    
    return dose_to_cavg


# --------------------------------------------------------------------
# Dynamic timeout and stiffness handling
# --------------------------------------------------------------------
def calculate_timeout_for_dose(full_mg: float, base_timeout: float = 15.0) -> float:
    """
    Calculate dynamic timeout based on dose.
    Higher doses lead to stiffer ODEs and longer solve times.
    
    Uses exponential scaling: timeout = base_timeout * (1 + dose_factor)
    where dose_factor increases with dose.
    
    Args:
        full_mg: Full maintenance dose in mg
        base_timeout: Base timeout for low doses (default 15s)
    
    Returns:
        Timeout in seconds
    """
    # Empirical scaling: doses >8mg are significantly stiffer
    if full_mg <= 1.6:
        # Low doses: base timeout
        return base_timeout
    elif full_mg <= 8.0:
        # Medium doses: 2x timeout
        return base_timeout * 2.0
    elif full_mg <= 24.0:
        # High doses: 4x timeout
        return base_timeout * 4.0
    else:
        # Very high doses (48, 60mg): 6x timeout
        return base_timeout * 6.0


def get_solver_settings_for_dose(full_mg: float) -> dict:
    """
    Get optimized solver settings based on dose level.
    Now that equations are fixed, we can use tighter tolerances across all doses.
    
    Args:
        full_mg: Full maintenance dose in mg
    
    Returns:
        Dictionary with n_t_eval, rtol, atol, max_step
    """
    if full_mg <= 1.6:
        # Low doses: tightest settings for accuracy
        return {
            "n_t_eval": 400,  # ↑ Increased for smoother curves
            "rtol": 1e-4,     # ↑ Tightened (was 1e-3)
            "atol": 1e-7,     # ↑ Tightened (was 1e-6)
            "max_step": None,  # Automatic
        }
    elif full_mg <= 8.0:
        # Medium doses: good balance
        return {
            "n_t_eval": 350,  # ↑ Increased
            "rtol": 2e-4,     # ↑ Tightened (was 2e-3)
            "atol": 2e-7,     # ↑ Tightened (was 2e-6)
            "max_step": 1.0,  # Allow larger steps than before
        }
    elif full_mg <= 24.0:
        # High doses: still accurate
        return {
            "n_t_eval": 300,  # ↑ Increased
            "rtol": 5e-4,     # ↑ Tightened (was 5e-3)
            "atol": 5e-7,     # ↑ Tightened (was 5e-6)
            "max_step": 0.5,  # Moderate steps
        }
    else:
        # Very high doses: slightly relaxed but still much better than before
        return {
            "n_t_eval": 250,  # ↑ Increased
            "rtol": 1e-3,     # ↑ Tightened (was 1e-2)
            "atol": 1e-6,     # ↑ Tightened (was 1e-5)
            "max_step": 0.3,  # Small steps for stability
        }


# --------------------------------------------------------------------
# Per-dose wrapper
# --------------------------------------------------------------------
def run_population_for_dose_parallel(
    full_mg: float,
    n_patients: int,
    priming_fraction: float = 0.1,
    intermediate_fraction: float = 0.5,
    t_end: float = 84.0,
    tumor_fraction_cutoff: float = 0.3,
    seed: int = 123,
    n_jobs: int = -1,
    backend: str = "sequential",  # Only stable option - parallel execution has issues
    n_t_eval: Optional[int] = None,  # If None, auto-selects based on dose
    rtol: Optional[float] = None,  # If None, auto-selects based on dose
    atol: Optional[float] = None,  # If None, auto-selects based on dose
    max_step: Optional[float] = None,  # If None, auto-selects based on dose
    timeout_per_patient: Optional[float] = None,  # If None, auto-calculates based on dose
) -> pd.DataFrame:
    """
    Convenience wrapper:
    Builds a clinical Phase I regimen and runs a full parallel virtual population.
    
    Automatically adjusts timeout and solver settings based on dose level.
    Higher doses use longer timeouts and relaxed tolerances to handle stiffness.
    """

    print(f"[Main] Building regimen for {full_mg} mg")

    # Auto-select solver settings based on dose if not provided
    dose_settings = get_solver_settings_for_dose(full_mg)
    if n_t_eval is None:
        n_t_eval = dose_settings["n_t_eval"]
    if rtol is None:
        rtol = dose_settings["rtol"]
    if atol is None:
        atol = dose_settings["atol"]
    if max_step is None:
        max_step = dose_settings["max_step"]
    
    # Auto-calculate timeout based on dose if not provided
    if timeout_per_patient is None:
        timeout_per_patient = calculate_timeout_for_dose(full_mg)
    
    print(f"[Main] Dose {full_mg} mg: timeout={timeout_per_patient:.1f}s, "
          f"rtol={rtol:.0e}, atol={atol:.0e}, max_step={max_step}, n_t_eval={n_t_eval}")

    dosing = clinical_phaseI_regimen_mg(
        full_mg=full_mg,
        priming_fraction=priming_fraction,
        intermediate_fraction=intermediate_fraction,
        t_end=t_end,
        first_dose_day=0.0,
    )

    return run_virtual_population_parallel(
        n_patients=n_patients,
        dosing=dosing,
        t_span=(0.0, t_end),
        tumor_fraction_cutoff=tumor_fraction_cutoff,
        seed=seed,
        n_jobs=n_jobs,
        backend=backend,
        n_t_eval=n_t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        timeout_per_patient=timeout_per_patient,
    )
