# simulation.py
"""
Single-patient PBPK/PD simulation for the epcoritamab model.
Supports piecewise integration with multiple dosing events.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .parameters import ModelParameters
from .state_vector import StateIx, get_initial_state
from .dosing import DosingRegimen
from .odes import rhs


def simulate_patient(
    params: ModelParameters,
    dosing: Optional[DosingRegimen],
    t_span: Tuple[float, float] = (0.0, 84.0),
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    n_t_eval: Optional[int] = None,
    max_step: Optional[float] = None,
):
    """
    Simulate a single patient over t_span with piecewise integration for multiple doses.

    Args:
        params: ModelParameters instance
        dosing: DosingRegimen with dose events
        t_span: (t0, t_end) time span in days
        t_eval: Optional array of time points for output. If None, uses n_t_eval.
        rtol: Relative tolerance for BDF solver (default 1e-4)
        atol: Absolute tolerance for BDF solver (default 1e-7)
        n_t_eval: Number of time points for output if t_eval is None (default 500)
        max_step: Maximum step size for solver (None = automatic)

    Returns:
        t: Time array [n_time]
        y: State array [n_states, n_time]
    """
    # Unpack t_span
    t0, t_end = t_span

    # Validate dosing regimen
    if dosing is None:
        raise ValueError("dosing cannot be None")
    if not hasattr(dosing, 'events') or dosing.events is None:
        raise ValueError("dosing.events is None or missing")
    
    # Filter out any None events (defensive check)
    valid_events = []
    for i, ev in enumerate(dosing.events):
        if ev is None:
            raise ValueError(f"DoseEvent at index {i} is None")
        valid_events.append(ev)
    
    # Sort dose events by time and filter to events within t_span
    dose_events = sorted(valid_events, key=lambda e: e.time)
    dose_times = [ev.time for ev in dose_events if t0 <= ev.time <= t_end]

    # Generate t_eval if not provided
    if t_eval is None:
        if n_t_eval is None:
            n_t_eval = 250  # Relaxed default (was 500) for better performance
        t_eval = np.linspace(t0, t_end, n_t_eval)

    # If no doses or only one segment, use simple integration
    if len(dose_times) <= 1:
        y0 = get_initial_state(params)
        
        # Apply initial dose if any
        for ev in dose_events:
            if abs(ev.time - t0) < 1e-9 and ev.route.upper() == "SC":
                y0[StateIx.DRUG_SC] += ev.amount
                y0[StateIx.INJ] = 1.0  # Injection effect triggers T-cell redistribution

        solve_kwargs = {
            "fun": lambda t, y: rhs(t, y, params),
            "t_span": (t0, t_end),
            "y0": y0,
            "t_eval": t_eval,
            "method": "BDF",
            "rtol": rtol,
            "atol": atol,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step
        
        sol = solve_ivp(**solve_kwargs)

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.t, sol.y

    # Piecewise integration: integrate between each dose event
    all_times = [t0] + dose_times + [t_end]
    all_times = sorted(set(all_times))  
    
    t_segments = []
    y_segments = []

    y_current = get_initial_state(params)

    n_segments = len(all_times) - 1
    # Note: 24 segments from weekly dosing is normal; only warn if excessive
    if n_segments > 50:  # Only log if unusually many segments
        import warnings
        warnings.warn(f"Piecewise integration: {n_segments} segments from {len(dose_times)} dose events")

    for i in range(n_segments):
        t_seg_start = all_times[i]
        t_seg_end = all_times[i + 1]

        # Apply doses at the start of this segment
        for ev in dose_events:
            if abs(ev.time - t_seg_start) < 1e-9 and ev.route.upper() == "SC":
                y_current[StateIx.DRUG_SC] += ev.amount
                y_current[StateIx.INJ] = 1.0  # Injection effect triggers T-cell redistribution

        # Time points for this segment (filter t_eval within segment)
        t_seg_eval = t_eval[(t_eval >= t_seg_start) & (t_eval <= t_seg_end)]
        if len(t_seg_eval) == 0:
            # Ensure at least start and end points
            t_seg_eval = np.array([t_seg_start, t_seg_end])
        elif t_seg_eval[0] != t_seg_start:
            t_seg_eval = np.concatenate([[t_seg_start], t_seg_eval])
        if t_seg_eval[-1] != t_seg_end:
            t_seg_eval = np.concatenate([t_seg_eval, [t_seg_end]])

        # Integrate this segment
        # Build solve_ivp kwargs, only include max_step if it's not None
        solve_kwargs = {
            "fun": lambda t, y: rhs(t, y, params),
            "t_span": (t_seg_start, t_seg_end),
            "y0": y_current,
            "t_eval": t_seg_eval,
            "method": "BDF",
            "rtol": rtol,
            "atol": atol,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step
        
        sol = solve_ivp(**solve_kwargs)

        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed in segment [{t_seg_start:.2f}, {t_seg_end:.2f}]: {sol.message}"
            )

        # Store results (avoid duplicate endpoints)
        if i == 0:
            t_segments.append(sol.t)
            y_segments.append(sol.y)
        else:
            # Skip first point to avoid duplicate
            t_segments.append(sol.t[1:])
            y_segments.append(sol.y[:, 1:])

        # Update state for next segment
        y_current = sol.y[:, -1]

    t_full = np.concatenate(t_segments)
    y_full = np.concatenate(y_segments, axis=1)

    # Interpolate to requested t_eval if needed
    if len(t_full) != len(t_eval) or not np.allclose(t_full, t_eval):
        from scipy.interpolate import interp1d
        y_interp = np.zeros((y_full.shape[0], len(t_eval)))
        for i in range(y_full.shape[0]):
            interp_func = interp1d(t_full, y_full[i, :], kind="linear", 
                                   bounds_error=False, fill_value="extrapolate")
            y_interp[i, :] = interp_func(t_eval)
        return t_eval, y_interp

    return t_full, y_full
