# tumor_submodel.py
# Tumor growth:

# logistic-like growth term

# Tumor killing:

# -kkill_tumor * ATC_tumor * reachable_cells

# Update radius, volume, reachable layer, total tumor cells.
from __future__ import annotations

import numpy as np
from math import pi

from .parameters import ModelParameters
from .state_vector import StateIx

# Assumed tumor cell density [cells / cm^3].
# You can tweak this if you want tumor radius vs. cell count to match a
# particular calibration. 1e9 cells/cm^3 is a reasonable ballpark.
TUMOR_CELL_DENSITY = 1e9  # cells per cm^3


def _radius_from_cells(N_tumor: float) -> float:
    """
    Convert total tumor cells to an equivalent spherical radius [cm],
    assuming constant cell density TUMOR_CELL_DENSITY.
    """
    if N_tumor <= 0.0:
        return 0.0
    volume_cm3 = N_tumor / TUMOR_CELL_DENSITY
    r = (3.0 * volume_cm3 / (4.0 * pi)) ** (1.0 / 3.0)
    return r


def _reachable_fraction(r_cm: float, depth_cm: float) -> float:
    """
    Fraction of tumor cells that are reachable by ATCs, assuming only
    an outer shell of thickness 'depth_cm' is accessible.

    For a sphere of radius r:
        - If r <= depth: entire tumor is reachable (fraction = 1).
        - Else: reachable volume = V(r) - V(r - depth).

    Returns a value in [0, 1].
    """
    if r_cm <= 0.0:
        return 0.0
    if depth_cm >= r_cm:
        return 1.0

    V_total = (4.0 / 3.0) * pi * r_cm**3
    r_core = r_cm - depth_cm
    V_core = (4.0 / 3.0) * pi * r_core**3
    V_shell = V_total - V_core
    return max(min(V_shell / V_total, 1.0), 0.0)


def update_dydt_tumor(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
    tumor_type: str = "FL",
) -> dict:
    """Tumor growth and ATC-mediated killing.

    State:
        - TUMOR_CELLS_TOTAL: total tumor cells (cells).

    Dynamics:
        dN/dt = growth - kill

        growth = k_growth * N * (1 - tumor_capacity * N)
        kill   = kkill_tumor * C_ATC * C_Treach * 1e6

    where:
        - C_ATC   = (ATC_tumor + pATC_tumor) / (1e6 * Vnode)       [10^6 cells / L]
        - C_Treach= N_reachable / (1e6 * Vnode)                    [10^6 cells / L]
        - N_reachable = fraction_reachable * N

    Units are matched to the definition of kkill_tumor:
        kkill_tumor: L^2 / (10^6 cells · day)"""

    tum = params.tumor
    pk = params.pk

    tumor_kill_events: dict[str, float] = {}

    # --- Choose growth rate based on tumor type ---
    if tumor_type.upper() == "DLBCL":
        k_growth = tum.kgrowth_DLBCL
    else:
        k_growth = tum.kgrowth_FL

    ix_N = StateIx.TUMOR_CELLS_TOTAL
    N_tumor = max(y[ix_N], 0.0)

    if N_tumor <= 0.0:
        dydt[ix_N] += 0.0
        tumor_kill_events["tumor"] = 0.0
        return tumor_kill_events

    # logistic-like growth
    # growth = k_growth * N_tumor * (1.0 - tum.tumor_capacity * N_tumor)
    # tumor_capacity is 1 / (10^6 cells), N_tumor is in cells
    # so use N_tumor / 1e6 inside the logistic term
    growth = k_growth * N_tumor * (1.0 - tum.tumor_capacity * (N_tumor / 1e6))

    # geometry + reachable cells (you already had this logic)
    r_cm = _radius_from_cells(N_tumor)
    frac_reach = _reachable_fraction(r_cm, tum.depth)
    N_reach = frac_reach * N_tumor

    # Paper equation (Supplementary line 183) uses C_(vATC tumor), NOT including pATC
    ATC_tumor = y[StateIx.ATC_TUMOR_NODE]
    # pATC_tumor = y[StateIx.PATC_TUMOR_NODE]  # NOT included per paper equation
    N_ATC_eff = max(ATC_tumor, 0.0)  # Only vATC, not pATC

    if N_ATC_eff <= 0.0 or N_reach <= 0.0:
        kill_cells_per_day = 0.0
    else:
        Vnode_L = pk.Vnode
        kkill_tumor = tum.kkill_tumor

        C_ATC_units = N_ATC_eff / (1e6 * Vnode_L)
        C_Treach_units = N_reach / (1e6 * Vnode_L)

        v_units = kkill_tumor * C_ATC_units * C_Treach_units  # 10^6 cells/day
        kill_cells_per_day = v_units * 1e6                    # cells/day

    dNdt = growth - kill_cells_per_day

    # small safety clamp
    if N_tumor + dNdt * 1e-3 < 0.0:
        dNdt = -N_tumor

    dydt[ix_N] += dNdt

    tumor_kill_events["tumor"] = kill_cells_per_day
    return tumor_kill_events
