# bcell_kill_submodel.py
# Adds second-order term -kkill_BC * ATC * B to the B-cell state derivatives in each compartment.

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import StateIx


def _bcell_kill_rate_compartment(
    N_ATC: float,
    N_pATC: float,
    N_B: float,
    V_L: float,
    kkill_BC: float,
) -> float:
    """
    Compute B-cell killing rate in ONE compartment, in units of [cells/day].

    - N_ATC, N_pATC, N_B: counts of cells in this compartment (cells)
    - V_L: compartment volume [L]
    - kkill_BC: parameter with units L^2 / (10^6 cells · day)

    The original model uses concentrations in units of '10^6 cells / L'.
    Here, our states are in raw cells, so we convert:

        C_ATC_units = N_eff / (1e6 * V)
        C_B_units   = N_B   / (1e6 * V)

    v_units = kkill_BC * C_ATC_units * C_B_units   # in [10^6 cells / day]
    dN_B/dt = - v_units * 1e6                      # in [cells / day]

    which simplifies to:

        dN_B/dt = - kkill_BC * N_eff * N_B / (1e6 * V^2)
    """
    if N_B <= 0.0:
        return 0.0

    N_eff = N_ATC + N_pATC  # total effector T cells (activated + expanded)
    if N_eff <= 0.0:
        return 0.0

    # Compute rate in [10^6 cells / day]
    C_ATC_units = N_eff / (1e6 * V_L)  # "10^6 cells / L"
    C_B_units   = N_B  / (1e6 * V_L)

    v_units = kkill_BC * C_ATC_units * C_B_units   # 10^6 cells / day
    v_cells = v_units * 1e6                        # cells / day

    return v_cells


def update_dydt_bcell_kill(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
) -> None:
    """
    B-cell killing by activated T cells in blood, spleen, node, and lymph.

    For each compartment:

        dB/dt|kill = - v_kill  [cells/day]

    where v_kill is computed from kkill_BC, ATC_B + pATC_B, and B.

    This submodel ONLY modifies B-cell states. It does not yet feed back
    into the binding submodel (trimer -> CD3-Ab conversion); that can be
    layered in later if needed.
    """
    pk   = params.pk
    traf = params.trafficking
    bkill = params.bkill

    kkill_BC = bkill.kkill_BC

    # ---- Blood ----
    B_blood   = y[StateIx.B_BLOOD]
    ATC_blood = y[StateIx.ATC_B_BLOOD]
    pATC_blood = y[StateIx.PATC_B_BLOOD]

    V_blood_L = traf.Vblood  # use blood volume for B-cell density
    v_kill_blood = _bcell_kill_rate_compartment(
        N_ATC=ATC_blood,
        N_pATC=pATC_blood,
        N_B=B_blood,
        V_L=V_blood_L,
        kkill_BC=kkill_BC,
    )
    dydt[StateIx.B_BLOOD] -= v_kill_blood

    # ---- Spleen ----
    B_spleen   = y[StateIx.B_SPLEEN]
    ATC_spleen = y[StateIx.ATC_B_SPLEEN]
    pATC_spleen = y[StateIx.PATC_B_SPLEEN]

    V_spleen_L = traf.Vspleen_tissue  # tissue volume for B-cell density
    v_kill_spleen = _bcell_kill_rate_compartment(
        N_ATC=ATC_spleen,
        N_pATC=pATC_spleen,
        N_B=B_spleen,
        V_L=V_spleen_L,
        kkill_BC=kkill_BC,
    )
    dydt[StateIx.B_SPLEEN] -= v_kill_spleen

    # ---- Node ----
    B_node   = y[StateIx.B_NODE]
    ATC_node = y[StateIx.ATC_B_NODE]
    pATC_node = y[StateIx.PATC_B_NODE]

    V_node_L = pk.Vnode
    v_kill_node = _bcell_kill_rate_compartment(
        N_ATC=ATC_node,
        N_pATC=pATC_node,
        N_B=B_node,
        V_L=V_node_L,
        kkill_BC=kkill_BC,
    )
    dydt[StateIx.B_NODE] -= v_kill_node

    # ---- Lymph ----
    B_lymph   = y[StateIx.B_LYMPH]
    ATC_lymph = y[StateIx.ATC_B_LYMPH]
    pATC_lymph = y[StateIx.PATC_B_LYMPH]

    V_lymph_L = pk.Vlymph
    v_kill_lymph = _bcell_kill_rate_compartment(
        N_ATC=ATC_lymph,
        N_pATC=pATC_lymph,
        N_B=B_lymph,
        V_L=V_lymph_L,
        kkill_BC=kkill_BC,
    )
    dydt[StateIx.B_LYMPH] -= v_kill_lymph
