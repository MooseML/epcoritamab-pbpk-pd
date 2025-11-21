# trafficking_submodel.py
# Takes T-cell, B-cell, ATC populations and returns:
# dT_comp/dt terms from trafficking + natural death + production + injection effect + homeostasis
# trafficking_submodel.py
"""
Lymphocyte trafficking and turnover submodel.
---------------------------------------------

Implements the movement and turnover of T cells and B cells between:

- Blood / plasma
- Spleen tissue
- Tumor lymph node
- Lymph

plus:
- Zero-order production (kin) into blood
- First-order natural death (kout) in all compartments
- Adaptive feedback (AF) homeostasis (kt, r)
- Injection-effect compartment (INJ) that transiently increases
  trafficking from blood to spleen after each injection.

The same structure is used for:
- T cells (T_*)
- B cells (B_*)
"""

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import StateIx


def _update_single_cell_type(
    *,
    # state indices for this cell type
    ix_blood: int,
    ix_spleen: int,
    ix_node: int,
    ix_lymph: int,
    ix_af: int,
    # parameters
    kin: float,
    kout: float,
    baseline_plasma_density: float,
    params: ModelParameters,
    # global state
    y: np.ndarray,
    dydt: np.ndarray,
) -> None:
    """
    Generic lymphocyte trafficking for either T cells or B cells.

    Args:
        ix_blood, ix_spleen, ix_node, ix_lymph, ix_af:
            Indices into the state vector for blood/spleen/node/lymph
            cell counts and the adaptive-feedback (AF) state.
        kin: Zero-order production rate (cells/day).
        kout: First-order natural death rate (1/day).
        baseline_plasma_density: Baseline blood density (cells/mm^3),
            used in the homeostasis feedback term.
        params: ModelParameters (to get trafficking + homeostasis params).
        y: Current state vector.
        dydt: Derivative vector to be updated in place.
    """
    traf = params.trafficking

    # Unpack states
    cell_blood = y[ix_blood]
    cell_spleen = y[ix_spleen]
    cell_node = y[ix_node]
    cell_lymph = y[ix_lymph]

    AF = y[ix_af]  # adaptive feedback state

    # Injection-effect compartment (shared for T and B)
    INJ = y[StateIx.INJ]

    # ---------- Homeostasis (adaptive feedback) ----------
    # AF dynamics:
    #   dAF/dt = kt * ( (baseline / blood)^r - AF )
    eps = 1e-12  # avoid division by zero
    ratio = baseline_plasma_density / (cell_blood / (traf.Vblood * 1e6) + eps)
    target_AF = ratio ** traf.r
    dAF = traf.kt * (target_AF - AF)

    # ---------- Production ----------
    # Zero-order production into blood, scaled by AF:
    Ksyn = kin * AF  # cells/day

    # ---------- Trafficking flows ----------
    # blood -> spleen -> node -> lymph -> blood
    kpt = traf.kpt   # blood -> spleen
    ktn = traf.ktn   # spleen -> node
    knl = traf.knl   # node -> lymph
    klp = traf.klp   # lymph -> blood

    # Injection-effect increases trafficking from blood to spleen
    inj_factor = 1.0 + traf.INJ_Scaler * INJ

    Flow_blood_to_spleen = kpt * cell_blood * inj_factor
    Flow_spleen_to_node = ktn * cell_spleen
    Flow_node_to_lymph = knl * cell_node
    Flow_lymph_to_blood = klp * cell_lymph

    # ---------- Natural death (in all compartments) ----------
    death_blood = kout * cell_blood
    death_spleen = kout * cell_spleen
    death_node = kout * cell_node
    death_lymph = kout * cell_lymph

    # ---------- Assemble compartment derivatives ----------
    dCell_blood = (
        + Ksyn
        - death_blood
        - Flow_blood_to_spleen
        + Flow_lymph_to_blood
    )

    dCell_spleen = (
        + Flow_blood_to_spleen
        - death_spleen
        - Flow_spleen_to_node
    )

    dCell_node = (
        + Flow_spleen_to_node
        - death_node
        - Flow_node_to_lymph
    )

    dCell_lymph = (
        + Flow_node_to_lymph
        - death_lymph
        - Flow_lymph_to_blood
    )

    # ---------- Write back to global dydt ----------
    dydt[ix_blood] += dCell_blood
    dydt[ix_spleen] += dCell_spleen
    dydt[ix_node] += dCell_node
    dydt[ix_lymph] += dCell_lymph
    dydt[ix_af] += dAF


def update_dydt_trafficking(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
) -> None:
    """
    Update lymphocyte trafficking + turnover contributions to dydt.

    Handles both:
      - T cells (T_*)
      - B cells (B_*)
    sharing the same trafficking structure but with different kin/kout/baseline.
    """
    traf = params.trafficking

    # ---------- T cells ----------
    _update_single_cell_type(
        ix_blood=StateIx.T_BLOOD,
        ix_spleen=StateIx.T_SPLEEN,
        ix_node=StateIx.T_NODE,
        ix_lymph=StateIx.T_LYMPH,
        ix_af=StateIx.AF_T,
        kin=traf.kinTC,
        kout=traf.koutTC,
        baseline_plasma_density=traf.TCplasma_base,
        params=params,
        y=y,
        dydt=dydt,
    )

    # ---------- B cells ----------
    _update_single_cell_type(
        ix_blood=StateIx.B_BLOOD,
        ix_spleen=StateIx.B_SPLEEN,
        ix_node=StateIx.B_NODE,
        ix_lymph=StateIx.B_LYMPH,
        ix_af=StateIx.AF_B,
        kin=traf.kinBC,
        kout=traf.koutBC,
        baseline_plasma_density=traf.BCplasma_base,
        params=params,
        y=y,
        dydt=dydt,
    )

    # ---------- Injection-effect compartment ----------
    # dINJ/dt = -kdecay * INJ  (decays between injections)
    INJ = y[StateIx.INJ]
    dINJ = -traf.kdecay * INJ
    dydt[StateIx.INJ] += dINJ
