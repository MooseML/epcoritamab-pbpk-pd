# tcell_activation_submodel.py

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import StateIx

AVOGADRO = 6.022_140_76e23  # molecules/mol
NMOL_TO_MOLE = 1e-9  # mol per nmol

# NOTE: All lymphocyte states (T, B, ATC) are in cells, not 10^6 cells.
# sim_slope and sim_slopetumor are stored in parameters.py with paper values
# in units ((10^6 cells/day) * (cell/molecule)).
# We convert them to (cells/day * cell/molecule) by multiplying by 1e6.


def _trimers_per_cell(
    A_tri_nmols: float,
    n_cells: float,
) -> float:
    """
    Convert trimer amount [nmol] and target cell count [cells] to
    trimers per cell [molecules / cell].
    """
    if n_cells <= 0.0 or A_tri_nmols <= 0.0:
        return 0.0
    moles = A_tri_nmols * NMOL_TO_MOLE
    molecules = moles * AVOGADRO
    return molecules / n_cells  # molecules per cell


def _effective_kout_ATC(t: float, activation_params) -> float:
    """
    Effective elimination rate for ATC and pATC:

    - Before TAD + Tp: no natural death (kout = 0)
    - After TAD + Tp: kout = koutATC
    """
    if t < activation_params.TAD + activation_params.Tp:
        return 0.0
    return activation_params.koutATC


def update_dydt_tcell_activation(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
) -> None:
    """
    T-cell activation and clonal expansion submodel.

    - Virtual activated T cells against B cells (ATC_B) in:
        blood, spleen, node, lymph.
    - Clonally expanded activated T cells (pATC_B) in each compartment.
    - Trafficking of ATC_B and pATC_B uses the same kpt/ktn/knl/klp
      as normal T cells.
    - Activation rates are proportional to trimers per B cell in each
      compartment, with a delay TAD and slope sim_slope.
    - Natural elimination of ATCs is delayed by Tp.

    Tumor-specific activation (ATC_TUMOR_NODE, PATC_TUMOR_NODE) is
    structurally present but will be wired to tumor trimers once the
    tumor submodel and tumor binding are implemented.
    """
    act = params.activation
    traf = params.trafficking
    pk = params.pk

    # Paper values are in (10^6 cells/day * cell/molecule), so multiply by 1e6 to get (cells/day * cell/molecule)
    sim_slope_cells = act.sim_slope * 1e6
    sim_slope_tumor_cells = act.sim_slopetumor * 1e6

    # Effective elimination rate (0 before TAD+Tp)
    kout_eff = _effective_kout_ATC(t, act)

    # Helper: indices for B-cell side 
    # Binding trimers live in nmol; B cells and ATCs are in cells.
    blood = dict(
        name="blood",
        tri_ix=StateIx.TRIMER_BLOOD.value,
        B_ix=StateIx.B_BLOOD.value,
        ATC_ix=StateIx.ATC_B_BLOOD.value,
        pATC_ix=StateIx.PATC_B_BLOOD.value,
        volume_L=pk.Vplasma,
    )
    spleen = dict(
        name="spleen",
        tri_ix=StateIx.TRIMER_SPLEEN.value,
        B_ix=StateIx.B_SPLEEN.value,
        ATC_ix=StateIx.ATC_B_SPLEEN.value,
        pATC_ix=StateIx.PATC_B_SPLEEN.value,
        volume_L=pk.Vspleen,  # same as binding submodel
    )
    node = dict(
        name="node",
        tri_ix=StateIx.TRIMER_NODE.value,
        B_ix=StateIx.B_NODE.value,
        ATC_ix=StateIx.ATC_B_NODE.value,
        pATC_ix=StateIx.PATC_B_NODE.value,
        volume_L=pk.Vnode,
    )
    lymph = dict(
        name="lymph",
        tri_ix=StateIx.TRIMER_LYMPH.value,
        B_ix=StateIx.B_LYMPH.value,
        ATC_ix=StateIx.ATC_B_LYMPH.value,
        pATC_ix=StateIx.PATC_B_LYMPH.value,
        volume_L=pk.Vlymph,
    )

    comps = [blood, spleen, node, lymph]

    # 1) Activation and clonal expansion in each compartment 
    # accumulate local derivatives here then add trafficking.
    dATC = {c["name"]: 0.0 for c in comps}
    dPATC = {c["name"]: 0.0 for c in comps}

    for c in comps:
        name = c["name"]
        tri = y[c["tri_ix"]]
        B_cells = y[c["B_ix"]]
        ATC = y[c["ATC_ix"]]
        pATC = y[c["pATC_ix"]]

        # Trimers per B cell (molecules per cell)
        tri_per_B = _trimers_per_cell(tri, B_cells)

        # Activation only after TAD
        if t >= act.TAD:
            # First-order activation rate: sim_slope * (trimers per B cell)
            # Units: cells/day
            v_act = sim_slope_cells * tri_per_B
        else:
            v_act = 0.0

        # Activation source feeds ATC_B
        # Clonal expansion feeds pATC_B from ATC_B
        dATC[name] += v_act - kout_eff * ATC
        dPATC[name] += act.expand_factor * ATC - kout_eff * pATC

    # 2) Trafficking of ATC_B and pATC_B 
    # Same structure as T/B cell trafficking: blood -> spleen -> node -> lymph -> blood

    # Shortcuts
    ATC_blood  = y[StateIx.ATC_B_BLOOD]
    ATC_spleen = y[StateIx.ATC_B_SPLEEN]
    ATC_node   = y[StateIx.ATC_B_NODE]
    ATC_lymph  = y[StateIx.ATC_B_LYMPH]

    pATC_blood  = y[StateIx.PATC_B_BLOOD]
    pATC_spleen = y[StateIx.PATC_B_SPLEEN]
    pATC_node   = y[StateIx.PATC_B_NODE]
    pATC_lymph  = y[StateIx.PATC_B_LYMPH]

    kpt = traf.kpt
    ktn = traf.ktn
    knl = traf.knl
    klp = traf.klp

    # ATC_B trafficking
    dATC["blood"]  += -kpt * ATC_blood  + klp * ATC_lymph
    dATC["spleen"] += +kpt * ATC_blood  - ktn * ATC_spleen
    dATC["node"]   += +ktn * ATC_spleen - knl * ATC_node
    dATC["lymph"]  += +knl * ATC_node   - klp * ATC_lymph

    # pATC_B trafficking
    dPATC["blood"] += -kpt * pATC_blood  + klp * pATC_lymph
    dPATC["spleen"] += +kpt * pATC_blood  - ktn * pATC_spleen
    dPATC["node"] += +ktn * pATC_spleen - knl * pATC_node
    dPATC["lymph"] += +knl * pATC_node   - klp * pATC_lymph

    # 3) Write back into dydt 
    for c in comps:
        name = c["name"]
        ATC_ix = c["ATC_ix"]
        pATC_ix = c["pATC_ix"]
        dydt[ATC_ix]  += dATC[name]
        dydt[pATC_ix] += dPATC[name]

        # 4) Tumor-directed ATC activation (real implementation) 

        # Tumor total cells
        N_tumor = y[StateIx.TUMOR_CELLS_TOTAL]

        # If no tumor left, keep ATC_TUMOR_NODE and PATC_TUMOR_NODE at 0
        if N_tumor > 0.0:

            A_tri_tumor_nmols = y[StateIx.TRIMER_TUMOR]  # real tumor trimers now
            
            # Trimers per tumor cell (molecules / cell)
            tri_per_tum_cell = _trimers_per_cell(A_tri_tumor_nmols, N_tumor)

            # RELU threshold rule from Supplement (lines 152-154):
            # RELU = 0.01 when tri_per_tum_cell < Threshold
            # RELU = 1.0  when tri_per_tum_cell >= Threshold
            if tri_per_tum_cell < act.Trimer_Threshold:
                v_act_tumor = 0.01 * sim_slope_tumor_cells * tri_per_tum_cell  
            else:
                v_act_tumor = sim_slope_tumor_cells * tri_per_tum_cell

            # ATC and pATC states
            ATC_T = y[StateIx.ATC_TUMOR_NODE]
            pATC_T = y[StateIx.PATC_TUMOR_NODE]

            # Natural elimination after delay
            kout_eff_T = kout_eff

            # Update tumor ATC pools
            dATC_tumor = v_act_tumor - kout_eff_T * ATC_T
            dPATC_tumor = act.expand_factor * ATC_T - kout_eff_T * pATC_T

            dydt[StateIx.ATC_TUMOR_NODE] += dATC_tumor
            dydt[StateIx.PATC_TUMOR_NODE] += dPATC_tumor
        else:
            # If tumor is gone, keep values clamped to 0
            dydt[StateIx.ATC_TUMOR_NODE] += 0.0
            dydt[StateIx.PATC_TUMOR_NODE] += 0.0