# binding_submodel.py
# Law of mass action equations:

# free CD3/CD20

# CD3–Ab, CD20–Ab dimers

# trimers
# binding_submodel.py
"""
Epcoritamab binding submodel.
-----------------------------

This module will implement the law-of-mass-action binding of
epcoritamab to CD3 and CD20 and trimer formation in all
lymphocyte-containing compartments:

- Blood / plasma
- Spleen
- Tumor lymph node
- Lymph

Species (per compartment, conceptually):
- Free drug (already tracked as DRUG_* in the PK submodel)
- CD3-Ab dimers
- CD20-Ab dimers
- Trimers (T-Ab-B or T-Ab-tumor)

The exact ODEs follow the "Epcoritamab-binding submodel" section
of the supplementary methods. Once we have the R code / explicit
equations, we will fill in the rate expressions below.
"""
# binding_submodel.py

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import StateIx

AVOGADRO = 6.022_140_76e23  # molecules / mol

# If your T/B states are in "10^6 cells", set this to 1e6.
# If they are raw cells, keep it at 1.0.
CELLS_PER_UNIT = 1.0

COMPARTMENTS = {
    "blood": dict(
        drug_ix=StateIx.DRUG_PLASMA,
        T_ix=StateIx.T_BLOOD,
        B_ix=StateIx.B_BLOOD,
        d3_ix=StateIx.CD3_AB_DIMER_BLOOD,
        d20_ix=StateIx.CD20_AB_DIMER_BLOOD,
        tri_ix=StateIx.TRIMER_BLOOD,
        volume_name="Vplasma",  # use PK plasma volume
    ),
    "spleen": dict(
        drug_ix=StateIx.DRUG_SPLEEN,
        T_ix=StateIx.T_SPLEEN,
        B_ix=StateIx.B_SPLEEN,
        d3_ix=StateIx.CD3_AB_DIMER_SPLEEN,
        d20_ix=StateIx.CD20_AB_DIMER_SPLEEN,
        tri_ix=StateIx.TRIMER_SPLEEN,
        volume_name="Vspleen",  # PK spleen ISF volume
    ),
    "node": dict(
        drug_ix=StateIx.DRUG_NODE,
        T_ix=StateIx.T_NODE,
        B_ix=StateIx.B_NODE,
        d3_ix=StateIx.CD3_AB_DIMER_NODE,
        d20_ix=StateIx.CD20_AB_DIMER_NODE,      # CD20 dimers on B cells
        tri_ix=StateIx.TRIMER_NODE,             # B-cell trimers
        d20_tumor_ix=StateIx.CD20_AB_DIMER_TUMOR,
        tri_tumor_ix=StateIx.TRIMER_TUMOR,
        volume_name="Vnode",
    ),
    "lymph": dict(
        drug_ix=StateIx.DRUG_LYMPH,
        T_ix=StateIx.T_LYMPH,
        B_ix=StateIx.B_LYMPH,
        d3_ix=StateIx.CD3_AB_DIMER_LYMPH,
        d20_ix=StateIx.CD20_AB_DIMER_LYMPH,
        tri_ix=StateIx.TRIMER_LYMPH,
        volume_name="Vlymph",
    ),
    # Tumor-specific CD20 binding will be added later
}


def _receptor_conc_nmolar(cell_amount: float, receptors_per_cell: float, volume_L: float) -> float:
    """
    Convert a cell state into receptor concentration [nmol/L].
    cell_amount: T or B state value (model units, possibly "10^6 cells").
    """
    cells = cell_amount * CELLS_PER_UNIT           # cells
    molecules = receptors_per_cell * cells         # molecules
    moles = molecules / AVOGADRO                   # mol
    nmol = moles * 1e9                             # nmol
    return nmol / max(volume_L, 1e-12)             # nmol / L


def _update_compartment_binding(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
    comp_name: str,
    spec: dict,
    bkill_events: dict,
) -> None:
    pk = params.pk
    bind = params.binding
    traf = params.trafficking  # currently only used for koutTC / koutBC

    drug_ix = spec["drug_ix"].value
    T_ix = spec["T_ix"].value
    B_ix = spec["B_ix"].value
    d3_ix = spec["d3_ix"].value
    d20_ix = spec["d20_ix"].value
    tri_ix = spec["tri_ix"].value

    V = getattr(pk, spec["volume_name"])  # liters

    # Amounts (nmol) -> concentrations (nmol/L)
    A_ab = y[drug_ix]
    A_d3 = y[d3_ix]
    A_d20 = y[d20_ix]
    A_tri = y[tri_ix]

    C_ab = A_ab / V
    C_d3 = A_d3 / V
    C_d20 = A_d20 / V
    C_tri = A_tri / V
  
    # Total receptor concentrations (nmol/L) on live cells
    C_CD3_tot = _receptor_conc_nmolar(y[T_ix], bind.RCD3, V)
    C_CD20_tot = _receptor_conc_nmolar(y[B_ix], bind.RCD20, V)

    # Free receptors (each trimer uses one CD3 and one CD20)
    C_CD3_free = max(C_CD3_tot - C_d3 - C_tri, 0.0)
    C_CD20_free = max(C_CD20_tot - C_d20 - C_tri, 0.0)

    # ---- Elementary reaction rates (nmol/L/day) ----
    # 1) Ab + CD3_free <-> CD3-Ab
    v_on3 = bind.konCD3 * C_ab * C_CD3_free
    v_off3 = bind.koffCD3 * C_d3

    # 2) Ab + CD20_free <-> CD20-Ab
    v_on20 = bind.konCD20 * C_ab * C_CD20_free
    v_off20 = bind.koffCD20 * C_d20

    # 3) CD3-Ab + CD20_free <-> Trimer
    v_tri_from3 = bind.konCD20 * C_d3 * C_CD20_free
    v_tri_to_d3 = bind.koffCD20 * C_tri

    # 4) CD20-Ab + CD3_free <-> Trimer
    v_tri_from20 = bind.konCD3 * C_d20 * C_CD3_free
    v_tri_to_d20 = bind.koffCD3 * C_tri

    # 5) Internalization of dimers
    v_int3 = bind.kintCD3 * C_d3
    v_int20 = bind.kintCD20 * C_d20

    # 6) Loss of dimers/trimers via natural T/B cell turnover
    koutTC = traf.koutTC
    koutBC = traf.koutBC

    # T-cell death removes CD3-Ab and converts trimers into CD20-Ab
    v_Tdeath_d3 = koutTC * C_d3
    v_Tdeath_tri = koutTC * C_tri

    # B-cell death removes CD20-Ab and converts trimers into CD3-Ab
    v_Bdeath_d20 = koutBC * C_d20
    v_Bdeath_tri = koutBC * C_tri

    # 7) Extra loss of trimers due to ATC-mediated B-cell killing
    # Convert B-cell kill rate [cells/day] into nmol/L/day
    # v_kill_cells_per_day must be supplied by B-cell kill submodel
    v_kill_cells_per_day = bkill_events.get(comp_name, 0.0)

    # molecules per trimer = 1 antibody molecule
    nmol_per_trimer = (1.0 / AVOGADRO) * 1e9     # nmol per molecule

    # nmol/day consumed by killing
    v_kill_nmol_per_day = v_kill_cells_per_day * nmol_per_trimer

    # convert nmol/day → nmol/L/day
    v_kill_BC = v_kill_nmol_per_day / V

    # ---- ODEs in concentration space (nmol/L/day) ----

    # Free Ab
    dC_ab = (
        - v_on3 + v_off3
        - v_on20 + v_off20
    )

    # CD3-Ab dimers
    dC_d3 = (
        + v_on3 - v_off3
        - v_tri_from3 + v_tri_to_d3
        - v_int3
        - v_Tdeath_d3
        + v_Bdeath_tri
        + v_kill_BC
    )

    # CD20-Ab dimers
    dC_d20 = (
        + v_on20 - v_off20
        - v_tri_from20 + v_tri_to_d20
        - v_int20
        - v_Bdeath_d20
        + v_Tdeath_tri
    )

    # Trimers
    dC_tri = (
        + v_tri_from3 + v_tri_from20
        - v_tri_to_d3 - v_tri_to_d20
        - v_Tdeath_tri - v_Bdeath_tri
        - v_kill_BC
    )

    # ---- Back to amount space (nmol/day) ----
    dydt[drug_ix] += dC_ab * V
    dydt[d3_ix]   += dC_d3 * V
    dydt[d20_ix]  += dC_d20 * V
    dydt[tri_ix]  += dC_tri * V


def _update_node_binding(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
    spec: dict,
    bkill_events: dict,
    tumor_kill_events: dict,
) -> None:
    """
    Specialized binding for the tumor lymph node compartment:
    - B-cell CD20 binding: CD20_AB_DIMER_NODE, TRIMER_NODE
    - Tumor cell CD20 binding: CD20_AB_DIMER_TUMOR, TRIMER_TUMOR
    - Shared CD3 pool on T cells (T_NODE)
    """

    pk = params.pk
    bind = params.binding
    traf = params.trafficking

    drug_ix = spec["drug_ix"].value
    T_ix = spec["T_ix"].value
    B_ix = spec["B_ix"].value
    d3_ix = spec["d3_ix"].value
    d20_B_ix = spec["d20_ix"].value
    tri_B_ix = spec["tri_ix"].value

    d20_T_ix = spec["d20_tumor_ix"].value
    tri_T_ix = spec["tri_tumor_ix"].value

    V = getattr(pk, spec["volume_name"])  # Vnode [L]

    # --- Amounts (nmol) ---
    A_ab   = y[drug_ix]
    A_d3   = y[d3_ix]
    A_d20B = y[d20_B_ix]
    A_triB = y[tri_B_ix]
    A_d20T = y[d20_T_ix]
    A_triT = y[tri_T_ix]

    # --- Concentrations (nmol/L) ---
    C_ab   = A_ab   / V
    C_d3   = A_d3   / V
    C_d20B = A_d20B / V
    C_triB = A_triB / V
    C_d20T = A_d20T / V
    C_triT = A_triT / V

    # --- Receptor totals (nmol/L) ---
    T_cells     = y[T_ix]
    B_cells     = y[B_ix]
    Tumor_cells = y[StateIx.TUMOR_CELLS_TOTAL]

    C_CD3_tot   = _receptor_conc_nmolar(T_cells, bind.RCD3, V)
    C_CD20B_tot = _receptor_conc_nmolar(B_cells, bind.RCD20, V)
    C_CD20T_tot = _receptor_conc_nmolar(Tumor_cells, bind.RCD20tumor, V)

    # --- Free receptors ---
    C_CD3_free   = max(C_CD3_tot   - C_d3 - C_triB - C_triT, 0.0)
    C_CD20B_free = max(C_CD20B_tot - C_d20B - C_triB,        0.0)
    C_CD20T_free = max(C_CD20T_tot - C_d20T - C_triT,        0.0)

    # ---- Elementary reaction rates (nmol/L/day) ----

    # 1) Ab + CD3_free <-> CD3-Ab
    v_on3  = bind.konCD3 * C_ab * C_CD3_free
    v_off3 = bind.koffCD3 * C_d3

    # 2) Ab + CD20_free (B) <-> CD20-Ab (B)
    v_on20B  = bind.konCD20 * C_ab * C_CD20B_free
    v_off20B = bind.koffCD20 * C_d20B

    # 3) Ab + CD20_free (Tumor) <-> CD20-Ab (Tumor)
    v_on20T  = bind.konCD20 * C_ab * C_CD20T_free
    v_off20T = bind.koffCD20 * C_d20T

    # 4) CD3-Ab + CD20_free (B) <-> Trimer_B
    v_tri_from3_B = bind.konCD20 * C_d3   * C_CD20B_free
    v_tri_to_d3_B = bind.koffCD20 * C_triB

    # 5) CD3-Ab + CD20_free (Tumor) <-> Trimer_T
    v_tri_from3_T = bind.konCD20 * C_d3   * C_CD20T_free
    v_tri_to_d3_T = bind.koffCD20 * C_triT

    # 6) CD20-Ab (B) + CD3_free <-> Trimer_B
    v_tri_from20_B = bind.konCD3 * C_d20B * C_CD3_free
    v_tri_to_d20_B = bind.koffCD3 * C_triB

    # 7) CD20-Ab (Tumor) + CD3_free <-> Trimer_T
    v_tri_from20_T = bind.konCD3 * C_d20T * C_CD3_free
    v_tri_to_d20_T = bind.koffCD3 * C_triT

    # 8) Internalization
    v_int3    = bind.kintCD3  * C_d3
    v_int20B  = bind.kintCD20 * C_d20B
    v_int20T  = bind.kintCD20 * C_d20T

    # 9) Natural cell death effects
    koutTC = traf.koutTC
    koutBC = traf.koutBC

    # T-cell death:
    v_Tdeath_d3   = koutTC * C_d3
    v_Tdeath_triB = koutTC * C_triB
    v_Tdeath_triT = koutTC * C_triT

    # B-cell death:
    v_Bdeath_d20B = koutBC * C_d20B
    v_Bdeath_triB = koutBC * C_triB

    # 10) B-cell killing by ATC in node: breaks Trimer_B -> CD3-Ab
    v_kill_cells_per_day = bkill_events.get("node", 0.0)
    nmol_per_trimer = (1.0 / AVOGADRO) * 1e9
    v_kill_nmol_per_day = v_kill_cells_per_day * nmol_per_trimer
    v_kill_BC = v_kill_nmol_per_day / V    # nmol/L/day

    # 11) Tumor-cell killing by ATC: breaks Trimer_TUMOR -> CD3-Ab
    v_kill_tumor_cells_per_day = tumor_kill_events.get("tumor", 0.0)
    v_kill_tumor_nmol_per_day = v_kill_tumor_cells_per_day * nmol_per_trimer
    v_kill_tumor = v_kill_tumor_nmol_per_day / V

    # ---- ODEs in concentration space (nmol/L/day) ----

    # Free Ab
    dC_ab = (
        - v_on3  + v_off3
        - v_on20B + v_off20B
        - v_on20T + v_off20T
    )

    # CD3-Ab dimers
    dC_d3 = (
        + v_on3 - v_off3
        - v_tri_from3_B - v_tri_from3_T
        + v_tri_to_d3_B + v_tri_to_d3_T
        - v_int3
        - v_Tdeath_d3
        + v_Bdeath_triB
        + v_kill_BC
        + v_kill_tumor    # <-- NEW: tumor trimers broken to CD3-Ab
    )
    # CD20-Ab on B cells
    dC_d20B = (
        + v_on20B - v_off20B
        - v_tri_from20_B + v_tri_to_d20_B
        - v_int20B
        - v_Bdeath_d20B
        + v_Tdeath_triB
    )

    # Trimers on B cells
    dC_triB = (
        + v_tri_from3_B + v_tri_from20_B
        - v_tri_to_d3_B - v_tri_to_d20_B
        - v_Tdeath_triB - v_Bdeath_triB
        - v_kill_BC
    )

    # CD20-Ab on tumor cells
    dC_d20T = (
        + v_on20T - v_off20T
        - v_tri_from20_T + v_tri_to_d20_T
        - v_int20T
        + v_Tdeath_triT
        # (tumor cell death effects on trimers can be added later)
    )

    # Trimers on tumor cells
    dC_triT = (
        + v_tri_from3_T + v_tri_from20_T
        - v_tri_to_d3_T - v_tri_to_d20_T
        - v_Tdeath_triT
        - v_kill_tumor    # <-- NEW: trimers consumed by tumor cell killing
    )


    # ---- Back to amount space (nmol/day) ----
    dydt[drug_ix]    += dC_ab   * V
    dydt[d3_ix]      += dC_d3   * V
    dydt[d20_B_ix]   += dC_d20B * V
    dydt[tri_B_ix]   += dC_triB * V
    dydt[d20_T_ix]   += dC_d20T * V
    dydt[tri_T_ix]   += dC_triT * V

def update_dydt_binding(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
    bkill_events: dict,
    tumor_kill_events: dict,
) -> None:
    """
    Update binding-related entries of dydt in place.

    bkill_events:  {'blood': ..., 'spleen': ..., 'node': ..., 'lymph': ...}
    tumor_kill_events: {'tumor': kill_cells_per_day}
    """
    for comp_name, spec in COMPARTMENTS.items():
        if comp_name == "node":
            _update_node_binding(
                t=t,
                y=y,
                params=params,
                dydt=dydt,
                spec=spec,
                bkill_events=bkill_events,
                tumor_kill_events=tumor_kill_events,
            )
        else:
            _update_compartment_binding(
                t=t,
                y=y,
                params=params,
                dydt=dydt,
                comp_name=comp_name,
                spec=spec,
                bkill_events=bkill_events,
            )
