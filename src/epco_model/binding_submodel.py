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
        d20_ix=StateIx.CD20_AB_DIMER_NODE,
        tri_ix=StateIx.TRIMER_NODE,
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
    spec: dict,
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
    #    (to be wired in once ATC states exist)
    v_kill_BC = 0.0

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


def update_dydt_binding(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
) -> None:
    """
    Update binding-related entries of dydt in place.
    """
    for spec in COMPARTMENTS.values():
        _update_compartment_binding(t, y, params, dydt, spec)
